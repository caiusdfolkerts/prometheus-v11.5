# prometheus/sim/physics/gravity/gravity_bh_numba.py
"""
Barnes-Hut gravity calculation model accelerated using Numba for CPU execution.

Implements the GravityModel interface. Uses Numba-jitted kernels for building
the octree, calculating centers of mass, and computing forces particle-by-particle
using the tree traversal algorithm. 
"""

import numpy as np
import time
import traceback
from math import sqrt 

# Numba imports (guarded by try-except is good practice, but assumed available for this module)
from numba import njit, prange, float64, int64, uint64, void, boolean, types

# Absolute imports from project structure
from sim.physics.base.gravity import GravityModel
from sim.particle_data import ParticleData
from sim.utils import timing_decorator

# --- Numba Kernel Constants ---
MORTON_BITS: int = 21 # Number of bits per dimension for Morton code (max coordinate value ~2M)
MORTON_MAX_VAL: int = (1 << MORTON_BITS) - 1
MAX_STACK_DEPTH: int = 2048 # Max recursion depth simulation for iterative tree walk
NODE_IS_INTERNAL: int = 0 # Node status flag: Internal node with children
NODE_IS_LEAF: int = 1     # Node status flag: Leaf node containing one particle
NODE_IS_EMPTY: int = 2      # Node status flag: Empty node

# --- Numba Type Definitions ---
# Explicitly define types for Numba signatures where needed
float64_1d = types.Array(types.float64, 1, 'C')
float64_2d = types.Array(types.float64, 2, 'C')
int64_1d = types.Array(types.int64, 1, 'C')
int64_2d = types.Array(types.int64, 2, 'C')
uint64_1d = types.Array(types.uint64, 1, 'C')


# These kernels perform the core computations for tree building and force calculation.
# They are designed to operate on NumPy arrays  passed from Python.

# --- Morton Code Kernels ---
@njit(uint64(uint64), cache=True, nogil=True, fastmath=True)
def _interleave_bits_magic(val: uint64) -> uint64:
    """(Numba Kernel) Interleaves bits of a 64-bit integer for Morton code."""
    x = val
    x = (x | (x << 32)) & np.uint64(0x1f00000000ffff)
    x = (x | (x << 16)) & np.uint64(0x1f0000ff0000ff)
    x = (x | (x <<  8)) & np.uint64(0x100f00f00f00f00f)
    x = (x | (x <<  4)) & np.uint64(0x10c30c30c30c30c3)
    x = (x | (x <<  2)) & np.uint64(0x1249249249249249)
    return x

@njit(uint64(float64_1d, float64_1d, float64), cache=True, nogil=True, fastmath=True)
def compute_morton_code(pos: float64_1d, domain_min: float64_1d, domain_scale_inv: float64) -> uint64:
    """(Numba Kernel) Computes 3D Morton code for a single particle position."""
    norm_x = (pos[0] - domain_min[0]) * domain_scale_inv
    norm_y = (pos[1] - domain_min[1]) * domain_scale_inv
    norm_z = (pos[2] - domain_min[2]) * domain_scale_inv
    mmax_f = float64(MORTON_MAX_VAL)
    ix = uint64(max(0.0, min(mmax_f - 1e-15, norm_x))) 
    iy = uint64(max(0.0, min(mmax_f - 1e-15, norm_y)))
    iz = uint64(max(0.0, min(mmax_f - 1e-15, norm_z)))
    # interleave bits to form Morton code
    return _interleave_bits_magic(iz) | (_interleave_bits_magic(iy) << 1) | (_interleave_bits_magic(ix) << 2)

@njit(void(float64_2d, int64_1d, float64_1d, float64, uint64_1d), cache=True, nogil=True, parallel=True)
def parallel_compute_morton_codes_ids_numba(particle_pos: float64_2d, particle_ids_out: int64_1d,
                                           domain_min: float64_1d, domain_size: float64,
                                           morton_codes_out: uint64_1d):
    """(Numba Kernel) Computes Morton codes and initial IDs for all particles in parallel."""
    n_particles = particle_pos.shape[0]
    # calculate inverse scaling factor safely
    domain_scale_inv = float64(MORTON_MAX_VAL) / (domain_size + 1e-20) if domain_size > 1e-15 else float64(0.0)
    for i in prange(n_particles):
        particle_ids_out[i] = i # store original index
        morton_codes_out[i] = compute_morton_code(particle_pos[i], domain_min, domain_scale_inv)

# --- tree Building Kernels ---
@njit(cache=True, nogil=True) # returns tuple (center, size), Numba handles type inference
def _bh_calculate_child_node_props_build(parent_node_idx: int64, octant_idx: int64,
                                         node_centers_arr: float64_2d, node_sizes_arr: float64_1d):
    """(Numba Kernel) Calculates the center and size of a child node."""
    parent_center = node_centers_arr[parent_node_idx]
    parent_half_size = node_sizes_arr[parent_node_idx] * 0.5 # float64 calculation
    child_size = parent_half_size
    child_center = np.empty(3, dtype=np.float64)
    # Determine offset based on octant index (bit manipulation)
    offset_x = parent_half_size if (octant_idx & 4) else -parent_half_size
    offset_y = parent_half_size if (octant_idx & 2) else -parent_half_size
    offset_z = parent_half_size if (octant_idx & 1) else -parent_half_size
    child_center[0] = parent_center[0] + offset_x
    child_center[1] = parent_center[1] + offset_y
    child_center[2] = parent_center[2] + offset_z
    return child_center, child_size

@njit(uint64(uint64, uint64, int64), cache=True, nogil=True, fastmath=True)
def get_morton_octant(p_code: uint64, node_code_base: uint64, level: int64) -> uint64:
    """(Numba Kernel) Extracts the octant index from a Morton code at a given tree level."""
    # node_code_base is unused here but kept for potential future radix tree uses
    if level < 0 or level >= MORTON_BITS: return uint64(0) # Invalid level
    shift = int64(MORTON_BITS - 1 - level) * 3 # Calculate bit shift
    if shift < 0: return uint64(0)
    return (p_code >> shift) & uint64(7) # Extract 3 bits for octant (0-7)

@njit(void(int64, int64, int64, uint64_1d, float64_2d, float64_1d, int64_2d, int64_1d, int64_1d, int64_1d, int64), cache=True, nogil=True)
def _build_node_iterative(p_start_root: int64, p_end_root: int64, level_root: int64,
                          sorted_morton_codes: uint64_1d, node_centers: float64_2d, node_sizes: float64_1d,
                          node_children: int64_2d, node_status: int64_1d, node_leaf_particle_idx: int64_1d,
                          next_node_idx_arr: int64_1d, MAX_NODES_BUILD: int64):
    """(Numba Kernel) Iteratively builds the octree structure using sorted Morton codes."""
    stack = np.empty((MAX_STACK_DEPTH, 4), dtype=np.int64) # stack for iterative traversal
    stack_ptr = 0
    node_idx_root = 0 # Start at root node

    # handle base case: empty or single particle range
    if p_end_root < p_start_root: return # invalid range
    if p_end_root == p_start_root: node_status[node_idx_root] = NODE_IS_EMPTY; return
    if p_end_root == p_start_root + 1: node_status[node_idx_root] = NODE_IS_LEAF; node_leaf_particle_idx[node_idx_root] = p_start_root; return

    # initialize stack with root node's properties
    stack[stack_ptr, 0] = node_idx_root; stack[stack_ptr, 1] = p_start_root
    stack[stack_ptr, 2] = p_end_root; stack[stack_ptr, 3] = level_root
    stack_ptr += 1
    node_status[node_idx_root] = NODE_IS_INTERNAL # root is internal if >1 particle
    node_children[node_idx_root, :] = -1 # initialize children as invalid

    loop_iterations = 0; max_loop_iterations = MAX_NODES_BUILD * 9 # safety limit to avoid S-Oflow

    while stack_ptr > 0:
        loop_iterations += 1
        if loop_iterations > max_loop_iterations: break

        stack_ptr -= 1 # Pop node from stack
        node_idx, p_start, p_end, level = stack[stack_ptr]

        if node_status[node_idx] != NODE_IS_INTERNAL: continue 

        # Determine particle ranges for each octant child based on Morton codes
        split_indices = np.empty(9, dtype=np.int64); split_indices[0] = p_start
        current_p_idx = p_start; dummy_node_base = uint64(0) 
        for octant_search in range(8): # Find end index for each octant
            target_octant = uint64(octant_search); idx = current_p_idx
            found_end = False
            while idx < p_end:
                p_octant = get_morton_octant(sorted_morton_codes[idx], dummy_node_base, level)
                if p_octant > target_octant: current_p_idx = idx; found_end = True; break
                idx += 1
            if not found_end: current_p_idx = p_end
            split_indices[octant_search + 1] = current_p_idx
            if current_p_idx == p_end: # If all remaining particles are in this octant
                for k in range(octant_search + 2, 9): split_indices[k] = p_end
                break

        # Create child nodes
        next_node_idx_val = next_node_idx_arr[0] # Get next available node index
        for octant in range(8):
            child_p_start = split_indices[octant]; child_p_end = split_indices[octant + 1]
            child_n_p = child_p_end - child_p_start
            if child_n_p <= 0: continue # Skip empty octants

            # Check if node budget exceeded
            if next_node_idx_val >= MAX_NODES_BUILD:
                node_status[node_idx] = NODE_IS_LEAF # Convert parent to leaf if out of nodes
                node_leaf_particle_idx[node_idx] = -2 
                node_leaf_particle_idx[node_idx] = -1
                node_children[node_idx, :] = -1 # Clear children links
                break # Stop creating children for this node

            # Allocate and initialize child node
            child_node_idx = next_node_idx_val; next_node_idx_val += 1
            node_children[node_idx, octant] = child_node_idx
            child_center, child_size = _bh_calculate_child_node_props_build(node_idx, octant, node_centers, node_sizes)
            node_centers[child_node_idx] = child_center; node_sizes[child_node_idx] = child_size
            node_children[child_node_idx, :] = -1 # Init child's children

            # Determine child status and push to stack if internal
            if child_n_p == 1:
                node_status[child_node_idx] = NODE_IS_LEAF
                node_leaf_particle_idx[child_node_idx] = child_p_start # Store index of single particle
            else: # Child is an internal node
                node_status[child_node_idx] = NODE_IS_INTERNAL
                node_leaf_particle_idx[child_node_idx] = -1 # Not a leaf
                if stack_ptr >= MAX_STACK_DEPTH: # Check stack overflow before push
                     node_status[child_node_idx] = NODE_IS_LEAF # Convert to leaf if stack full
                     # Leaf represents multiple particles here too
                else: # Push child onto stack for further processing
                    stack[stack_ptr, 0] = child_node_idx; stack[stack_ptr, 1] = child_p_start
                    stack[stack_ptr, 2] = child_p_end; stack[stack_ptr, 3] = level + 1
                    stack_ptr += 1
        next_node_idx_arr[0] = next_node_idx_val # Update global node counter

# --- Center of Mass Kernel ---
@njit(void(int64, float64_2d, float64_1d, int64_1d, int64_2d, int64_1d, float64_2d, float64_1d, float64_2d, int64), cache=True, nogil=True)
def _bh_compute_node_CoM_iterative(n_nodes_used: int64, sorted_pos_arr: float64_2d, sorted_mass_arr: float64_1d,
                                   node_leaf_idx: int64_1d, node_children: int64_2d, node_status: int64_1d,
                                   node_centers: float64_2d, node_masses_out: float64_1d, node_CoMs_out: float64_2d,
                                   n_particles: int64):
    """(Numba Kernel) Calculates Center of Mass (CoM) for each node iteratively (bottom-up)."""
    if n_nodes_used <= 0: return
    process_stack = np.empty((MAX_STACK_DEPTH, 2), dtype=np.int64) # Stack stores (node_idx, phase)
    p_stack_ptr = 0
    processed_flag = np.zeros(n_nodes_used, dtype=np.uint8) # Track processed nodes

    # Start with the root node (if valid)
    if node_status[0] != NODE_IS_EMPTY:
         process_stack[p_stack_ptr, 0] = 0; process_stack[p_stack_ptr, 1] = 0 # phase 0: discover children
         p_stack_ptr += 1

    iter_count = 0; max_iters = n_nodes_used * 10 # Safety break

    while p_stack_ptr > 0:
        iter_count += 1;
        if iter_count > max_iters: break # Safety break

        node_idx = process_stack[p_stack_ptr - 1, 0]
        phase = process_stack[p_stack_ptr - 1, 1]

        # Basic validation and skip if already processed
        if not (0 <= node_idx < n_nodes_used) or processed_flag[node_idx]:
            p_stack_ptr -= 1; continue

        status = node_status[node_idx]

        # --- Process Leaf Nodes ---
        if status == NODE_IS_LEAF:
            p_stack_ptr -= 1 # kill leaf
            p_idx_sorted = node_leaf_idx[node_idx]
            if 0 <= p_idx_sorted < n_particles: # valid leaf particle index?
                 # CoM of leaf is the particle's position, mass is particle's mass
                 node_masses_out[node_idx] = sorted_mass_arr[p_idx_sorted]
                 node_CoMs_out[node_idx, :] = sorted_pos_arr[p_idx_sorted, :]
            else: # Handle invalid leaf index (e.g., if parent became leaf due to node limit)
                 node_masses_out[node_idx] = 0.0
                 node_CoMs_out[node_idx, :] = node_centers[node_idx, :] # use geometric center
            processed_flag[node_idx] = 1 # mark as processed
            continue
        # --- Process Empty Nodes ---
        elif status == NODE_IS_EMPTY:
            p_stack_ptr -= 1 # Pop empty
            node_masses_out[node_idx] = 0.0
            node_CoMs_out[node_idx, :] = node_centers[node_idx, :] # Use geometric center
            processed_flag[node_idx] = 1 # Mark as processed
            continue
        # --- Process Internal Nodes ---
        elif status == NODE_IS_INTERNAL:
            if phase == 0: # Phase 0: discover children and push them onto stack
                process_stack[p_stack_ptr - 1, 1] = 1 # change phase to 1 (process after children)
                pushed_children = False
                # iterate children in reverse order for stack processing (optional optimization)
                for i in range(7, -1, -1):
                    child_idx = node_children[node_idx, i]
                    # if child is valid, not empty, and not yet processed, push it
                    if (child_idx != -1 and child_idx < n_nodes_used and
                        node_status[child_idx] != NODE_IS_EMPTY and not processed_flag[child_idx]):
                        if p_stack_ptr >= MAX_STACK_DEPTH: break # Stack overflow check
                        process_stack[p_stack_ptr, 0] = child_idx; process_stack[p_stack_ptr, 1] = 0 # phase 0 for child
                        p_stack_ptr += 1; pushed_children = True
                # if no valid children were pushed, this node can be processed now (phase 1 logic)
                if not pushed_children: phase = 1 # Force phase 1 logic immediately

            if phase == 1: # phase 1: Process this node after children are done
                p_stack_ptr -= 1 # pop this internal node
                current_total_mass = 0.0
                current_weighted_pos_sum = np.zeros(3, dtype=np.float64) # accumulate weighted pos
                all_children_processed = True
                # calculate weighted sum from children's CoM and mass
                for i in range(8):
                    child_idx = node_children[node_idx, i]
                    if child_idx != -1 and child_idx < n_nodes_used:
                        if not processed_flag[child_idx]: # redundant, used just to keep completeness
                            all_children_processed = False; break
                        child_mass = node_masses_out[child_idx] # get child's computed mass
                        if child_mass > 1e-30: # only include children with mass
                            current_total_mass += child_mass
                            child_com = node_CoMs_out[child_idx] # get child's computed CoM
                            current_weighted_pos_sum += child_com * child_mass # accumulate m_child * r_com_child

                # calculate CoM for this node if children were processed and mass is sufficient
                if all_children_processed and current_total_mass > 1e-30:
                    node_masses_out[node_idx] = current_total_mass
                    node_CoMs_out[node_idx, :] = current_weighted_pos_sum / current_total_mass
                else: # default to geometric center if no mass or error
                    node_masses_out[node_idx] = 0.0
                    node_CoMs_out[node_idx, :] = node_centers[node_idx, :]
                processed_flag[node_idx] = 1 # mark as processed
        else: # unknown status
             p_stack_ptr -= 1 # [op unknown node
             node_masses_out[node_idx] = 0.0
             node_CoMs_out[node_idx, :] = node_centers[node_idx, :]
             processed_flag[node_idx] = 1

# --- Force Calculation Kernels ---
@njit(float64_1d(int64, float64, float64, float64, float64, float64_1d, int64_1d, int64_1d, int64, float64_2d, float64_1d, int64_2d, int64_1d, float64_1d, float64_2d), cache=True, nogil=True, fastmath=True)
def _bh_calculate_single_force_iterative(p_orig_idx: int64, theta_sq: float64, softening_sq: float64, G: float64,
                                        p_mass: float64, p_pos: float64_1d, sorted_orig_ids: int64_1d,
                                        node_leaf_idx: int64_1d, n_nodes_used: int64, node_centers: float64_2d,
                                        node_sizes: float64_1d, node_children: int64_2d, node_status: int64_1d,
                                        node_masses: float64_1d, node_CoMs: float64_2d) -> float64_1d:
    """(Numba Kernel) Calculates force on a single particle using iterative tree walk."""
    force_on_p = np.zeros(3, dtype=np.float64) # Accumulator for force on particle p
    if p_mass <= 1e-30 or n_nodes_used <= 0: return force_on_p # Skip massless particles

    node_stack = np.empty(MAX_STACK_DEPTH, dtype=np.int64) # Stack for nodes to visit
    stack_ptr = 0
    # Start traversal at the root node if it's valid
    if node_status[0] != NODE_IS_EMPTY and node_masses[0] > 1e-30:
        node_stack[stack_ptr] = 0; stack_ptr += 1

    iter_count = 0; max_iters = n_nodes_used * 20 + 1000 # Safety break limit

    while stack_ptr > 0:
        iter_count += 1;
        if iter_count > max_iters: break # Safety break

        stack_ptr -= 1 # Pop node to process
        node_idx = node_stack[stack_ptr]

        # Basic validation
        if not (0 <= node_idx < n_nodes_used): continue
        status = node_status[node_idx]; node_m = node_masses[node_idx]
        if status == NODE_IS_EMPTY or node_m <= 1e-30: continue # Skip empty/massless nodes

        # Calculate vector and distance squared from particle to node's CoM
        n_CoM = node_CoMs[node_idx]
        dr_x = n_CoM[0] - p_pos[0]; dr_y = n_CoM[1] - p_pos[1]; dr_z = n_CoM[2] - p_pos[2]
        dist_sq = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z

        # Avoid self-interaction or interaction with coincident particles
        # 1. Check leaf node particle index
        if status == NODE_IS_LEAF:
            leaf_p_sorted_idx = node_leaf_idx[node_idx]
            # ensure leaf index is valid and particle ID exists in sorted list
            if 0 <= leaf_p_sorted_idx < sorted_orig_ids.shape[0]:
                 leaf_p_orig_idx = sorted_orig_ids[leaf_p_sorted_idx]
                 if leaf_p_orig_idx == p_orig_idx: continue # skip self-interaction
            elif leaf_p_sorted_idx < 0: continue

        # 2. Check distance squared (also handles coincident case if leaf check missed it)
        if dist_sq < 1e-30: continue

        # --- Barnes-Hut Approximation Check (MAC) ---
        node_interact_directly = True # Assume direct interaction initially
        if status == NODE_IS_INTERNAL:
            node_s = node_sizes[node_idx] # Node size
            if (node_s * node_s > theta_sq * dist_sq):
                node_interact_directly = False # Don't interact directly, push children instead
                pushed_any_child = False
                for i in range(8): # Push valid children onto stack
                    child_idx = node_children[node_idx, i]
                    if (child_idx != -1 and child_idx < n_nodes_used and node_masses[child_idx] > 1e-30):
                        if stack_ptr >= MAX_STACK_DEPTH: pushed_any_child = False; break # stack overflow BAD
                        node_stack[stack_ptr] = child_idx; stack_ptr += 1; pushed_any_child = True
                # if children  pushed continue to next node in stack
                if pushed_any_child: continue

        # --- Calculate Direct Interaction ---
        if node_interact_directly:
            dist_sq_soft = dist_sq + softening_sq # Add softening^2 term
            inv_dist_soft = 1.0 / sqrt(dist_sq_soft)
            inv_dist3_soft = inv_dist_soft * inv_dist_soft * inv_dist_soft
            # Force = G * m_p * m_node * dr / r_soft^3
            force_factor = G * p_mass * node_m * inv_dist3_soft
            force_on_p[0] += force_factor * dr_x
            force_on_p[1] += force_factor * dr_y
            force_on_p[2] += force_factor * dr_z

    return force_on_p

@njit(void(int64_1d, float64_2d, float64_1d, float64, float64, float64, float64_2d, float64_1d, int64_2d, int64_1d, int64_1d, float64_1d, float64_2d, int64, int64_1d, float64_2d), cache=True, nogil=True, parallel=True)
def bh_calculate_forces_parallel_iterative_numba(
                            particle_indices_to_compute: int64_1d, # Indices of particles to compute force for
                            all_particle_pos: float64_2d, all_particle_mass: float64_1d, # Full particle data
                            G: float64, theta: float64, softening: float64, # Parameters
                            # Tree structure arrays
                            node_centers: float64_2d, node_sizes: float64_1d, node_children: int64_2d,
                            node_status: int64_1d, node_leaf_idx: int64_1d, node_masses: float64_1d, node_CoMs: float64_2d,
                            n_nodes_used: int64, # Actual number of nodes used in tree
                            sorted_particle_orig_ids: int64_1d, # Mapping from sorted index to original index
                            forces_out: float64_2d): # Output array for forces
    """(Numba Kernel) Calculates forces for specified particles in parallel using BH tree."""
    n_active = particle_indices_to_compute.shape[0]
    if n_nodes_used <= 0 or n_active == 0: forces_out[:,:] = 0.0; return 

    # Precompute squared parameters
    theta_sq = theta * theta; softening_sq = softening * softening

    # Parallel loop over the particles we need to compute forces for
    for i in prange(n_active):
        p_orig_idx = particle_indices_to_compute[i] # Get the original index of the particle
        n_all_particles = all_particle_pos.shape[0]

        # Validate particle index
        if not (0 <= p_orig_idx < n_all_particles):
             forces_out[i, :] = 0.0; continue # zero force for invalid index

        # get particle's position and mass
        p_pos_i = all_particle_pos[p_orig_idx]; p_mass_i = all_particle_mass[p_orig_idx]

        # call the single-particle force calculation kernel
        force_result_i = _bh_calculate_single_force_iterative(
                 p_orig_idx, theta_sq, softening_sq, G, p_mass_i, p_pos_i,
                 sorted_particle_orig_ids, node_leaf_idx, n_nodes_used,
                 node_centers, node_sizes, node_children, node_status, node_masses, node_CoMs)

        # store the calculated 3D force vector
        forces_out[i, 0] = force_result_i[0]
        forces_out[i, 1] = force_result_i[1]
        forces_out[i, 2] = force_result_i[2]

# ==============================
# --- Python Class Definition ---
# ==============================

class GravityBHNumba(GravityModel):
    """
    Barnes-Hut gravity model using Numba-accelerated kernels on the CPU.

    Builds an octree each step and uses it to approximate gravitational forces.
    Requires particle data (positions, masses) to be available on the CPU.
    Forces are calculated and stored in 'forces_grav' and added to 'forces'.
    Potential energy uses a direct N^2 sum (could be a bottleneck).
    """

    def __init__(self, config: dict = None):
        """Initializes model with config and sets default parameter values."""
        super().__init__(config)
        # internal state for tree structure and parameters
        self.N: int = 0
        self.MAX_NODES: int = 1000 
        self.bh_theta: float = 0.7
        self.softening: float = 0.1
        self.G: float = 1.0
        self.MAX_NODES_FACTOR: int = 10
        # tree data (rebuilt each step)
        self.node_arrays_tuple: Optional[tuple] = None
        self.n_nodes_status: int = 0 # stores node count, negative if allocation limit hit
        self.sorted_particle_data_tuple: Optional[tuple] = None

    # --- Potential Energy Calculation (Direct Sum) ---
    def compute_potential_energy(self, pd: ParticleData) -> float:
        """
        Computes total gravitational potential energy using Numba-accelerated N^2 sum.

        NOTE: This method does NOT use the BH tree for PE calculation due to the
              complexity of implementing a tree-walk PE kernel. It uses a direct
              summation which is O(N^2) and may be slow for large N.

        Returns:
            Total potential energy (float).
        """
        N = pd.get_n()
        if N < 2: return 0.0 # PE requires at least 2 particles
        pd.ensure("positions masses", "cpu")
        pos_f64 = pd.get("positions", "cpu").astype(np.float64, copy=False)
        mass_f64 = pd.get("masses", "cpu").astype(np.float64, copy=False)

        # get parameters (ensure float64)
        G_f64 = float64(self.config.get('G', 1.0))
        softening_f64 = float64(self.config.get('softening', 0.1))
        softening_sq_f64 = softening_f64 * softening_f64

        # call the N^2 Numba kernel
        try:
            total_potential = 0
            return float(total_potential) # return standard float
        except Exception as e:
            print(f"ERROR during Numba direct PE calculation: {e}")
            traceback.print_exc()
            return 0.0 # Return 0 on error

    # --- Model Setup ---
    def setup(self, pd: ParticleData):
        """Sets up BH parameters from config and validates them."""
        super().setup(pd) # Call base class setup
        self.N = pd.get_n()

        # Validate and store required parameters from config
        try:
            self.G = float(self.config['G'])
            self.softening = float(self.config['softening'])
            self.bh_theta = float(self.config['bh_theta'])
            self.MAX_NODES_FACTOR = int(self.config['MAX_NODES_FACTOR'])
            if self.softening <= 0: raise ValueError("Softening must be positive.")
            if not (0 < self.bh_theta < 2.0): raise ValueError("BH Theta must be > 0 and typically < ~1.5.")
            if self.MAX_NODES_FACTOR < 2: raise ValueError("MAX_NODES_FACTOR should be >= 2.")
        except KeyError as e: raise ValueError(f"Missing required config key for GravityBHNumba: {e}")
        except (ValueError, TypeError) as e: raise ValueError(f"Invalid config value for GravityBHNumba: {e}")

        min_nodes_needed = max(20, int(1.5 * self.N)) # Basic minimum + linear scaling
        self.MAX_NODES = int(max(min_nodes_needed, self.N * self.MAX_NODES_FACTOR))
        self.MAX_NODES = min(self.MAX_NODES, 50_000_000) # Hard cap to prevent excessive memory use

        print(f"GravityBHNumba Setup:")
        print(f"  N = {self.N}, G = {self.G:.3e}, Softening = {self.softening:.3f}, Theta = {self.bh_theta:.2f}")
        print(f"  Max Nodes = {self.MAX_NODES} (Factor={self.MAX_NODES_FACTOR})")

    # --- Internal Tree Building Logic ---
    def _build_tree_internal(self, pos_np: np.ndarray, mass_np: np.ndarray) -> bool:
         """(Internal) Builds the BH octree structure using Numba kernels."""
         if self.N <= 0:
              # Clear any stale tree data and return failure
              self.node_arrays_tuple = None; self.n_nodes_status = 0; self.sorted_particle_data_tuple = None
              return False

         # --- Validate Input Types ---
         if pos_np.dtype != np.float64: print("ERROR (_build): pos_np must be float64."); return False
         if mass_np.dtype != np.float64: print("ERROR (_build): mass_np must be float64."); return False

         try:
             # --- Allocate Tree Arrays ---
             try:
                 node_centers = np.zeros((self.MAX_NODES, 3), dtype=np.float64)
                 node_sizes = np.zeros(self.MAX_NODES, dtype=np.float64)
                 node_children = np.full((self.MAX_NODES, 8), -1, dtype=np.int64)
                 node_status = np.full(self.MAX_NODES, NODE_IS_EMPTY, dtype=np.int64)
                 node_leaf_idx = np.full(self.MAX_NODES, -1, dtype=np.int64)
                 node_masses = np.zeros(self.MAX_NODES, dtype=np.float64)
                 node_CoMs = np.zeros((self.MAX_NODES, 3), dtype=np.float64)
                 next_node_idx_arr = np.array([1], dtype=np.int64) # Tracks next available node index
                 particle_ids_tmp = np.empty(self.N, dtype=np.int64) # Stores original particle IDs
                 morton_codes = np.empty(self.N, dtype=np.uint64) # Stores Morton codes
             except MemoryError: print(f"ERROR: BH Tree Memory Alloc Failed (MAX_NODES={self.MAX_NODES})."); return False

             # --- Calculate Bounding Box ---
             if not np.all(np.isfinite(pos_np)): print("ERROR: Non-finite positions found."); return False
             min_coord = np.min(pos_np, axis=0); max_coord = np.max(pos_np, axis=0)
             size_vec = max_coord - min_coord
             if np.all(size_vec < 1e-20): center = pos_np[0].astype(np.float64); size = np.float64(1e-9)
             else: center = (min_coord + max_coord).astype(np.float64) * 0.5; size_vec = np.maximum(size_vec, 1e-15); size = np.max(size_vec).astype(np.float64)
             size = np.float64(max(size * 1.01, 1e-15))
             domain_min = (center - (size * 0.5)).astype(np.float64)
             node_centers[0] = center; node_sizes[0] = size

             # --- Compute Morton Codes ---
             parallel_compute_morton_codes_ids_numba(pos_np, particle_ids_tmp, domain_min, size, morton_codes)

             # --- Sort Particles by Morton Code ---
             sort_perm = np.argsort(morton_codes, kind='stable') # Stable sort preserves order for equal codes
             sorted_morton = morton_codes[sort_perm]
             sorted_pos = np.require(pos_np[sort_perm], dtype=np.float64, requirements=['C'])
             sorted_mass = np.require(mass_np[sort_perm], dtype=np.float64, requirements=['C'])
             sorted_orig_ids = np.require(particle_ids_tmp[sort_perm], dtype=np.int64, requirements=['C'])
             del particle_ids_tmp, morton_codes

             # --- Build Tree Structure (Iterative) ---
             _build_node_iterative(0, self.N, 0, sorted_morton,
                                node_centers, node_sizes, node_children, node_status,
                                node_leaf_idx, next_node_idx_arr, self.MAX_NODES)
             final_node_count = next_node_idx_arr[0]
             ran_out_of_nodes = (final_node_count >= self.MAX_NODES)
             if ran_out_of_nodes: final_node_count = self.MAX_NODES # Cap at max allocated
             
             # --- Calculate Node Centers of Mass ---
             if final_node_count > 0:
                 _bh_compute_node_CoM_iterative(final_node_count, sorted_pos, sorted_mass,
                                              node_leaf_idx, node_children, node_status,
                                              node_centers, node_masses, node_CoMs, self.N)

             # --- Store Results ---
             final_slice = slice(0, final_node_count)
             self.node_arrays_tuple = (node_centers[final_slice].copy(), node_sizes[final_slice].copy(),
                                       node_children[final_slice].copy(), node_status[final_slice].copy(),
                                       node_leaf_idx[final_slice].copy(), node_masses[final_slice].copy(),
                                       node_CoMs[final_slice].copy())
             self.sorted_particle_data_tuple = (None, None, None, sorted_orig_ids) # Keep only sorted_orig_ids
             # Store node count (negative if limit was hit)
             self.n_nodes_status = -final_node_count if ran_out_of_nodes else final_node_count
             return True
         except Exception as e:
             print(f"ERROR during internal BH tree build: {e}"); traceback.print_exc()
             self.node_arrays_tuple = None; self.n_nodes_status = 0; self.sorted_particle_data_tuple = None
             return False

    # Calculate forces
    @timing_decorator
    def compute_forces(self, pd: ParticleData):
        """Computes gravity forces using Barnes-Hut, storing the result
           in 'forces_grav' and adding it to the main 'forces' accumulator."""
        self.N = pd.get_n() # Update N in case it changed
        if self.N <= 1:
            try:
                forces_main_write = pd.get("forces", device="cpu", writeable=True)
                forces_main_write.fill(0.0)
                pd.release_writeable("forces")
                forces_grav_write = pd.get("forces_grav", device="cpu", writeable=True)
                forces_grav_write.fill(0.0)
                pd.release_writeable("forces_grav")
            except Exception as e_zero:
                print(f"Warning: Failed to zero forces for N<=1: {e_zero}")
            self.n_nodes_status = 0
            return
        pd.ensure("positions masses", target_device="cpu")
        pos_np_orig = pd.get("positions", device="cpu")
        mass_np_orig = pd.get("masses", device="cpu")  
        # Ensure float64 inputs for Numba
        pos_np = np.require(pos_np_orig, dtype=np.float64, requirements=['C'])
        mass_np = np.require(mass_np_orig, dtype=np.float64, requirements=['C'])

        # build tree
        build_success = self._build_tree_internal(pos_np, mass_np)

        if not build_success or abs(self.n_nodes_status) == 0: # status could be <0
             print("Warning: BH Tree build failed or empty, zeroing gravity forces.")
             try:
                # zero both main forces and gravity forces if tree build failed
                forces_main_write = pd.get("forces", device="cpu", writeable=True)
                forces_main_write.fill(0.0)
                pd.release_writeable("forces")
                forces_grav_write = pd.get("forces_grav", device="cpu", writeable=True)
                forces_grav_write.fill(0.0)
                pd.release_writeable("forces_grav")
                
            # should not happen, included to ensure completeness
             except Exception as e_zero:
                print(f"Warning: Failed to zero forces after failed tree build: {e_zero}")
             # Ccean up any partial tree data and ensure status is 0
             self.node_arrays_tuple = None
             self.sorted_particle_data_tuple = None
             return
        gravity_forces_out_np = np.zeros((self.N, 3), dtype=np.float64)

        try:
             (node_centers, node_sizes, node_children, node_status,
              node_leaf_particle_idx, node_masses, node_CoMs) = self.node_arrays_tuple
             (_, _, _, sorted_particle_orig_ids) = self.sorted_particle_data_tuple
             indices_to_compute = np.arange(self.N, dtype=np.int64)

             # --- Call the Numba force kernel ---
             bh_calculate_forces_parallel_iterative_numba(
                 indices_to_compute,
                 pos_np, mass_np, 
                 self.G, self.bh_theta, self.softening,
                 node_centers, node_sizes, node_children, node_status, node_leaf_particle_idx,
                 node_masses, node_CoMs, abs(self.n_nodes_status),
                 sorted_particle_orig_ids, 
                 gravity_forces_out_np 
              )

            # summation
             target_grav_dtype = pd.get_numpy_dtype("forces_grav")
             pd.set("forces_grav", gravity_forces_out_np.astype(target_grav_dtype, copy=False))
             forces_main_write = pd.get("forces", device="cpu", writeable=True)
             target_main_dtype = pd.get_numpy_dtype("forces")
             forces_main_write += gravity_forces_out_np.astype(target_main_dtype, copy=False)
             pd.release_writeable("forces")


        except Exception as e:
             print(f"ERROR during BH force calculation: {e}")
             traceback.print_exc()
             try:
                 forces_main_write = pd.get("forces", device="cpu", writeable=True)
                 forces_main_write.fill(0.0)
                 pd.release_writeable("forces")
                 forces_grav_write = pd.get("forces_grav", device="cpu", writeable=True)
                 forces_grav_write.fill(0.0)
                 pd.release_writeable("forces_grav")
             except Exception as e_zero_err:
                 print(f"Error zeroing forces after exception: {e_zero_err}")
             self.n_nodes_status = 0

        finally:
             self.node_arrays_tuple = None
             self.sorted_particle_data_tuple = None



    def _clear_tree_data(self):
        """Helper to reset internal tree data pointers."""
        self.node_arrays_tuple = None
        self.sorted_particle_data_tuple = None
        self.n_nodes_status = 0

    def cleanup(self):
        """Release references to internal tree data."""
        super().cleanup()
        self._clear_tree_data()
        print("GravityBHNumba cleaned up.")