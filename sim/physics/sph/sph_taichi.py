# prometheus/sim/physics/sph/sph_taichi.py
import numpy as np
import time
import traceback
from sim.physics.base.sph import SPHModel
from sim.particle_data import ParticleData
from sim.utils import timing_decorator
from math import pi

from sim.constants import CONST_a_rad

# try to import taichi, which is needed for this SPH implementation
try:
    import taichi as ti
    import taichi.math as tim
    from taichi.lang.util import to_numpy_type
    HAVE_TAICHI = True
# if taichi isn't found, create dummy versions so the rest of the file can be imported
except ImportError:
    HAVE_TAICHI = False
    class _DummyTiType: pass
    class ti:
        f32 = _DummyTiType()
        f64 = _DummyTiType()
        i32 = _DummyTiType()
        template = _DummyTiType()
        kernel = lambda func: func
        func = lambda func: func
        field = lambda *args, **kwargs: None
        Vector = lambda *args, **kwargs: None
        types = lambda: None
        types.vector = lambda *args, **kwargs: None
        grouped = lambda x: x
        ndrange = lambda *args: range(1) 
        floor = np.floor
        max = np.maximum
        min = np.minimum
        abs = np.abs
        sync = lambda: None
        atomic_add = lambda x, y: x + y 
        loop_config = lambda *args, **kwargs: lambda func: func 

    class tim:
        max = np.maximum
        min = np.minimum
        sqrt = np.sqrt
        dot = lambda x, y: np.dot(x, y) 

    to_numpy_type = lambda x: np.float32 

# figure out the default float and integer types taichi is using (e.g., f32, f64, i32)
DEFAULT_FP = ti.f64 
DEFAULT_IP = ti.i32 
VEC3 = None         

if HAVE_TAICHI:
    try:
        # gets the current configuration if taichi has been initialized
        cfg = ti.lang.impl.current_cfg() 
        if cfg:
            DEFAULT_FP = cfg.default_fp
            DEFAULT_IP = cfg.default_ip
        else:
            print("Warning: SPHTaichi Taichi config not found after import. Using default FP=f64, IP=i32.")
            DEFAULT_FP = ti.f64
            DEFAULT_IP = ti.i32
    except AttributeError:
        print("Warning: SPHTaichi Taichi not initialized at module import. Using default FP=f64, IP=i32.")
        DEFAULT_FP = ti.f64
        DEFAULT_IP = ti.i32
    except Exception as e:
         print(f"Warning: SPHTaichi error getting Taichi config at import: {e}. Using default FP=f64, IP=i32.")
         DEFAULT_FP = ti.f64
         DEFAULT_IP = ti.i32

    # define a 3D vector type using the default float precision
    VEC3 = ti.types.vector(3, DEFAULT_FP)
    print(f"SPHTaichi using Taichi FP: {DEFAULT_FP}, IP: {DEFAULT_IP}")

    # this kernel calculates the density for each SPH particle
    @ti.kernel
    def calculate_density_sph_ti_kernel(
        positions_field: ti.template(), masses_field: ti.template(),
        densities_field: ti.template(), # this field will store the output densities
        grid_dim_vec: ti.types.vector(3, ti.i32), # dimensions of the neighbor search grid
        grid_num_particles_field: ti.template(), # stores how many particles are in each grid cell
        grid_particle_indices_field: ti.template(), # stores the indices of particles in each grid cell
        domain_min_vec_arg: ti.template(), # the minimum corner of the simulation domain
        N: int, h: float, h_sq: float, density_floor: float, # simulation parameters
        W0_poly6: float, C_poly6: float, # precomputed SPH kernel constants (poly6 kernel used here)
        grid_inv_cell_size_val: float, index_dtype: ti.template(), # grid helper values
        FP: ti.template() # the float type (e.g., f32, f64) taichi is using
    ):
        # define some small numbers to avoid numerical issues, using the correct float type
        EPSILON: FP = ti.cast(1e-9, FP) * h          
        EPSILON_SQ: FP = EPSILON * EPSILON
        four_h_sq: FP = ti.cast(4.0, FP) * h_sq      
        # loop over all particles (taichi parallelizes this automatically)
        for i in range(N):
            pos_i = positions_field[i]
            mass_i = masses_field[i]
            # start the density sum with the particle's self-contribution using the poly6 kernel value at r=0
            density_sum: FP = ti.cast(W0_poly6, FP) * mass_i 

            # only calculate density if the particle has mass
            if mass_i > ti.cast(1e-25, FP): 
                # figure out which grid cell particle 'i' is in
                relative_pos_i = pos_i - domain_min_vec_arg
                cell_i = ti.floor(relative_pos_i * grid_inv_cell_size_val).cast(index_dtype)
                # make sure the cell index is within the grid bounds
                cell_i = tim.max(ti.cast(0, index_dtype), tim.min(cell_i, grid_dim_vec - ti.cast(1, index_dtype))) 

                # loop over the 3x3x3 block of neighboring grid cells (including the particle's own cell)
                for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    neighbor_cell = (cell_i + offset + grid_dim_vec) % grid_dim_vec # handle periodic boundaries
                    num_in_cell = grid_num_particles_field[neighbor_cell]
                    # loop over all particles 'j' in the neighbor cell
                    for k in range(num_in_cell):
                        j = grid_particle_indices_field[neighbor_cell, k]
                        if i == j: continue # don't interact a particle with itself
                        mass_j = masses_field[j]
                        # only consider neighbors with mass
                        if mass_j > ti.cast(1e-25, FP): 
                            pos_j = positions_field[j]
                            rij = pos_i - pos_j
                            dist_sq = rij.dot(rij)
                            # check if the neighbor is within 2*h (optimization) and not exactly coincident
                            if dist_sq < four_h_sq and dist_sq > EPSILON_SQ:
                                hsq_fp: FP = ti.cast(h_sq, FP)
                                # if within the kernel radius h
                                if dist_sq < hsq_fp:
                                    # calculate the Poly6 kernel value W(rij, h)
                                    hsq_minus_distsq: FP = hsq_fp - dist_sq
                                    kernel_val: FP = ti.cast(C_poly6, FP) * (hsq_minus_distsq**ti.cast(3.0, FP))
                                    # add the contribution m_j * W_ij to the density sum
                                    density_sum += mass_j * kernel_val

            # apply a minimum density floor and store the result
            densities_field[i] = tim.max(ti.cast(density_floor, FP), density_sum)
            
    # this kernel updates thermodynamic quantities (temperature, pressure) based on the ideal gas equation of state
    # it also precomputes the P/rho^2 term used in the SPH force calculation
    @ti.kernel
    def _update_eos_p_term_ti_kernel(
        internal_energies_field: ti.template(), masses_field: ti.template(),
        densities_field: ti.template(), # input fields
        temperatures_field: ti.template(), pressures_field: ti.template(), # output fields for T and P
        term_p_rho2_field: ti.template(), # output field for P/rho^2
        cv_sim: ti.template(), gas_constant_sim: ti.template(), # simulation specific heat capacity and gas constant
        min_temp: ti.template(), density_floor: ti.template(), # floors for stability
        use_rad_press: ti.template(), # flag to include radiation pressure
        FP: ti.template() # taichi float type
    ):
        """ calculates t, p (gas + optional radiation), and p/rho^2 term using taichi. """
        one_third_a_rad: FP = ti.cast(CONST_a_rad / 3.0, FP) # radiation constant term a/3
        cv_safe = ti.max(cv_sim, ti.cast(1e-50, FP)) # ensure cv is not zero
        P_ZERO: FP = ti.cast(0.0, FP)
        T_ZERO: FP = ti.cast(0.0, FP)

        # loop over all particles
        for i in temperatures_field:
            # calculate temperature T = u / Cv, where u is specific internal energy (U/m)
            mass_safe = ti.max(masses_field[i], ti.cast(1e-50, FP)) # avoid division by zero mass
            u_specific = internal_energies_field[i] / mass_safe
            temp_calc = u_specific / cv_safe
            # apply minimum temperature floor
            T_i = ti.max(min_temp, temp_calc)
            temperatures_field[i] = T_i

            # calculate pressure P = P_gas + P_rad
            rho_safe = ti.max(densities_field[i], density_floor) # apply density floor
            temp_safe = ti.max(T_i, T_ZERO) # ensure temperature is non-negative

            # ideal gas pressure P_gas = rho * R_sim * T
            Pgas = rho_safe * gas_constant_sim * temp_safe
            Prad = P_ZERO
            # optionally add radiation pressure P_rad = a/3 * T^4
            if use_rad_press:
                Prad = one_third_a_rad * (temp_safe**ti.cast(4.0, FP))
            P_i = Pgas + Prad
            pressures_field[i] = P_i

            # precompute the term P/rho^2 needed for the sph force calculation
            rho2_safe = rho_safe * rho_safe
            term_val = P_ZERO
            # avoid division by zero if pressure or density^2 is effectively zero
            if ti.abs(P_i) > ti.cast(1e-30, FP) and rho2_safe > ti.cast(1e-50, FP):
                 term_val = P_i / rho2_safe
            term_p_rho2_field[i] = term_val

    # this kernel calculates the sph forces (pressure + artificial viscosity)
    # and also the rate of change of internal energy due to pressure work and viscous heating
    @ti.kernel
    def calculate_sph_force_energy_visc_ti_kernel(
        positions_field: ti.template(), v_half_field: ti.template(), # positions and half-step velocities
        masses_field: ti.template(), densities_field: ti.template(), # mass and density
        term_P_rho2_field: ti.template(), # precomputed P/rho^2 term
        sph_forces_field: ti.template(), # output: accumulates sph forces onto existing forces
        work_terms_field: ti.template(), # output: stores du/dt from sph pressure work
        visc_heating_terms_field: ti.template(), # output: stores du/dt from sph artificial viscosity heating
        grid_dim_vec: ti.types.vector(3, ti.i32), # neighbor grid info
        grid_num_particles_field: ti.template(),
        grid_particle_indices_field: ti.template(),
        domain_min_vec_arg: ti.template(),
        N: int, h: float, h_sq: float, # simulation parameters
        grad_spiky_coeff: float, # precomputed spiky kernel gradient coefficient
        alpha_visc: float, beta_visc: float, # artificial viscosity parameters
        gamma_eos: float, # adiabatic index (needed for sound speed calculation in av)
        eta_sq_av: float, # artificial viscosity parameter eta^2 (prevents singularity)
        grid_inv_cell_size_val: float, index_dtype: ti.template(), # grid helpers
        FP: ti.template(), 
        debug_particle_idx: ti.i32 # index for optional debugging output (not used currently)
    ):
        # small numbers for stability
        EPSILON: FP = ti.cast(1e-9, FP) * h
        EPSILON_SQ: FP = EPSILON * EPSILON
        h_fp: FP = ti.cast(h, FP)
        h_sq_fp: FP = ti.cast(h_sq, FP)

        # zero out the energy rate fields at the beginning of the kernel
        for i in work_terms_field:
            work_terms_field[i] = ti.cast(0.0, FP)
        for i in visc_heating_terms_field:
            visc_heating_terms_field[i] = ti.cast(0.0, FP)

        ti.loop_config(block_dim=128) # suggest a block dimension for gpu execution
        # loop over all particles 'i'
        for i in range(N):
            # initialize the force accumulator for particle i
            sph_force_accum_i = ti.types.vector(3, FP)([0.0, 0.0, 0.0])
            mass_i: FP = masses_field[i]
            density_i: FP = densities_field[i]

            # only proceed if particle has mass and density
            if mass_i > ti.cast(1e-30, FP) and density_i > ti.cast(1e-30, FP):
                pos_i = positions_field[i]
                v_half_i = v_half_field[i] # use half-step velocity for viscosity calculation
                # read the precomputed P_i/rho_i^2 term
                term1: FP = term_P_rho2_field[i]
                # calculate sound speed squared c^2 = gamma * P / rho = gamma * (P/rho^2) * rho
                c_sq_i = gamma_eos * term1 * density_i 
                c_i = tim.sqrt(ti.max(c_sq_i, ti.cast(0.0,FP))) # sound speed c_i

                # find neighbors using the grid
                relative_pos_i = pos_i - domain_min_vec_arg
                cell_i = ti.floor(relative_pos_i * ti.cast(grid_inv_cell_size_val, FP)).cast(index_dtype)
                cell_i = tim.max(ti.cast(0, index_dtype), tim.min(cell_i, grid_dim_vec - ti.cast(1, index_dtype)))

                # loop over neighboring cells
                for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                    neighbor_cell = (cell_i + offset + grid_dim_vec) % grid_dim_vec
                    num_in_cell = grid_num_particles_field[neighbor_cell]
                    # loop over particles 'j' in the neighboring cell
                    for k in range(num_in_cell):
                        j = grid_particle_indices_field[neighbor_cell, k]
                        if i == j: continue # skip self
                        mass_j: FP = masses_field[j]
                        density_j: FP = densities_field[j]
                        # check if neighbor is valid
                        if mass_j > ti.cast(1e-30, FP) and density_j > ti.cast(1e-30, FP):
                            pos_j = positions_field[j]
                            rij = pos_i - pos_j
                            dist_sq: FP = rij.dot(rij)

                            # check if within kernel radius h and not coincident
                            if dist_sq < h_sq_fp and dist_sq > EPSILON_SQ:
                                r: FP = tim.sqrt(dist_sq)
                                h_minus_r: FP = h_fp - r
                                h_minus_r_sq: FP = h_minus_r * h_minus_r
                                # calculate gradient of the spiky kernel grad(W_spiky)
                                inv_r = ti.cast(1.0, FP) / ti.max(r, ti.cast(1e-9, FP)*h_fp) # safe 1/r
                                spiky_grad_mag_factor: FP = ti.cast(grad_spiky_coeff, FP) * h_minus_r_sq * inv_r
                                gW_ij = spiky_grad_mag_factor * rij # gradient vector pointing from j to i

                                # read precomputed P_j/rho_j^2 term
                                term2: FP = term_P_rho2_field[j]
                                # this is the symmetric SPH pressure term (P_i/rho_i^2 + P_j/rho_j^2)
                                pressure_term_sum: FP = term1 + term2

                                # calculate the pressure force contribution on particle 'i' due to 'j'
                                # F_press_i = - sum_j m_j * (P_i/rho_i^2 + P_j/rho_j^2) * grad(W_ij)
                                force_pressure_contrib_on_i = -mass_j * pressure_term_sum * gW_ij 
                                # accumulate pressure force
                                sph_force_accum_i += force_pressure_contrib_on_i 

                                # calculate the rate of internal energy change due to pressure work (PdV work)
                                v_half_j = v_half_field[j] # use half-step velocity
                                dv_ij = v_half_i - v_half_j
                                # rate of work done by pressure force between the pair: dU_pair/dt = - F_press_ij . dv_ij
                                # note: force_pressure_contrib_on_i is the force ON i, so F_ij = -force_pressure_contrib_on_i
                                dU_pair_dt_pressure: FP = (-force_pressure_contrib_on_i).dot(dv_ij)
                                # distribute the work symmetrically as a specific energy change rate (du/dt)
                                work_increment_i: FP = ti.cast(0.0, FP)
                                if mass_i > ti.cast(1e-30, FP): work_increment_i = ti.cast(0.5, FP) * dU_pair_dt_pressure / mass_i
                                work_increment_j: FP = ti.cast(0.0, FP)
                                if mass_j > ti.cast(1e-30, FP): work_increment_j = ti.cast(0.5, FP) * dU_pair_dt_pressure / mass_j
                                
                                # add contributions atomically to avoid race conditions in parallel execution
                                # only add for one direction of the pair (e.g., i < j) to count each pair's contribution once
                                if i < j : 
                                    ti.atomic_add(work_terms_field[i], work_increment_i)
                                    ti.atomic_add(work_terms_field[j], work_increment_j)
                                elif j < i: 
                                    # this handles the case where the loop order isn't guaranteed; add i's part now
                                    ti.atomic_add(work_terms_field[i], work_increment_i)


                                # --- Artificial Viscosity (standard Monaghan type) ---
                                # this term adds dissipation, typically only for converging flows (shocks)
                                Pi_ij: FP = ti.cast(0.0, FP) # initialize viscosity term
                                force_av_contrib_on_i = ti.types.vector(3,FP)([0.0,0.0,0.0]) # initialize av force

                                # check if viscosity is enabled
                                if alpha_visc > ti.cast(1e-9, FP) or beta_visc > ti.cast(1e-9,FP):
                                    v_dot_r: FP = dv_ij.dot(rij) # project relative velocity onto separation vector
                                    # only apply viscosity if particles are approaching (v_dot_r < 0)
                                    if v_dot_r < ti.cast(0.0, FP): 
                                        # calculate sound speed for particle j
                                        c_sq_j = gamma_eos * term2 * density_j
                                        c_j = tim.sqrt(ti.max(c_sq_j, ti.cast(0.0,FP)))
                                        
                                        # calculate average sound speed and density for the pair
                                        c_ij: FP = ti.cast(0.5, FP) * (c_i + c_j)
                                        rho_ij: FP = ti.cast(0.5, FP) * (density_i + density_j)
                                        
                                        # calculate mu_ij term, related to v_dot_r / r
                                        mu_ij_numerator: FP = h_fp * v_dot_r
                                        # denominator includes eta^2 term to prevent singularity when r -> 0
                                        mu_ij_denominator: FP = dist_sq + ti.cast(eta_sq_av, FP) * h_sq_fp
                                        mu_ij: FP = mu_ij_numerator / ti.max(mu_ij_denominator, ti.cast(1e-9,FP)*h_sq_fp)

                                        # calculate the viscosity term Pi_ij (Monaghan 1992 form)
                                        Pi_ij = (-alpha_visc * c_ij * mu_ij + beta_visc * mu_ij * mu_ij) / rho_ij
                                        Pi_ij = ti.max(Pi_ij, ti.cast(0.0, FP)) # ensure viscosity is dissipative (non-negative)

                                        # calculate the artificial viscosity force contribution on i
                                        # F_av_i = - sum_j m_j * Pi_ij * grad(W_ij)
                                        force_av_contrib_on_i = -mass_j * Pi_ij * gW_ij
                                        # accumulate viscosity force
                                        sph_force_accum_i += force_av_contrib_on_i
                                
                                # --- Viscous Heating from AV ---
                                # calculate the rate of internal energy change due to artificial viscosity
                                if Pi_ij > ti.cast(1e-30, FP): # only if viscosity was applied
                                    # rate of energy dissipation for the pair: dU_pair_av/dt = - F_av_ij . dv_ij
                                    dU_pair_dt_av: FP = (-force_av_contrib_on_i).dot(dv_ij) 
                                    
                                    # distribute heating symmetrically as specific energy rate (du/dt)
                                    visc_heating_increment_i: FP = ti.cast(0.0, FP)
                                    if mass_i > ti.cast(1e-30, FP): visc_heating_increment_i = ti.cast(0.5, FP) * dU_pair_dt_av / mass_i
                                    visc_heating_increment_j: FP = ti.cast(0.0, FP)
                                    if mass_j > ti.cast(1e-30, FP): visc_heating_increment_j = ti.cast(0.5, FP) * dU_pair_dt_av / mass_j

                                    # add contributions atomically and symmetrically
                                    if i < j:
                                        ti.atomic_add(visc_heating_terms_field[i], visc_heating_increment_i)
                                        ti.atomic_add(visc_heating_terms_field[j], visc_heating_increment_j)
                                    elif j < i:
                                        ti.atomic_add(visc_heating_terms_field[i], visc_heating_increment_i)


            # add the accumulated sph force (pressure + viscosity) to the particle's total force
            # note: this ADDS to whatever force might already be there (e.g., gravity)
            sph_forces_field[i] += sph_force_accum_i

    # kernel to reset the particle counts in the neighbor grid
    @ti.kernel
    def reset_grid_counts_ti_kernel(grid_num_particles_field: ti.template()):
        for I in ti.grouped(grid_num_particles_field):
            grid_num_particles_field[I] = 0

    # kernel to build the neighbor grid: assign each particle to a grid cell
    @ti.kernel
    def build_particle_grid_ti_kernel(
        positions_field: ti.template(),
        grid_num_particles_field: ti.template(), # stores counts per cell (atomic add)
        grid_particle_indices_field: ti.template(), # stores particle indices per cell
        N: int,
        grid_inv_cell_size_val: float, # 1 / cell_size
        grid_max_particles_per_cell: int, # maximum number of particles allowed per cell
        grid_dim_vec: ti.types.vector(3, ti.i32),
        domain_min_vec_arg: ti.template(), 
        index_dtype: ti.template()
    ):
        # loop over all particles
        for i in range(N):
            pos_i = positions_field[i]
            # calculate cell index based on position
            relative_pos = pos_i - domain_min_vec_arg
            cell_indices = ti.floor(relative_pos * grid_inv_cell_size_val).cast(index_dtype)
            # clamp index to grid bounds
            cell_indices = tim.max(ti.cast(0, index_dtype), tim.min(cell_indices, grid_dim_vec - ti.cast(1, index_dtype)))

            # atomically increment the particle counter for this cell and get the slot index
            slot_index = ti.atomic_add(grid_num_particles_field[cell_indices], 1)
            # if there's space in the cell, store the particle's index
            if slot_index < grid_max_particles_per_cell:
                grid_particle_indices_field[cell_indices, slot_index] = i
            # if the cell is full, undo the increment (particle is not added to grid)
            else:
                ti.atomic_sub(grid_num_particles_field[cell_indices], 1)
                # note: this means very dense regions might miss some neighbor interactions if grid_max_particles_per_cell is too low

# this class manages the SPH calculations using the Taichi kernels
class SPHTaichi(SPHModel):
    def __init__(self, config: dict = None):
        super().__init__(config)
        if not HAVE_TAICHI: raise ImportError("Taichi required for SPHTaichi.")
        # make sure taichi is initialized before we try to get its config
        try:
            current_cfg = ti.lang.impl.current_cfg()
            if current_cfg is None: raise RuntimeError("Taichi not initialized")
        except AttributeError: raise RuntimeError("Taichi init error")

        # store the float and integer types taichi is using
        self.ti_fp_dtype = current_cfg.default_fp
        self.ti_index_dtype = current_cfg.default_ip
        # get the corresponding numpy types
        self.np_fp_dtype = to_numpy_type(self.ti_fp_dtype)
        self.np_index_dtype = to_numpy_type(self.ti_index_dtype)

        print(f"SPHTaichi Initialized: Taichi FP={self.ti_fp_dtype}, IP={self.ti_index_dtype} | NumPy FP={self.np_fp_dtype.__name__}, IP={self.np_index_dtype.__name__}")

        # initialize parameters needed for SPH
        self.N = 0
        self.h_py = 1.0; self.h_sq_py = 1.0; self.density_floor_py = 1e-9
        # kernel constants (will be calculated in setup)
        self.W0_poly6_py = 0.0; self.C_poly6_py = 0.0; self.grad_spiky_coeff_py = 0.0
        # artificial viscosity parameters
        self.alpha_py = 0.0; self.beta_py = 0.0
        self.gamma_eos_py = 1.4 
        self.eta_sq_av_py = 0.01 

        # parameters for the neighbor search grid
        self.cell_size_py = 0.0; self.grid_inv_cell_size_py = 0.0
        self.grid_max_particles_per_cell = 64 # default, will be adjusted in setup
        self.grid_dim_vec_ti = None; self.grid_shape_tuple_py = None
        self.box_size_vec_ti = None; self.domain_min_vec_ti = None

        # parameters needed for the internal EOS calculation
        self.cv_sim_py = 1.0 
        self.gas_constant_sim_py = 1.0 
        self.min_temp_py = 10.0 
        self.use_rad_press_py = False 

        # internal taichi fields (managed by this class, not ParticleData)
        self.grid_num_particles = None; self.grid_particle_indices = None
        self.term_p_rho2_field = None; # note: visc_heating_terms is now in ParticleData

        # store the target numpy precision
        self.target_dtype_np = self.np_fp_dtype

    # setup method, called when the model is selected or simulation restarts
    def setup(self, pd: ParticleData):
        super().setup(pd) # call base class setup
        self.N = pd.get_n() # get current number of particles
        self.target_dtype_np = self.np_fp_dtype # store numpy float type
        print(f"SPHTaichi (Setup): Using NumPy target dtype: {self.target_dtype_np.__name__}")
        print(f"SPHTaichi (Setup): Using Taichi runtime FP: {self.ti_fp_dtype}, IP: {self.ti_index_dtype}")

        # extract SPH parameters from the configuration dictionary
        try:
            self.h_py = float(self.config['h']); self.h_py = max(self.h_py, 1e-6) # smoothing length
            self.h_sq_py = self.h_py * self.h_py
            self.density_floor_py = float(self.config.get('density_floor', 1e-9)) # minimum density allowed
            domain_size_py = float(self.config['L']) # simulation box size
            # artificial viscosity coefficients
            self.alpha_py = float(self.config.get('sph_visc_alpha', 0.0)) 
            self.beta_py = float(self.config.get('sph_visc_beta', 0.0))
            # adiabatic index (gamma) needed for sound speed calculation
            self.gamma_eos_py = float(self.config.get('sph_eos_gamma', 1.4)) 
        except KeyError as e: raise ValueError(f"Missing SPH config key: {e}") from e
        except (TypeError, ValueError) as e: raise ValueError(f"Invalid SPH config value: {e}") from e

        # extract thermodynamics parameters needed for the internal EOS calculation
        try:
            self.cv_sim_py = float(self.config['cv']) # simulation specific heat capacity
            self.gas_constant_sim_py = float(self.config['gas_constant_sim']) # simulation gas constant R_sim
            self.min_temp_py = float(self.config['min_temperature']) # minimum temperature floor
            self.use_rad_press_py = bool(self.config['use_rad_press']) # flag for radiation pressure
        except KeyError as e: raise ValueError(f"Missing Thermo config key needed by SPHTaichi EOS: {e}") from e
        except (TypeError, ValueError) as e: raise ValueError(f"Invalid Thermo config value for SPHTaichi EOS: {e}") from e

        # validate parameters
        if domain_size_py <= 1e-6: raise ValueError("Domain size 'L' must be positive.")
        if self.cv_sim_py <= 0: raise ValueError("'cv' must be positive.")
        if self.gas_constant_sim_py <= 0: raise ValueError("'gas_constant_sim' must be positive.")

        # setup the neighbor search grid parameters
        self.cell_size_py = self.h_py # typically set cell size equal to smoothing length
        if self.cell_size_py <= 1e-9: raise ValueError("Cell size must be positive.")
        self.grid_inv_cell_size_py = 1.0 / self.cell_size_py
        # calculate grid dimensions based on domain size and cell size
        grid_dim_np = np.ceil(np.array([domain_size_py] * 3) * self.grid_inv_cell_size_py).astype(self.np_index_dtype)
        MAX_GRID_DIM = 256 # limit grid dimensions for performance/memory
        grid_dim_np = np.minimum(grid_dim_np, MAX_GRID_DIM)
        grid_dim_np = np.maximum(grid_dim_np, 1) # ensure at least 1 cell per dimension
        new_grid_shape_tuple = tuple(grid_dim_np.tolist())
        # estimate average particles per cell and adjust max particles allowed per cell dynamically
        avg_particles_per_cell = self.N / float(max(1, np.prod(grid_dim_np)))
        new_max_particles_per_cell = int(max(128, min(8192, avg_particles_per_cell * 25 + 250)))
        print(f"SPHTaichi Grid Setup: Target CellSize={self.cell_size_py:.3f}, Dims={new_grid_shape_tuple}, MaxPerCell={new_max_particles_per_cell}")

        # create taichi vectors for domain boundaries and grid dimensions
        self.box_size_vec_ti = ti.Vector([domain_size_py]*3, dt=self.ti_fp_dtype)
        self.domain_min_vec_ti = self.box_size_vec_ti * -0.5 # assume centered box [-L/2, L/2)
        self.grid_dim_vec_ti = ti.Vector(grid_dim_np.tolist(), dt=self.ti_index_dtype) 

        # check if internal taichi fields need reallocation (e.g., N changed, grid size changed)
        realloc_grid = (self.grid_num_particles is None or
                        self.grid_num_particles.shape != new_grid_shape_tuple or
                        self.grid_particle_indices.shape != new_grid_shape_tuple + (new_max_particles_per_cell,))
        realloc_particle_fields = (self.term_p_rho2_field is None or
                                   (self.N > 0 and self.term_p_rho2_field.shape[0] != self.N))

        # if reallocation is needed, clean up old fields and allocate new ones
        if realloc_grid or realloc_particle_fields:
             print(f"INFO: Allocating/Reallocating SPHTaichi internal fields... N={self.N}")
             self.cleanup_internal_fields() # release old fields first
             self.grid_shape_tuple_py = new_grid_shape_tuple
             self.grid_max_particles_per_cell = new_max_particles_per_cell
             # only allocate if grid dimensions and N are valid
             if all(d > 0 for d in self.grid_shape_tuple_py) and self.N >= 0:
                 try:
                     # allocate taichi fields for the grid and the P/rho^2 term
                     self.grid_num_particles = ti.field(dtype=ti.i32, shape=self.grid_shape_tuple_py)
                     self.grid_particle_indices = ti.field(dtype=self.ti_index_dtype, shape=self.grid_shape_tuple_py + (self.grid_max_particles_per_cell,))
                     if self.N > 0: # only allocate particle-sized fields if N > 0
                         self.term_p_rho2_field = ti.field(dtype=self.ti_fp_dtype, shape=self.N)
                     print("INFO: SPHTaichi fields allocated.")
                 except Exception as e:
                     # handle allocation errors
                     print(f"ERROR: Failed to allocate SPHTaichi fields: {e}"); traceback.print_exc()
                     self.grid_num_particles = None; self.grid_particle_indices = None; self.term_p_rho2_field = None
                     raise RuntimeError("Failed to allocate SPHTaichi fields") from e
             else:
                 print(f"Warning: Invalid grid dims {self.grid_shape_tuple_py} or N={self.N}.")

        # precompute SPH kernel normalization constants (using float64 for accuracy)
        np_fp_64 = np.float64
        h_py_64 = np_fp_64(self.h_py)
        if h_py_64 > 1e-9:
            h_inv = np_fp_64(1.0 / h_py_64); h6 = h_py_64**6; h9 = h_py_64**9
            # Poly6 kernel normalization constant C_poly6 = 315 / (64 * pi * h^9)
            self.C_poly6_f64 = np_fp_64(315.0 / (64.0 * pi)) * h_inv**9
            # Poly6 kernel value at zero distance W0 = C_poly6 * h^6
            self.W0_poly6_f64 = self.C_poly6_f64 * h6
            # Spiky kernel gradient coefficient = -45 / (pi * h^6)
            self.grad_spiky_coeff_f64 = np_fp_64(-45.0 / (pi * h6))
        else:
            # handle case where h is too small
            self.C_poly6_f64 = np_fp_64(0.0); self.W0_poly6_f64 = np_fp_64(0.0); self.grad_spiky_coeff_f64 = np_fp_64(0.0)

        # convert constants to native python floats for passing to taichi kernels
        self.W0_poly6_py_ti_fp = float(self.W0_poly6_f64)
        self.C_poly6_py_ti_fp = float(self.C_poly6_f64)
        self.grad_spiky_coeff_py_ti_fp = float(self.grad_spiky_coeff_f64)

        print(f"SPHTaichi Setup Complete: N={self.N}, h={self.h_py:.3f}, alpha={self.alpha_py:.2f}, beta={self.beta_py:.2f}, gamma_eos={self.gamma_eos_py:.2f}")
        print(f"  Using internal EOS with SimCv={self.cv_sim_py:.3e}, SimGasConst={self.gas_constant_sim_py:.3e}")

        # ensure all particle data fields needed by the kernels are present on the taichi device
        pd.ensure([
            "positions", "masses", "velocities", "v_half_temp", "densities",
            "pressures", "temperatures", "forces", "work_terms",
            "visc_heating_terms", "internal_energies" # internal_energies is needed for the internal EOS
            ], "gpu:ti")

    # internal helper method to build the neighbor search grid
    @timing_decorator
    def _build_grid_internal(self, pd: ParticleData):
        if self.N == 0: return True # nothing to do if no particles
        # check if grid fields are allocated
        if self.grid_num_particles is None or self.grid_particle_indices is None: return False
        pos_field = pd.get("positions", "gpu:ti") # get positions from taichi backend
        try:
            # call the kernels to reset counts and assign particles to cells
            reset_grid_counts_ti_kernel(self.grid_num_particles)
            build_particle_grid_ti_kernel(
                pos_field, self.grid_num_particles, self.grid_particle_indices, self.N,
                self.grid_inv_cell_size_py, self.grid_max_particles_per_cell,
                self.grid_dim_vec_ti, self.domain_min_vec_ti, self.ti_index_dtype
            )
            ti.sync() # make sure kernels finish
            return True
        except Exception as e:
            # handle errors during grid build
            print(f"ERROR building Taichi SPH grid: {e}"); traceback.print_exc(); return False

    # method to compute SPH density
    @timing_decorator
    def compute_density(self, pd: ParticleData, neighbor_list=None, debug_particle_idx: int = -1):
        if self.N == 0: pd.set("densities", np.zeros(0, dtype=self.target_dtype_np)); return # handle N=0 case
        # first, build the neighbor grid
        if not self._build_grid_internal(pd):
             print("Warning: Taichi SPH grid build failed. Skipping density calculation."); return
        # ensure required fields are on the taichi device
        pd.ensure("positions masses densities", "gpu:ti")
        pos_field = pd.get("positions", "gpu:ti"); mass_field = pd.get("masses", "gpu:ti"); dens_field = pd.get("densities", "gpu:ti")
        if self.grid_num_particles is None: print("ERROR: Grid not allocated."); return # sanity check
        try:
            # call the density calculation kernel
            calculate_density_sph_ti_kernel(
                 pos_field, mass_field, dens_field, self.grid_dim_vec_ti, self.grid_num_particles, self.grid_particle_indices,
                 self.domain_min_vec_ti, self.N, self.h_py, self.h_sq_py, self.density_floor_py,
                 self.W0_poly6_py_ti_fp, self.C_poly6_py_ti_fp, 
                 self.grid_inv_cell_size_py, self.ti_index_dtype, self.ti_fp_dtype
             )
            ti.sync() # wait for kernel
            # mark density data as up-to-date on the taichi device
            pd._location["densities"] = "gpu:ti"; pd._gpu_dirty["densities"] = False
        except Exception as e: print(f"ERROR during Taichi density kernel: {e}"); traceback.print_exc()

    # method to compute SPH pressure forces, artificial viscosity forces, work terms, and heating terms
    @timing_decorator
    def compute_pressure_force(self, pd: ParticleData, neighbor_list=None, debug_particle_idx: int = -1):
        """ computes sph forces (pressure + av) and energy terms (work, av heating).
            includes internal calculation of t, p, and p/rho^2 term using taichi kernels. """
        if self.N == 0: 
            # handle N=0 case, ensuring output arrays are empty
            if pd.get_n() == 0: 
                pd.set("forces", np.zeros((0,3), dtype=pd.get_dtype("forces")))
                pd.set("work_terms", np.zeros(0, dtype=pd.get_dtype("work_terms")))
                pd.set("visc_heating_terms", np.zeros(0, dtype=pd.get_dtype("visc_heating_terms")))
            return

        # check if internally managed taichi fields are allocated
        if self.grid_num_particles is None or self.grid_particle_indices is None \
           or self.term_p_rho2_field is None : 
            print("Warning: Skipping SPH force/work/visc calculation. Internal fields not allocated.")
            # zero the output fields on cpu if skipping calculation
            work_np = np.zeros(self.N, dtype=pd.get_dtype("work_terms"))
            visc_np = np.zeros(self.N, dtype=pd.get_dtype("visc_heating_terms"))
            pd.set("work_terms", work_np, source_device="cpu")
            pd.set("visc_heating_terms", visc_np, source_device="cpu")
            return

        # ensure all required particle data is on the taichi device
        required_fields = [
            "positions", "v_half_temp", "masses", "densities",
            "internal_energies", # needed for internal eos
            "temperatures", "pressures", # output fields for internal eos
            "forces", "work_terms", "visc_heating_terms" # accumulated/output fields
        ]
        pd.ensure(required_fields, "gpu:ti")

        # get handles to the taichi fields from ParticleData
        pos_field = pd.get("positions", "gpu:ti")
        v_half_field = pd.get("v_half_temp", "gpu:ti") # half-step velocity
        mass_field = pd.get("masses", "gpu:ti")
        dens_field = pd.get("densities", "gpu:ti")
        internal_energies_field = pd.get("internal_energies", "gpu:ti")
        temperatures_field = pd.get("temperatures", "gpu:ti") # output of eos kernel
        pressures_field = pd.get("pressures", "gpu:ti") # output of eos kernel
        force_field = pd.get("forces", "gpu:ti") # sph forces will be ADDED to this
        work_field = pd.get("work_terms", "gpu:ti") # stores du/dt from pressure work
        visc_heat_field = pd.get("visc_heating_terms", "gpu:ti") # stores du/dt from av heating

        # step 1: calculate T, P, and P/rho^2 using the internal EOS kernel
        try:
            _update_eos_p_term_ti_kernel(
                internal_energies_field, mass_field, dens_field, # inputs
                temperatures_field, pressures_field, # outputs T, P
                self.term_p_rho2_field, # output P/rho^2 term
                self.cv_sim_py, # simulation parameters
                self.gas_constant_sim_py,
                self.min_temp_py,
                self.density_floor_py,
                bool(self.use_rad_press_py), # flag
                self.ti_fp_dtype # taichi float type
            )
            ti.sync() # wait for eos kernel
            # mark T and P as updated on the taichi device
            pd._location["temperatures"] = "gpu:ti"; pd._gpu_dirty["temperatures"] = False
            pd._location["pressures"] = "gpu:ti"; pd._gpu_dirty["pressures"] = False

        except Exception as e:
            # handle errors during eos calculation
            print(f"ERROR computing internal Taichi EOS/Pressure term: {e}"); traceback.print_exc()
            # zero energy terms if eos fails, as forces won't be correct
            work_np = np.zeros(self.N, dtype=pd.get_dtype("work_terms"))
            visc_np = np.zeros(self.N, dtype=pd.get_dtype("visc_heating_terms"))
            pd.set("work_terms", work_np, source_device="cpu")
            pd.set("visc_heating_terms", visc_np, source_device="cpu")
            return # stop here if eos failed

        # step 2: calculate sph forces (pressure + av) and energy terms (work + av heating)
        try:
            # call the main sph force/energy kernel
            calculate_sph_force_energy_visc_ti_kernel(
                positions_field=pos_field,
                v_half_field=v_half_field,
                masses_field=mass_field,
                densities_field=dens_field,
                term_P_rho2_field=self.term_p_rho2_field, # input precomputed term
                sph_forces_field=force_field, # accumulates force
                work_terms_field=work_field, # output work rate
                visc_heating_terms_field=visc_heat_field, # output viscosity heating rate
                grid_dim_vec=self.grid_dim_vec_ti, # grid info
                grid_num_particles_field=self.grid_num_particles,
                grid_particle_indices_field=self.grid_particle_indices,
                domain_min_vec_arg=self.domain_min_vec_ti, 
                N=self.N, # simulation parameters
                h=self.h_py, h_sq=self.h_sq_py,
                grad_spiky_coeff=self.grad_spiky_coeff_py_ti_fp, 
                alpha_visc=self.alpha_py, # viscosity parameters
                beta_visc=self.beta_py,   
                gamma_eos=self.gamma_eos_py, # eos parameter needed for sound speed
                eta_sq_av=self.eta_sq_av_py, # viscosity parameter
                grid_inv_cell_size_val=self.grid_inv_cell_size_py, # grid helpers
                index_dtype=self.ti_index_dtype,
                FP=self.ti_fp_dtype, 
                debug_particle_idx=debug_particle_idx
            )
            ti.sync() # wait for force kernel
            # mark forces and energy terms as updated on the taichi device
            pd._location["forces"] = "gpu:ti"; pd._gpu_dirty["forces"] = False
            pd._location["work_terms"] = "gpu:ti"; pd._gpu_dirty["work_terms"] = False
            pd._location["visc_heating_terms"] = "gpu:ti"; pd._gpu_dirty["visc_heating_terms"] = False
        except Exception as e:
            # handle errors during force calculation
            print(f"ERROR during Taichi SPH force/energy/visc kernel execution: {e}")
            traceback.print_exc()
            # zero energy terms if force calculation fails
            if self.N > 0:
                work_np = np.zeros(self.N, dtype=pd.get_dtype("work_terms"))
                visc_np = np.zeros(self.N, dtype=pd.get_dtype("visc_heating_terms"))
                pd.set("work_terms", work_np, source_device="cpu")
                pd.set("visc_heating_terms", visc_np, source_device="cpu")

    # helper to clean up internal taichi fields managed by this class
    def cleanup_internal_fields(self):
        fields_to_clean = ['grid_num_particles', 'grid_particle_indices', 'term_p_rho2_field']
        cleaned_count = 0
        for field_name in fields_to_clean:
            field = getattr(self, field_name, None)
            if field is not None:
                # just setting to none should be enough for taichi's garbage collection
                setattr(self, field_name, None) 
                cleaned_count += 1
        return cleaned_count

    # cleanup method called when the model is deselected or simulation ends
    def cleanup(self):
        super().cleanup() # call base class cleanup first
        print("Cleaning up SPHTaichi internal fields...")
        cleaned_count = self.cleanup_internal_fields()
        if cleaned_count > 0:
            print(f"SPHTaichi internal fields cleared ({cleaned_count} fields).")
        else:
            print("SPHTaichi internal fields were already clear or not allocated.")