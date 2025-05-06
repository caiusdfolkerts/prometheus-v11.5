# prometheus/sim/physics/gravity/gravity_pp_gpu.py
"""
Direct Particle-Particle (N^2) gravity calculation using Taichi kernels.

This model uses Taichi to parallelize the N^2 force summation, suitable for
execution on GPUs or multi-core CPUs supported by Taichi. It is simpler than
Barnes-Hut but still O(N^2) complexity.
"""

import numpy as np
import time
import traceback
from typing import Dict, Optional

# Absolute imports from project structure
from sim.physics.base.gravity import GravityModel
from sim.particle_data import ParticleData
from sim.utils import timing_decorator

# --- Conditional Taichi Import ---
try:
    import taichi as ti
    import taichi.math as tim # Taichi math functions
    HAVE_TAICHI = True
except ImportError:
    ti = None
    tim = None
    HAVE_TAICHI = False

# ==============================
# --- Taichi Kernels ---
# ==============================

if HAVE_TAICHI:
    @ti.kernel
    def compute_forces_n2_ti_kernel(
        pos_field: ti.template(),   
        mass_field: ti.template(),   
        forces_out: ti.template(),  
        N: ti.i32,                 
        G: ti.template(),            
        softening_sq: ti.template(),
        FP: ti.template()           
    ):
        """(Taichi Kernel) Computes forces via N^2 summation in parallel."""
        # define vector type and small epsilon based on kernel's float type (FP)
        vec3 = ti.types.vector(3, FP)
        EPSILON = FP(1e-18) # small number to prevent zero div

        #parallel loop over target particles 'i'
        for i in forces_out: #taichi implicitly parallelizes this outer loop
            force_i = vec3(0.0) # init force accumulator for particle i
            pos_i = pos_field[i]
            mass_i = mass_field[i]

            # inner loop over source particles 'j'
            for j in range(N):
                if i == j: continue # skip self-interaction

                mass_j = mass_field[j]
                if mass_j <= 0: continue # skip massless source particles

                # Vector from j to i
                dr = pos_i - pos_field[j]
                dist_sq = dr.dot(dr)
                dist_sq_soft = dist_sq + softening_sq

                if dist_sq_soft > EPSILON:
                    inv_dist_soft = FP(1.0) / tim.sqrt(dist_sq_soft)
                    inv_dist3_soft = inv_dist_soft * inv_dist_soft * inv_dist_soft
                    force_i -= G * mass_j * inv_dist3_soft * dr

            # Store final force F_i = m_i * sum(a_ij)
            forces_out[i] = mass_i * force_i

    @ti.kernel
    def compute_potential_n2_ti_kernel(
        pos_field: ti.template(),    
        mass_field: ti.template(),  
        N: ti.i32,                  
        G: ti.template(),           
        softening_sq: ti.template(), 
        FP: ti.template()          
    ) -> FP: 
        """(Taichi Kernel) Computes total potential energy via N^2 sum."""
        total_potential = FP(0.0)
        EPSILON = FP(1e-18)

        for i in range(N):
            mass_i = mass_field[i]
            if mass_i <= 0: continue
            pos_i = pos_field[i]
            for j in range(i + 1, N): 
                mass_j = mass_field[j]
                if mass_j <= 0: continue

                dr = pos_i - pos_field[j]
                dist_sq = dr.dot(dr)
                dist_sq_soft = dist_sq + softening_sq

                if dist_sq_soft > EPSILON:
                    inv_dist_soft = FP(1.0) / tim.sqrt(dist_sq_soft)
                    ti.atomic_add(total_potential, -G * mass_i * mass_j * inv_dist_soft)

        return total_potential

# ==============================
# --- Python Class Definition ---
# ==============================

class GravityPPGpu(GravityModel):
    """Direct N^2 gravity calculation using Taichi kernels (CPU/GPU)."""

    def __init__(self, config: Optional[Dict] = None):
        """Initializes model, checking for Taichi and determining FP type."""
        super().__init__(config)
        if not HAVE_TAICHI:
            raise ImportError("Taichi is required for GravityPPGpu but not found.")
        try: self.ti_fp_dtype = ti.lang.impl.current_cfg().default_fp
        except Exception as e: raise RuntimeError("Taichi not initialized before GravityPPGpu init?") from e
        print(f"GravityPPGpu using Taichi dtype: {self.ti_fp_dtype}")
        self.ti_forces_out: Optional[ti.Field] = None
        self.N: int = 0

    def setup(self, pd: ParticleData):
        """Validates config, ensures data on Taichi device, allocates output field."""
        super().setup(pd)
        self.N = pd.get_n()
        try:
            self.G = float(self.config['G'])
            self.softening = float(self.config['softening'])
            if self.softening <= 0: raise ValueError("Softening must be positive.")
        except KeyError as e: raise ValueError(f"Missing required config key for GravityPPGpu: {e}")
        except (ValueError, TypeError) as e: raise ValueError(f"Invalid config value for GravityPPGpu: {e}")

        print(f"GravityPPGpu Setup: N={self.N}, G={self.G:.3e}, Softening={self.softening:.3f}, TaichiFP={self.ti_fp_dtype}")

        pd.ensure("positions masses", "gpu:ti")

        # Allocate the Taichi output field if needed (size changes or first time)
        if self.ti_forces_out is None or self.ti_forces_out.shape[0] != self.N:
             if self.N > 0:
                 self.ti_forces_out = ti.Vector.field(3, dtype=self.ti_fp_dtype, shape=self.N)
                 print(f"  Allocated Taichi force output field with shape {(self.N, 3)}")
             else:
                 self.ti_forces_out = None # ensure is None if N=0

        # Reset field before usijng it in compute_forces
        if self.ti_forces_out is not None: self.ti_forces_out.fill(0.0)

    def compute_forces(self, pd: ParticleData):
        """Computes forces using the Taichi N^2 kernel."""
        N = pd.get_n()
        if N <= 0: return # skip if no particles
        if self.ti_forces_out is None: # check if output field exists (Should do)
            print("Error: GravityPPGpu output field not allocated. Skipping force calculation."); return
        if N == 1: # Zero forces for single particle
            self.ti_forces_out.fill(0.0); ti.sync() # Zero GPU field
            try: # Also zero CPU fields
                f_main = pd.get("forces", "cpu", writeable=True); f_main.fill(0.0); pd.release_writeable("forces")
                f_grav = pd.get("forces_grav", "cpu", writeable=True); f_grav.fill(0.0); pd.release_writeable("forces_grav")
            except Exception as e: print(f"Warn: Failed to zero CPU forces for N=1: {e}")
            return

        # get input Taichi fields from ParticleData
        pos_field = pd.get("positions", "gpu:ti")
        mass_field = pd.get("masses", "gpu:ti")

        # prepare kernel arguments with correct Taichi types
        softening_sq_ti = self.ti_fp_dtype(self.softening * self.softening)
        G_ti = self.ti_fp_dtype(self.G)
        N_ti32 = np.int32(N) # Pass N as i32 (common Taichi practice)

        try:
             # --- Call Taichi Kernel ---
             compute_forces_n2_ti_kernel(pos_field, mass_field, self.ti_forces_out,
                                        N_ti32, G_ti, softening_sq_ti, self.ti_fp_dtype)
             ti.sync() # Ensure kernel finishes before reading results

             # --- transfer result back to CPU NumPy and store in ParticleData ---
             target_dtype = pd.get_dtype("forces") # get dtype expected by ParticleData
             forces_gpu_np = self.ti_forces_out.to_numpy().astype(target_dtype, copy=False)

             pd.set("forces_grav", forces_gpu_np) # store gravity-specific forces
             forces_main = pd.get("forces", "cpu", writeable=True) # get writeable main forces
             forces_main += forces_gpu_np # add Taichi result
             pd.release_writeable("forces")

        except Exception as e:
             print(f"ERROR during Taichi PP GPU force calculation: {e}")
             traceback.print_exc()
             # Attempt to zero forces on error
             try:
                f_main = pd.get("forces", "cpu", writeable=True); f_main.fill(0.0); pd.release_writeable("forces")
                f_grav = pd.get("forces_grav", "cpu", writeable=True); f_grav.fill(0.0); pd.release_writeable("forces_grav")
                if self.ti_forces_out: self.ti_forces_out.fill(0.0); ti.sync()
             except Exception as e_zero: print(f"Warn: Failed zeroing forces after error: {e_zero}")

    def compute_potential_energy(self, pd: ParticleData) -> float:
        """Computes total potential energy using an N^2 Taichi kernel."""
        N = pd.get_n()
        if N < 2: return 0.0

        # ensure data is on the Taichi device
        pd.ensure("positions masses", "gpu:ti")
        pos_field = pd.get("positions", "gpu:ti")
        mass_field = pd.get("masses", "gpu:ti")

        # prepare kernel arguments
        softening_sq_ti = self.ti_fp_dtype(self.softening * self.softening)
        G_ti = self.ti_fp_dtype(self.G)
        N_ti32 = np.int32(N)

        try:
            total_potential = compute_potential_n2_ti_kernel(
                pos_field, mass_field, N_ti32, G_ti, softening_sq_ti, self.ti_fp_dtype
            )
            ti.sync() # ensure kernel completes
            return float(total_potential) # return standard Python float
        except Exception as e:
            print(f"ERROR during Taichi PP GPU potential energy calculation: {e}")
            traceback.print_exc()
            return 0.0

    def cleanup(self):
        """Releases the internal Taichi force field."""
        super().cleanup()
        self.ti_forces_out = None 
        print("GravityPPGpu cleaned up.")