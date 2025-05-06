# prometheus/sim/physics/gravity/gravity_pp_cpu.py
import numpy as np
import traceback
from sim.physics.base.gravity import GravityModel # Absolute import
from sim.particle_data import ParticleData        # Absolute import
from sim.utils import timing_decorator            # Absolute import


class GravityPPCpu(GravityModel):
    """Direct Particle-Particle N^2 gravity calculation using pure NumPy."""
    def setup(self, pd: ParticleData):
        super().setup(pd)
        self.N = pd.get_n()
        self.G = self.config.get('G', 1.0)
        self.softening = self.config.get('softening', 0.1)
        self.G = float(self.G)
        self.softening = float(self.softening)
        print(f"GravityPPCpu Setup: N={self.N}, G={self.G}, Softening={self.softening}")

    @timing_decorator
    def compute_forces(self, pd: ParticleData):
        pd.ensure("positions masses", "cpu")
        pos_np = pd.get("positions", "cpu")
        mass_np = pd.get("masses", "cpu")

        pos_f64 = pos_np.astype(np.float64, copy=False)
        mass_f64 = mass_np.astype(np.float64, copy=False)
        
        calculated_grav_forces_f64 = np.zeros_like(pos_f64) # Accumulator for calculated gravity forces

        softening_sq_f64 = self.softening * self.softening
        G_f64 = self.G

        if G_f64 != 0.0: #oOnly compute if G is non-zero
            for i in range(self.N): # loop over particles
                force_i = np.zeros(3, dtype=np.float64)
                p_pos_i = pos_f64[i]
                p_mass_i = mass_f64[i]

                for j in range(self.N): # loop over source particles
                    if i == j: continue # skip single self-interaction
                    
                    p_mass_j = mass_f64[j]
                    if p_mass_j <= 0: continue # skip massless source

                    dr = p_pos_i - pos_f64[j] # vector from j to i
                    dist_sq = np.dot(dr, dr)
                    dist_sq_soft = dist_sq + softening_sq_f64
                    
                    if dist_sq_soft > 1e-30: # avoid zero div
                        inv_dist_soft = 1.0 / np.sqrt(dist_sq_soft)
                        inv_dist3_soft = inv_dist_soft * inv_dist_soft * inv_dist_soft
                        force_i -= G_f64 * p_mass_j * inv_dist3_soft * dr 
                
                calculated_grav_forces_f64[i] = force_i * p_mass_i # F_i = m_i * sum(a_ij)
        
        target_grav_dtype = pd.get_numpy_dtype("forces_grav")
        forces_grav_write = pd.get("forces_grav", "cpu", writeable=True)
        forces_grav_write[:] = calculated_grav_forces_f64.astype(target_grav_dtype, copy=False)
        pd.release_writeable("forces_grav")

        forces_main_write = pd.get("forces", "cpu", writeable=True)
        target_main_dtype = pd.get_numpy_dtype("forces")
        # Ensure main forces are initialized if this is the first force calc
        forces_main_write += calculated_grav_forces_f64.astype(target_main_dtype, copy=False)
        pd.release_writeable("forces")

    @timing_decorator
    def compute_potential_energy(self, pd: ParticleData) -> float:
        N = pd.get_n()
        if N < 2: return 0.0

        pd.ensure("positions masses", "cpu")
        pos_np = pd.get("positions", "cpu")
        mass_np = pd.get("masses", "cpu")

        pos_f64 = pos_np.astype(np.float64, copy=False)
        mass_f64 = mass_np.astype(np.float64, copy=False)
        
        G_f64 = float(self.G)
        softening_sq_f64 = float(self.softening * self.softening)
        total_potential_energy = 0.0

        if G_f64 == 0.0: return 0.0

        for i in range(N):
            mass_i = mass_f64[i]
            if mass_i <= 0: continue
            for j in range(i + 1, N): # Iterate j > i to count each pair once
                mass_j = mass_f64[j]
                if mass_j <= 0: continue

                dr = pos_f64[i] - pos_f64[j]
                dist_sq = np.dot(dr, dr)
                dist_soft = np.sqrt(dist_sq + softening_sq_f64)
                
                if dist_soft > 1e-15: # Avoid division by zero
                    total_potential_energy -= G_f64 * mass_i * mass_j / dist_soft
        
        return total_potential_energy