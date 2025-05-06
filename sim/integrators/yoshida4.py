# prometheus/sim/integrators/yoshida4.py
"""
Implementation of the 4th-order Yoshida symplectic integrator.
"""

import numpy as np
import time 
import traceback
from typing import Callable, Optional, Dict
from ..physics_kernels.thermo_eos import calculate_temperature_numba

from .base import Integrator
from sim.particle_data import ParticleData


# --- Yoshida 4th Order Coefficients ---
W0 = 1.0 / (2.0 - 2.0**(1.0/3.0))
W1 = (1.0 - 2.0**(1.0/3.0)) / (2.0 - 2.0**(1.0/3.0))
C1 = W0 / 2.0
C4 = W0 / 2.0
C2 = (W0 + W1) / 2.0
C3 = (W0 + W1) / 2.0
D1 = W0
D3 = W0
D2 = W1

class Yoshida4(Integrator):
    """
    Implements the 4th-order Yoshida symplectic integrator (Yoshida 1990).
    """

    def __init__(self):
        """Initialize integrator state."""
        self._current_config: Optional[Dict] = None

    def _kick(self, pd: ParticleData, inv_m: np.ndarray, factor_dt: float, stage: int, step_num_for_log: int):
        """Internal helper: Kick stage updates velocity and internal energy."""
        required = ["forces", "velocities", "internal_energies", "masses",
                    "work_terms", "cooling_rates", "fusion_rates", "visc_heating_terms",
                    "temperatures"]
        try:
            pd.ensure(required, target_device="cpu")
            f = pd.get("forces", "cpu"); m = pd.get("masses", "cpu") 
            vel = pd.get("velocities", "cpu", writeable=True) # Write v
            accel = f * inv_m[:, np.newaxis] # a = F/m 
            vel += accel * factor_dt
            pd.release_writeable("velocities")
           
            # --- Internal Energy Kick U += m * (du/dt) * (Dk*dt) ---
            u_tot = pd.get("internal_energies", "cpu", writeable=True) 
            
            work_rate = pd.get("work_terms", "cpu")
            cool_rate = pd.get("cooling_rates", "cpu")
            fus_rate = pd.get("fusion_rates", "cpu")
            visc_rate = pd.get("visc_heating_terms", "cpu") 
            
            total_du_dt = work_rate + visc_rate + fus_rate - cool_rate
            delta_U = m * total_du_dt * factor_dt
            u_tot += delta_U.astype(u_tot.dtype, copy=False)

            if self._current_config:
                cv_sim = self._current_config.get('cv')
                min_temp_floor_val = self._current_config.get('min_temperature', 0.0)
                if cv_sim is not None and cv_sim > 1e-30 and min_temp_floor_val > 0.0:
                    # Calculate min_U_total based on per-particle mass
                    min_U_total_floor_per_particle = m * cv_sim * min_temp_floor_val
                    np.maximum(u_tot, min_U_total_floor_per_particle, out=u_tot) # Apply floor in-place
            
            pd.release_writeable("internal_energies")

            # --- Recalculate Temperatures from updated Internal Energies ---
            if self._current_config:
                cv_sim = self._current_config.get('cv')
                min_temp_for_calc = self._current_config.get('min_temperature')

                if cv_sim is not None and min_temp_for_calc is not None:
                    new_temperatures_arr = calculate_temperature_numba(u_tot, cv_sim, min_temp_for_calc)
                    
                    # Get a writeable view/reference to ParticleData's temperatures array
                    temperatures_pd_view = pd.get("temperatures", "cpu", writeable=True)
                    
                    if temperatures_pd_view.shape == new_temperatures_arr.shape and \
                       temperatures_pd_view.dtype == new_temperatures_arr.dtype:
                        temperatures_pd_view[:] = new_temperatures_arr # Assign in-place
                    else:
                        print(f"    [WARN Kick Stage {stage}] Temperature array shape/dtype mismatch during update. "
                              f"Expected {temperatures_pd_view.shape} ({temperatures_pd_view.dtype}), "
                              f"Got {new_temperatures_arr.shape} ({new_temperatures_arr.dtype}). Skipping T update for safety.")
                    
                    pd.release_writeable("temperatures") # Release the lock on temperatures
                else:
                    print(f"    [WARN Kick Stage {stage}] Missing 'cv' or 'min_temperature' in config. Cannot update temperatures.")
            # --- End Temperature Recalculation ---
           
        except Exception as e:
            print(f"ERROR during Yoshida4 kick (Stage {stage}): {e}"); traceback.print_exc()
            # Ensure locks are released on error
            if pd._locked_writeable.get("velocities"): pd.release_writeable("velocities")
            if pd._locked_writeable.get("internal_energies"): pd.release_writeable("internal_energies")
            if pd._locked_writeable.get("temperatures"): pd.release_writeable("temperatures")
            raise 

    def _drift(self, pd: ParticleData, factor_dt: float, stage: int, step_num_for_log: int):
        """Internal helper: Drift stage updates positions."""
        try:
            pd.ensure(["velocities", "positions"], target_device="cpu")
            vel = pd.get("velocities", "cpu") # Read v
            pos = pd.get("positions", "cpu", writeable=True) # Write x
            pos += vel * factor_dt 

            # Apply Periodic Boundary Conditions if enabled in config
            if self._current_config and self._current_config.get('use_pbc', False):
                L_config = self._current_config.get('L_box_size_vec') 
                if L_config is not None:
                    if isinstance(L_config, (list, np.ndarray)) and len(L_config) == 3:
                        Lx, Ly, Lz = L_config[0], L_config[1], L_config[2]
                        if Lx > 0: pos[:, 0] = (pos[:, 0] + Lx * 0.5) % Lx - Lx * 0.5
                        if Ly > 0: pos[:, 1] = (pos[:, 1] + Ly * 0.5) % Ly - Ly * 0.5
                        if Lz > 0: pos[:, 2] = (pos[:, 2] + Lz * 0.5) % Lz - Lz * 0.5
                    elif isinstance(L_config, (float, int)) and L_config > 0: # Scalar L
                        pos[:] = (pos + L_config * 0.5) % L_config - L_config * 0.5

            pd.release_writeable("positions")

        except Exception as e:
            print(f"ERROR during Yoshida4 drift (Stage {stage}): {e}"); traceback.print_exc()
            if pd._locked_writeable.get("positions"): pd.release_writeable("positions")
            raise

    def step(self, pd: ParticleData, dt: float, forces_cb: Callable[[], None], config: Optional[Dict] = None, current_step: int = -1, debug_particle_idx: int = -1):
        """
        Performs one Yoshida 4th-order step, including energy integration
        """
        step_num_for_log = current_step + 1
        print(f"\n--- Yoshida4 Step {step_num_for_log} Start (dt={dt:.4e}) ---")
        self._current_config = config

        try:
            pd.ensure(["masses"], target_device="cpu")
            m = pd.get("masses", "cpu")
            inv_m = np.zeros_like(m, dtype=m.dtype)
            valid_m = m > 1e-30
            inv_m[valid_m] = 1.0 / m[valid_m]

            # Stage 1
            print("    [Stage 1] Drift(C1)...")
            self._drift(pd, C1 * dt, stage=1, step_num_for_log=step_num_for_log)
            print("    [Stage 1] forces_cb()...")
            forces_cb(debug_particle_idx=debug_particle_idx) 
            print("    [Stage 1] Kick(D1)...")
            self._kick(pd, inv_m, D1 * dt, stage=1, step_num_for_log=step_num_for_log)

            # Stage 2
            print("    [Stage 2] Drift(C2)...")
            self._drift(pd, C2 * dt, stage=2, step_num_for_log=step_num_for_log)
            print("    [Stage 2] forces_cb()...")
            forces_cb(debug_particle_idx=debug_particle_idx)
            print("    [Stage 2] Kick(D2)...")
            self._kick(pd, inv_m, D2 * dt, stage=2, step_num_for_log=step_num_for_log)

            # Stage 3
            print("    [Stage 3] Drift(C3)...")
            self._drift(pd, C3 * dt, stage=3, step_num_for_log=step_num_for_log)
            print("    [Stage 3] forces_cb()...")
            forces_cb(debug_particle_idx=debug_particle_idx)
            print("    [Stage 3] Kick(D3)...")
            self._kick(pd, inv_m, D3 * dt, stage=3, step_num_for_log=step_num_for_log)

            # Stage 4 (Final Drift)
            print("    [Stage 4] Drift(C4)...")
            self._drift(pd, C4 * dt, stage=4, step_num_for_log=step_num_for_log)
    
            print("    [End of Step] Final forces_cb() for next step's start...")
            forces_cb(debug_particle_idx=debug_particle_idx)

            print(f"--- Yoshida4 Step {step_num_for_log} End ---")

        except Exception as e_step:
            print(f"ERROR during Yoshida4 step {step_num_for_log} execution: {e_step}")
            traceback.print_exc()
            raise
        finally:
            self._current_config = None