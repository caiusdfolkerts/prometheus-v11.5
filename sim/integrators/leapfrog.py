# prometheus/sim/integrators/leapfrog.py
import numpy as np
import traceback
from typing import Callable, Dict, Optional
from .base import Integrator
from sim.particle_data import ParticleData

class Leapfrog(Integrator):
    """
    implements the leapfrog (kick-drift-kick) time integration scheme.
    this version includes updates for internal energy based on averaged rates.
    """
    _rate_fields = ["work_terms", "visc_heating_terms", "cooling_rates", "fusion_rates"] # names of energy rate fields

    def setup(self, pd: ParticleData, config: dict = None):
        """
        sets up the leapfrog integrator.
        ensures that '_prev' versions of rate fields exist in particledata.
        """
        print("Leapfrog Integrator Setup.")
        # ensure previous rate fields exist for storing rates from the previous step
        for field_name in self._rate_fields:
            prev_field_name = f"{field_name}_prev"
            if prev_field_name not in pd.get_attribute_names():
                # if a '_prev' field doesn't exist, define it based on the original rate field
                original_shape, original_dtype, backends = pd._attr_definitions[field_name]
                pd._attr_definitions[prev_field_name] = (original_shape, original_dtype, backends)
                full_shape = (pd.N,) + original_shape
                pd._data_cpu[prev_field_name] = np.zeros(full_shape, dtype=original_dtype)
                pd._location[prev_field_name] = "cpu"
                pd._gpu_copies[prev_field_name] = {}
                pd._gpu_dirty[prev_field_name] = False
                pd._locked_writeable[prev_field_name] = False
            else:
                 # if it exists, zero it out
                 try:
                     rate_prev_write = pd.get(prev_field_name, "cpu", writeable=True)
                     rate_prev_write.fill(0.0)
                     pd.release_writeable(prev_field_name)
                 except Exception as e_zero:
                      print(f"  Warn: Failed to zero existing '{prev_field_name}': {e_zero}")

    def step(self, pd: ParticleData, dt: float, forces_callback: callable, config: Optional[Dict] = None, current_step: int = -1, debug_particle_idx: int = -1):
        """
        performs one leapfrog integration step.
        follows the kick-drift-kick (kdk) scheme:
        1. kick 1: update velocities by dt/2 using forces at time t.
        2. drift: update positions by dt using velocities at t+dt/2.
        3. force/rate calculation: compute forces and energy rates at new positions (t+dt).
        4. energy update: update internal energies using averaged rates over dt.
        5. kick 2: update velocities by another dt/2 using forces at t+dt.
        """
        pd.ensure("positions velocities forces masses internal_energies", "cpu") # ensure core arrays are on cpu

        # store previous energy rates (rates at time t)
        for field_name in self._rate_fields:
            prev_field_name = f"{field_name}_prev"
            try:
                current_rate_cpu = pd.get(field_name, "cpu")
                if current_rate_cpu is not None:
                    pd.set(prev_field_name, current_rate_cpu, source_device="cpu")
            except Exception as e: print(f"      ERROR storing prev rate '{field_name}': {e}")

        # kick 1: v(t) -> v(t+dt/2)
        try:
            pd.ensure("forces", "cpu") # ensure forces are current on cpu
            forces = pd.get("forces", "cpu"); masses = pd.get("masses", "cpu")
            velocities = pd.get("velocities", "cpu") # current velocities v(t)
            v_half_temp = pd.get("v_half_temp", "cpu", writeable=True) # temporary storage for v(t+dt/2)
            masses_safe = np.maximum(masses, 1e-30) # avoid division by zero for massless particles
            accel = forces / masses_safe[:, np.newaxis] # a(t) = F(t)/m
            v_half_temp[:] = velocities + accel * (dt * 0.5) # v(t+dt/2) = v(t) + a(t) * dt/2
            pd.release_writeable("v_half_temp")
        except Exception as e:
            print(f"      ERROR during Kick 1: {e}"); traceback.print_exc()
            if pd._locked_writeable.get("v_half_temp", False): pd.release_writeable("v_half_temp")

        # drift: x(t) -> x(t+dt)
        try:
            v_half = pd.get("v_half_temp", "cpu") # get v(t+dt/2)
            positions = pd.get("positions", "cpu", writeable=True) # current positions x(t)
            positions[:] += v_half * dt # x(t+dt) = x(t) + v(t+dt/2) * dt
            pd.release_writeable("positions")
        except Exception as e:
            print(f"      ERROR during Drift: {e}"); traceback.print_exc()
            if pd._locked_writeable.get("positions", False): pd.release_writeable("positions")

        # force/rate calculation: f(t+dt) and rates(t+dt)
        # this callback will update forces and energy rates based on new positions x(t+dt)
        forces_callback(debug_particle_idx=debug_particle_idx)

        # energy update: u(t) -> u(t+dt)
        try:
            # get current rates (rates at t+dt) and previous rates (rates at t)
            rates_now = {name: pd.get(name, "cpu") for name in self._rate_fields}
            rates_prev = {name: pd.get(f"{name}_prev", "cpu") for name in self._rate_fields}
            if any(v is None for v in rates_now.values()) or any(v is None for v in rates_prev.values()):
                missing_now = [k for k,v in rates_now.items() if v is None]
                missing_prev = [k for k,v in rates_prev.items() if v is None]
                raise ValueError(f"Missing rate arrays! Now: {missing_now}, Prev: {missing_prev}")
            # average the rates over the timestep
            avg_rates = {name: 0.5 * (rates_prev[name].astype(np.float64) + rates_now[name].astype(np.float64))
                         for name in self._rate_fields}
            # calculate average rate of change of specific internal energy (du/dt)
            du_dt_avg = (avg_rates["work_terms"] + avg_rates["visc_heating_terms"] + avg_rates["fusion_rates"]
                         - avg_rates["cooling_rates"])
            internal_energies = pd.get("internal_energies", "cpu", writeable=True) # u_total(t)
            masses = pd.get("masses", "cpu")
            delta_U_total = masses * du_dt_avg * dt # change in total internal energy per particle
            internal_energies[:] += delta_U_total.astype(internal_energies.dtype, copy=False) # u_total(t+dt) = u_total(t) + dU_total
            # apply a floor to internal energy based on minimum temperature
            min_temp = config.get('min_temperature', 0.0)
            cv_sim = config.get('cv', None)
            if cv_sim is not None and cv_sim > 0 and min_temp > 0:
                 min_U_floor = masses * cv_sim * min_temp
                 internal_energies[:] = np.maximum(internal_energies, min_U_floor)
            pd.release_writeable("internal_energies")
        except Exception as e:
            print(f"      ERROR during Energy Update: {e}"); traceback.print_exc()
            if pd._locked_writeable.get("internal_energies", False): pd.release_writeable("internal_energies")

        # kick 2: v(t+dt/2) -> v(t+dt)
        try:
            pd.ensure("forces v_half_temp", "cpu") # ensure forces (at t+dt) and v_half are on cpu
            forces = pd.get("forces", "cpu"); masses = pd.get("masses", "cpu")
            v_half = pd.get("v_half_temp", "cpu") # v(t+dt/2)
            velocities = pd.get("velocities", "cpu", writeable=True) # this will become v(t+dt)
            masses_safe = np.maximum(masses, 1e-30)
            accel = forces / masses_safe[:, np.newaxis] # a(t+dt) = F(t+dt)/m
            velocities[:] = v_half + accel * (dt * 0.5) # v(t+dt) = v(t+dt/2) + a(t+dt) * dt/2
            pd.release_writeable("velocities")
        except Exception as e:
             print(f"      ERROR during Kick 2: {e}"); traceback.print_exc()
             if pd._locked_writeable.get("velocities", False): pd.release_writeable("velocities")