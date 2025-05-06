# prometheus/sim/physics/thermo/thermo_numba.py
"""
Thermodynamics model implementation using Numba-accelerated kernels for CPU execution.

Handles Equation of State (EOS) calculations (Temperature, Pressure from density
and internal energy) and computes energy change rates from cooling and fusion processes.
Relies on simulation-specific constants (Cv, R_sim) passed via configuration.
"""

import numpy as np
import traceback
from typing import Dict, Optional

# Absolute imports from project structure
from sim.physics.base.thermo import ThermodynamicsModel
from sim.particle_data import ParticleData
# Physical constant needed only for optional radiation pressure
from sim.constants import CONST_a_rad
from sim.utils import timing_decorator

# --- Numba Import and Kernel Definitions ---
try:
    from numba import njit, prange, float64, boolean, int64
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    # Define dummy decorators if Numba is not available, allowing import but failing at runtime
    njit = lambda *args, **kwargs: lambda func: func # type: ignore
    prange = range # type: ignore

# --- Numba Kernels ---
# These kernels perform the core calculations, optimized for CPU execution.
# They operate on float64 arrays for internal precision.

@njit(float64[:](float64[:], float64[:], float64, float64), cache=True, fastmath=True, parallel=True)
def calculate_temperature_numba(u_total_np: np.ndarray, mass_np: np.ndarray,
                                cv_sim_f64: float, min_temp_f64: float) -> np.ndarray:
    """(Numba Kernel) Calculates Temperature T = (U / mass) / cv_sim."""
    N = u_total_np.shape[0]
    T_np = np.empty(N, dtype=float64)
    cv_safe = max(cv_sim_f64, 1e-50) # Avoid division by zero for Cv
    mass_safe_floor = 1e-50 # Avoid division by zero for mass

    for i in prange(N): # Parallel loop over particles
        mass_safe = max(mass_np[i], mass_safe_floor)
        u_specific = u_total_np[i] / mass_safe # u = U/m (specific internal energy)
        temp_calc = u_specific / cv_safe      # T = u / Cv (temperature calculation)
        T_np[i] = max(min_temp_f64, temp_calc) # Apply minimum temperature floor
    return T_np

@njit(float64[:](float64[:], float64[:], float64, float64, boolean, int64), cache=True, fastmath=True, parallel=True)
def calculate_pressure_numba(rho_np: np.ndarray, T_np: np.ndarray, gas_constant_sim_f64: float,
                             density_floor_f64: float, use_rad_press_bool: bool,
                             debug_particle_idx: int) -> np.ndarray: # Removed unused debug index from signature
    """(Numba Kernel) Calculates Pressure P = P_gas + P_rad."""
    N = rho_np.shape[0]
    P_np = np.empty(N, dtype=float64)
    one_third_a_rad = CONST_a_rad / 3.0 # Precompute radiation constant factor
    temp_safe_floor = 1e-15 # Floor for temperature in Prad calculation

    for i in prange(N): # Parallel loop
        rho_safe = max(rho_np[i], density_floor_f64) # Apply density floor
        temp_safe = T_np[i] # Temperature already floored in T calculation

        # Gas Pressure: P_gas = rho * R_sim * T
        Pgas = rho_safe * gas_constant_sim_f64 * temp_safe

        # Radiation Pressure (optional): P_rad = a/3 * T^4
        Prad = 0.0
        if use_rad_press_bool:
            temp_for_rad = max(temp_safe, temp_safe_floor) # Ensure T > 0 for T^4
            Prad = one_third_a_rad * (temp_for_rad**4)

        P_np[i] = Pgas + Prad # Total pressure
        # Note: Printing from parallel Numba kernels is generally discouraged/unreliable
    return P_np

@njit(float64[:](float64[:], float64[:], float64, boolean, float64, float64), cache=True, fastmath=True, parallel=True)
def calculate_cooling_rate_numba(rho_np: np.ndarray, T_np: np.ndarray, density_floor_f64: float,
                                 use_cooling_bool: bool, cooling_coeff_f64: float,
                                 cooling_beta_f64: float) -> np.ndarray:
    """(Numba Kernel) Calculates specific cooling rate du/dt (power law)."""
    N = rho_np.shape[0]
    cool_rate_np = np.zeros(N, dtype=float64) # Initialize with zeros
    if not use_cooling_bool or cooling_coeff_f64 <= 0: return cool_rate_np # Skip if disabled

    temp_safe_floor = 1e-9 # Floor for temperature in power law

    for i in prange(N): # Parallel loop
        rho_safe = max(rho_np[i], density_floor_f64)
        temp_safe = max(T_np[i], temp_safe_floor)
        # Cooling rate = coeff * rho * T^beta
        cool_rate_np[i] = cooling_coeff_f64 * rho_safe * (temp_safe ** cooling_beta_f64)
    return cool_rate_np

# <<< MODIFIED KERNEL SIGNATURE AND IMPLEMENTATION >>>
@njit(float64[:](
    float64[:], float64[:], float64[:], float64[:], # rho, T, u_total, mass
    float64, boolean, float64, float64, float64,    # density_floor, use_fusion, thresh, coeff, power
    float64, float64                                 # dt_simulation, max_rel_increase
    ), cache=True, fastmath=True, parallel=True)
def calculate_fusion_rate_numba(
    rho_np: np.ndarray, T_np: np.ndarray,
    internal_energies_np: np.ndarray, masses_np: np.ndarray,
    density_floor_f64: float,
    use_fusion_bool: bool, fusion_thresh_f64: float,
    fusion_coeff_f64: float, fusion_power_f64: float,
    dt_simulation_f64: float, max_rel_increase_f64: float
    ) -> np.ndarray:
    """(Numba Kernel) Calculates specific fusion heating rate du/dt, with capping."""
    N = rho_np.shape[0]
    fusion_rate_np = np.zeros(N, dtype=float64) # Initialize with zeros
    if not use_fusion_bool or fusion_coeff_f64 <= 0: return fusion_rate_np # Skip if disabled

    mass_safe_floor = 1e-50
    dt_safe = max(dt_simulation_f64, 1e-30) # Avoid division by zero for dt

    for i in prange(N): # Parallel loop
        if T_np[i] >= fusion_thresh_f64: # Check temperature threshold
            rho_safe = max(rho_np[i], density_floor_f64)
            # Calculate uncapped fusion rate
            fusion_rate_uncapped = fusion_coeff_f64 * rho_safe * (T_np[i] ** fusion_power_f64)

            if fusion_rate_uncapped > 0 and max_rel_increase_f64 > 0:
                mass_safe = max(masses_np[i], mass_safe_floor)
                u_specific_current = internal_energies_np[i] / mass_safe

                if u_specific_current > 1e-30: # Only apply relative cap if current energy is non-trivial
                    # Max allowed specific energy increase in one dt: du_spec_max = factor * u_spec_current
                    # Corresponding max rate: du_spec_max / dt
                    max_allowed_specific_rate = (max_rel_increase_f64 * u_specific_current) / dt_safe
                    fusion_rate_np[i] = min(fusion_rate_uncapped, max_allowed_specific_rate)
                else:
                    # If u_specific is tiny, uncapped rate might be very large relative to it.
                    # However, T is high (>= fusion_thresh), so u_specific should be T*Cv.
                    # This path implies Cv is pathologically small or an error state.
                    # For now, allow uncapped rate here, relying on T_thresh.
                    # Consider an absolute rate cap if this becomes an issue.
                    fusion_rate_np[i] = fusion_rate_uncapped
            else: # No fusion or cap factor is zero
                fusion_rate_np[i] = fusion_rate_uncapped
        # else: fusion_rate_np[i] remains 0.0
    return fusion_rate_np


# --- Python Class Definition ---

class ThermoNumba(ThermodynamicsModel):
    """Thermodynamics model using Numba kernels for EOS and energy change rates."""

    def setup(self, pd: ParticleData):
        """Validates config and stores necessary parameters."""
        if not HAVE_NUMBA: raise ImportError("Numba is required for ThermoNumba but not found.")
        super().setup(pd) # Call base class setup
        self.N = pd.get_n()

        # Pre-extract and validate config parameters needed by kernels
        try:
            # Simulation constants (must be present)
            self.gas_constant_sim = float(self.config['gas_constant_sim'])
            self.cv_sim = float(self.config['cv']) # Simulation Cv [Energy/Mass/Temp]
            self.dt_simulation = float(self.config['dt']) # Simulation timestep for capping
            # Basic parameters
            self.min_temperature = float(self.config['min_temperature'])
            self.density_floor = float(self.config.get('density_floor', 1e-9))
            self.use_rad_press = bool(self.config['use_rad_press'])
            # Cooling parameters
            self.use_cooling = bool(self.config['use_cooling'])
            self.cooling_coeff = float(self.config['cooling_coeff'])
            self.cooling_beta = float(self.config['cooling_beta'])
            # Fusion parameters
            self.use_fusion = bool(self.config['use_fusion'])
            self.fusion_thresh = float(self.config['fusion_thresh'])
            self.fusion_coeff = float(self.config['fusion_coeff'])
            self.fusion_power = float(self.config['fusion_power'])
            # <<< READ NEW FUSION CAPPING PARAMETER >>>
            self.fusion_max_rel_increase = float(self.config['FUSION_MAX_SPECIFIC_ENERGY_INCREASE_PER_DT'])


            # Store target NumPy dtype for casting results
            self.target_dtype = pd.get_numpy_dtype("temperatures")

            # Sanity checks
            if self.cv_sim <= 0: raise ValueError("'cv' must be positive.")
            if self.gas_constant_sim <= 0: raise ValueError("'gas_constant_sim' must be positive.")
            if self.dt_simulation <= 0: raise ValueError("'dt' must be positive for fusion capping.")
            if not (0 <= self.fusion_max_rel_increase <= 1.0): # Allow 0 to disable cap effect, 1 for 100%
                print(f"Warning: FUSION_MAX_SPECIFIC_ENERGY_INCREASE_PER_DT ({self.fusion_max_rel_increase}) is outside typical [0,1] range.")


        except KeyError as e: raise ValueError(f"Missing required config key for ThermoNumba: {e}") from e
        except (ValueError, TypeError) as e: raise ValueError(f"Invalid config value for ThermoNumba: {e}") from e

        # Log setup parameters
        print(f"ThermoNumba Setup: N={self.N}, SimGasConst={self.gas_constant_sim:.2e}, SimCv={self.cv_sim:.2e}, MinT={self.min_temperature:.1f}")
        print(f"  Cooling: Use={self.use_cooling}, Coeff={self.cooling_coeff:.1e}, Beta={self.cooling_beta:.2f}")
        print(f"  Fusion: Use={self.use_fusion}, Thresh={self.fusion_thresh:.1e}, Coeff={self.fusion_coeff:.1e}, Power={self.fusion_power:.1f}")
        print(f"  Fusion Cap: Max Rel Increase per dt = {self.fusion_max_rel_increase:.2f}")
        print(f"  RadPress: {self.use_rad_press}, DensityFloor: {self.density_floor:.1e}, Sim_dt for cap: {self.dt_simulation:.2e}")


    # @timing_decorator # Uncomment for timing
    def update_eos_quantities(self, pd: ParticleData, debug_particle_idx: int = -1):
        """Calculates Temperature and Pressure using Numba EOS kernels."""
        if self.N == 0: return # Skip if no particles

        idx = debug_particle_idx if 0 <= debug_particle_idx < self.N else -1 # Validate debug index

        try:
            # Ensure required input data is on CPU
            pd.ensure("internal_energies densities masses", "cpu")
            u_np_orig = pd.get("internal_energies", "cpu") # Total U
            rho_np_orig = pd.get("densities", "cpu")
            mass_np_orig = pd.get("masses", "cpu")

            # --- Prepare inputs for Numba kernels (use float64 for precision) ---
            u_f64 = u_np_orig.astype(np.float64, copy=False)
            rho_f64 = rho_np_orig.astype(np.float64, copy=False)
            mass_f64 = mass_np_orig.astype(np.float64, copy=False)
            cv_sim_f64 = float64(self.cv_sim)
            gas_const_sim_f64 = float64(self.gas_constant_sim)
            min_temp_f64 = float64(self.min_temperature)
            density_floor_f64 = float64(self.density_floor)
            use_rad_press_bool = boolean(self.use_rad_press)

            # --- Debug Print Inputs (Optional) ---
            if idx != -1:
                print(f"\n--- ThermoNumba EOS Update | DEBUG PARTICLE {idx} ---")
                print(f"  IN : u={u_f64[idx]:.6e}, rho={rho_f64[idx]:.6e}, m={mass_f64[idx]:.6e}")
                print(f"  PARAM: SimCv={cv_sim_f64:.6e}, SimRg={gas_const_sim_f64:.2e}, MinT={min_temp_f64:.1f}, DensF={density_floor_f64:.1e}, RadP={use_rad_press_bool}")

            # --- Call Numba Kernels ---
            temp_f64 = calculate_temperature_numba(u_f64, mass_f64, cv_sim_f64, min_temp_f64)
            press_f64 = calculate_pressure_numba(rho_f64, temp_f64, gas_const_sim_f64, density_floor_f64, use_rad_press_bool, int64(idx))

            # --- Debug Print Outputs (Optional) ---
            if idx != -1:
                rho_safe_dbg = max(rho_f64[idx], density_floor_f64); temp_safe_dbg = temp_f64[idx]
                pg_dbg = rho_safe_dbg * gas_const_sim_f64 * temp_safe_dbg
                pr_dbg = (CONST_a_rad / 3.0) * (max(temp_safe_dbg, 1e-15)**4) if use_rad_press_bool else 0.0
                print(f"  OUT: T={temp_f64[idx]:.6e} K")
                print(f"  OUT: P_gas={pg_dbg:.4e}, P_rad={pr_dbg:.4e}, P_tot={press_f64[idx]:.4e}")
                print(f"------------------------------------------------")

            # --- Set results back into ParticleData (casting to target dtype) ---
            pd.set("temperatures", temp_f64.astype(self.target_dtype, copy=False), source_device="cpu")
            pd.set("pressures", press_f64.astype(self.target_dtype, copy=False), source_device="cpu")

            # Optional: Print summary stats after update
            # print(f"  Thermo EOS Summary: T [{np.min(temp_f64):.2e}, {np.max(temp_f64):.2e}] K, P [{np.min(press_f64):.2e}, {np.max(press_f64):.2e}]")

        except Exception as e:
            print(f"ERROR during Numba EOS update: {e}"); traceback.print_exc()


    # @timing_decorator # Uncomment for timing
    def compute_energy_changes(self, pd: ParticleData, debug_particle_idx: int = -1):
        """Calculates cooling and fusion energy change rates using Numba."""
        if self.N == 0: return # Skip if no particles
        idx = debug_particle_idx if 0 <= debug_particle_idx < self.N else -1 # Validate debug index

        try:
            # Ensure required input data is on CPU
            # <<< ADD internal_energies and masses FOR FUSION CAPPING >>>
            pd.ensure("densities temperatures internal_energies masses", "cpu")
            rho_np_orig = pd.get("densities", "cpu")
            temp_np_orig = pd.get("temperatures", "cpu")
            u_total_np_orig = pd.get("internal_energies", "cpu")
            masses_np_orig = pd.get("masses", "cpu")


            # --- Prepare inputs for Numba kernels (use float64) ---
            rho_f64 = rho_np_orig.astype(np.float64, copy=False)
            temp_f64 = temp_np_orig.astype(np.float64, copy=False)
            u_total_f64 = u_total_np_orig.astype(np.float64, copy=False) # For fusion cap
            masses_f64 = masses_np_orig.astype(np.float64, copy=False)   # For fusion cap

            density_floor_f64 = float64(self.density_floor)
            use_cooling_bool = boolean(self.use_cooling)
            cooling_coeff_f64 = float64(self.cooling_coeff)
            cooling_beta_f64 = float64(self.cooling_beta)
            use_fusion_bool = boolean(self.use_fusion)
            fusion_thresh_f64 = float64(self.fusion_thresh)
            fusion_coeff_f64 = float64(self.fusion_coeff)
            fusion_power_f64 = float64(self.fusion_power)
            # <<< PASS NEW PARAMS FOR FUSION CAPPING >>>
            dt_simulation_f64 = float64(self.dt_simulation)
            fusion_max_rel_increase_f64 = float64(self.fusion_max_rel_increase)


            # --- Debug Print Inputs (Optional) ---
            if idx != -1:
                 print(f"\n--- ThermoNumba Energy Changes | DEBUG PARTICLE {idx} ---")
                 print(f"  IN : rho={rho_f64[idx]:.4e}, T={temp_f64[idx]:.4e}, u_tot={u_total_f64[idx]:.4e}, m={masses_f64[idx]:.4e}")
                 print(f"  PARAM: useCool={use_cooling_bool}, coolCoeff={cooling_coeff_f64:.2e}, coolBeta={cooling_beta_f64:.2f}")
                 print(f"  PARAM: useFus={use_fusion_bool}, fusThresh={fusion_thresh_f64:.2e}, fusCoeff={fusion_coeff_f64:.2e}, fusPower={fusion_power_f64:.1f}")
                 print(f"  PARAM: dt_sim={dt_simulation_f64:.2e}, fusMaxRelInc={fusion_max_rel_increase_f64:.2f}")


            # --- Call Numba Kernels ---
            cool_f64 = calculate_cooling_rate_numba(rho_f64, temp_f64, density_floor_f64, use_cooling_bool, cooling_coeff_f64, cooling_beta_f64)
            # <<< CALL MODIFIED FUSION KERNEL >>>
            fusion_f64 = calculate_fusion_rate_numba(
                rho_f64, temp_f64, u_total_f64, masses_f64,
                density_floor_f64, use_fusion_bool, fusion_thresh_f64,
                fusion_coeff_f64, fusion_power_f64,
                dt_simulation_f64, fusion_max_rel_increase_f64
            )

            # --- Debug Print Outputs (Optional) ---
            if idx != -1:
                print(f"  OUT: Cooling Rate = {cool_f64[idx]:.6e}")
                print(f"  OUT: Fusion Rate  = {fusion_f64[idx]:.6e}")
                # Could also print uncapped fusion rate here if kernel was modified to return it for debug
                print(f"--------------------------------------------------")

            # --- Set results back into ParticleData (casting to target dtype) ---
            pd.set("cooling_rates", cool_f64.astype(self.target_dtype, copy=False), source_device="cpu")
            pd.set("fusion_rates", fusion_f64.astype(self.target_dtype, copy=False), source_device="cpu")

            # Optional: Print summary stats after update
            # if self.use_cooling: print(f"  Thermo CoolRate Summary: [{np.min(cool_f64):.2e}, {np.max(cool_f64):.2e}]")
            # if self.use_fusion: print(f"  Thermo FusRate Summary : [{np.min(fusion_f64):.2e}, {np.max(fusion_f64):.2e}]")

        except Exception as e:
            print(f"ERROR during Numba energy change calculation: {e}"); traceback.print_exc()