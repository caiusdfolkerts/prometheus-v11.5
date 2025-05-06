

# prometheus/config/default_settings.py
import numpy as np

# IMPORTANT NOTE: IF USING GPU ON METAL API, USE_DOUBLE_PRECISION MUST BE FALSE
# SECOND IMPORTANT NOTE: IF WANT SPH TAICHI TO RUN ON CPU, USE DEFAULT_TAICHI_ARCH = CPU

DEFAULT_SETTINGS = {
    # --- Simulation Control ---
    'dt': 5e-5,               # Simulation time step
    'max_steps': -1,            # Maximum number of steps (-1 for unlimited)
    'max_time': -1,           # Maximum simulation time (-1 for unlimited)
    'start_running': False,     # Start simulation immediately?
    'USE_DOUBLE_PRECISION': False,# User preference, effective value determined in main.py
    'default_taichi_arch': 'gpu', # Preferred Taichi backend arch

    # --- Initial Conditions ---
    'N': 10_000,                 # Number of particles
    'radius': 10.0,             # Initial distribution radius (SimUnits)
    'v_scale': 5.0,             # Initial velocity scaling factor (SimUnits)
    'mass_min': 0.8,            # Minimum particle mass (SimUnits)
    'mass_max': 1.2,            # Maximum particle mass (SimUnits)
    'initial_temp': 2000.0,     # Initial particle temperature (Kelvin)

    # --- Gravity Parameters ---
    'G': 60.0,                   # Gravitational constant (SimUnits) - Set to 0 for hydro tests
    'softening': 1.0,          # Gravitational softening length (SimUnits)
    'bh_theta': 0.5,            # Barnes-Hut opening angle
    'MAX_NODES_FACTOR': 10,     # BH Tree node allocation factor (N * factor)

    # --- SPH Parameters ---
    'h': 2.0,                   # SPH smoothing length (SimUnits)
    'sph_density_floor': 1e-9,  # Simulation density units floor
    'sph_eos_gamma': 5/3,       # Adiabatic index (unitless) - Used for Cv calculation
    'sph_visc_alpha': 0.0,      # SPH viscosity alpha (default=off)
    'sph_visc_beta': 0.0,   
    'rotation_strength': 2.0,

    # --- Thermodynamics Parameters (Simulation Units Focus) ---
    'mu': 0.6,                  # Mean molecular weight (unitless, used only for Prad if enabled)
    'gas_constant_sim': 3.0,    # SIMULATION UNITS
    'min_temperature': 10.0,    
    'density_floor': 1e-9,      
    'use_rad_press': True,   
    'use_cooling': True,
    'cooling_coeff': 1e-3,
    'FUSION_MAX_SPECIFIC_ENERGY_INCREASE_PER_DT': 0.1,
    'cooling_beta': 0.5,
    'use_fusion': True,
    'fusion_thresh': 1e4,
    'fusion_coeff': 1e-24,
    'fusion_power': 5.0,


    # --- Model Defaults ---
    'default_gravity_model': 'gravity_bh_numba',
    'default_sph_model': 'sph_taichi',
    'default_thermo_model': 'thermo_numba',
    'default_color_model': 'color_temp_numba',
    'default_integrator': 'leapfrog',

    # --- UI / Visualization ---
    'color_speed_percentile': 99.5,
    'color_speed_min_floor': 0.0,
    'color_temp_min': 100.0,
    'color_temp_max': 1e6,

    # --- Graphing Settings ---
    'GRAPH_SETTINGS': {
        'enable_plotting': True,
        'plot_interval_steps': 10,
        'output_dir': 'output',
        'clear_data_on_restart': True,
        'plot_energy_components': True,
        'plot_energy_rates': True,
        'plot_angular_momentum': True,
        'plot_com_velocity': True,
        'plot_max_rho': True,
        'plot_max_temp': True,
        'plot_avg_temp': True,
        'plot_min_max_separation': True,
        'plot_avg_density': True,
        'plot_max_pressure': True,
        'plot_avg_pressure': True,
        'plot_sph_visc_heat_rate': True,
        'plot_num_fusing_particles': True,
        'plot_avg_spec_int_energy': True,
        'plot_bh_nodes': True,
        'plot_step_timing': True,
        'plot_hist_speed': True,
        'plot_hist_temp': True,
        'plot_hist_density': True,
        'plot_hist_spec_int_energy': True,
        'plot_hist_sph_neighbors': True,
        'plot_profile_density': True,
        'plot_profile_temp': True,
        'plot_profile_pressure': True,
        'plot_profile_spec_int_energy': True,
        'plot_phase_T_rho': True,
        'plot_phase_P_rho': True,
        'plot_scatter_cool_temp': True,
        'plot_scatter_fus_temp': True,
        'histogram_bins': 50,
        'radial_profile_bins': 30,
        'phase_plot_sample_frac': 0.2,
    },
    # 'L' calculated below
}


# --- Calculate Derived Defaults ---

# Calculate 'L' based on 'radius'
radius_val = DEFAULT_SETTINGS.get('radius')
if radius_val is not None:
    calculated_L = float(radius_val) * 2.0 * 1.1
    DEFAULT_SETTINGS['L'] = calculated_L
    print(f"DEFAULT_SETTINGS: Calculated L = {DEFAULT_SETTINGS['L']:.1f} from radius={radius_val:.1f}")
else:
    print("Warning: Cannot calculate default 'L'. Using fallback L=50.0")
    DEFAULT_SETTINGS['L'] = 50.0
    
if 'gas_constant_sim' in DEFAULT_SETTINGS and 'sph_eos_gamma' in DEFAULT_SETTINGS:
    R_sim_val = DEFAULT_SETTINGS['gas_constant_sim']
    gamma_val = DEFAULT_SETTINGS['sph_eos_gamma']
    if R_sim_val > 1e-15 and abs(gamma_val - 1.0) > 1e-9:
        cv_sim_calculated = R_sim_val / (gamma_val - 1.0)
        DEFAULT_SETTINGS['cv'] = cv_sim_calculated # Set the simulation Cv
        print(f"DEFAULT_SETTINGS: Calculated Simulation Cv = {cv_sim_calculated:.3e} from R_sim={R_sim_val:.3e}, gamma={gamma_val:.2f}")
    else:
        raise ValueError("Cannot calculate Simulation Cv from invalid R_sim/gamma in default settings.")
else:
    raise ValueError("Missing 'gas_constant_sim' or 'sph_eos_gamma' needed to calculate Simulation Cv.")


# --- Validation ---
if 'cv' not in DEFAULT_SETTINGS:
     raise ValueError("Failed to set or derive 'cv' (Simulation Specific Heat Capacity) in default settings.")

# Ensure density floor consistency
if 'density_floor' not in DEFAULT_SETTINGS and 'sph_density_floor' in DEFAULT_SETTINGS:
     DEFAULT_SETTINGS['density_floor'] = DEFAULT_SETTINGS['sph_density_floor']
elif 'sph_density_floor' not in DEFAULT_SETTINGS and 'density_floor' in DEFAULT_SETTINGS:
     DEFAULT_SETTINGS['sph_density_floor'] = DEFAULT_SETTINGS['density_floor']
elif 'density_floor' not in DEFAULT_SETTINGS and 'sph_density_floor' not in DEFAULT_SETTINGS:
     print("Warning: Missing density_floor/sph_density_floor, defaulting to 1e-9")
     DEFAULT_SETTINGS['density_floor'] = 1e-9
     DEFAULT_SETTINGS['sph_density_floor'] = 1e-9

# --- Model/Integrator Validation ---
required_models = ['default_gravity_model', 'default_sph_model', 'default_thermo_model', 'default_color_model']
required_integrator = 'default_integrator'
if not all(key in DEFAULT_SETTINGS for key in required_models + [required_integrator]):
    raise ValueError("Default settings missing one or more required model/integrator IDs.")

# Check if the selected default SPH model ID exists
sph_default_id = DEFAULT_SETTINGS.get('default_sph_model')
sph_model_found = False
if sph_default_id:
    try:
        from config.available_models import AVAILABLE_MODELS
        sph_model_found = any(model_def['id'] == sph_default_id for model_def in AVAILABLE_MODELS.get('sph', []))

        if not sph_model_found:
            print(f"DEFAULT_SETTINGS FATAL ERROR: Default SPH model ID '{sph_default_id}' not found among AVAILABLE_MODELS['sph'].")
            raise ValueError(f"Default SPH model ID '{sph_default_id}' is not defined in available_models.py")

    except ImportError:
         print("DEFAULT_SETTINGS WARNING: Could not import AVAILABLE_MODELS to validate default SPH model ID existence.")
    except Exception as e:
         print(f"DEFAULT_SETTINGS WARNING: Error validating default SPH model ID existence: {e}")
else:
    print("DEFAULT_SETTINGS FATAL ERROR: 'default_sph_model' key is missing.")
    raise ValueError("'default_sph_model' key missing in DEFAULT_SETTINGS.")

print("Default settings processing finished.")
