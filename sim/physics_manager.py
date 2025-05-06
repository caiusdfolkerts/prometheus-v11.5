# prometheus/sim/physics_manager.py
"""
Manages the selection and execution of different physics models.

This class loads definitions of available physics models (gravity, SPH, thermo,
color), checks their backend requirements against available system capabilities,
allows the Simulator to select active models for each domain, and orchestrates
the execution of these models in the correct physical sequence during a
simulation step. It also handles pre-initialized models provided during restarts.
"""

from typing import Dict, List, Optional, Tuple, Type
import traceback
import numpy as np

# Absolute imports from project structure
from sim.particle_data import ParticleData
from sim.physics.base.physics import PhysicsModel
# Import base classes for type hints (optional but good practice)
from sim.physics.base.gravity import GravityModel
from sim.physics.base.sph import SPHModel
from sim.physics.base.thermo import ThermodynamicsModel
from sim.physics.base.color import ColorModel
# Utilities and configuration
from sim.utils import dynamic_import, check_backend_availability
from config.available_models import AVAILABLE_MODELS

class PhysicsManager:
    """Handles selection, setup, and execution of active physics models."""

    # Define supported physics domains
    MODEL_TYPES = ["gravity", "sph", "thermo", "color"]

    def __init__(self, pd: ParticleData, initial_config: dict, backend_availability_flags: Dict[str, bool]):
        self._pd = pd
        self._config = initial_config.copy()
        self._backend_flags = backend_availability_flags # Store backend status
        self._pre_initialized_models: Dict[str, PhysicsModel] = {}
        self._available_models: Dict[str, Dict[str, Dict]] = {}
        self._active_models: Dict[str, Optional[PhysicsModel]] = {mtype: None for mtype in self.MODEL_TYPES}
        self._active_model_ids: Dict[str, Optional[str]] = {mtype: None for mtype in self.MODEL_TYPES}
        self._load_available_models()

    def set_pre_initialized_models(self, models: Dict[str, PhysicsModel]):
        """Stores pre-initialized model instances (e.g., from main thread during restart)."""
        self._pre_initialized_models = models if models else {}
        if self._pre_initialized_models:
            print(f"PhysicsManager: Received {len(self._pre_initialized_models)} pre-initialized models: {list(self._pre_initialized_models.keys())}")

    def _load_available_models(self):
        for model_type in self.MODEL_TYPES:
            self._available_models[model_type] = {}
            model_list = AVAILABLE_MODELS.get(model_type, [])
            if not isinstance(model_list, list): continue
            for model_def in model_list:
                if not isinstance(model_def, dict) or 'id' not in model_def: continue
                model_id = model_def['id']
                required_backend = model_def.get("required_backend", "numpy")
                backend_ok = check_backend_availability(
                    required_backend,
                    taichi_init_flag=self._backend_flags.get("taichi", False),
                    cupy_avail_flag=self._backend_flags.get("cupy", False)
                )
                model_def['_backend_available'] = backend_ok
                self._available_models[model_type][model_id] = model_def
    
    def get_available_models(self) -> Dict[str, List[Dict]]:
         """Returns available models grouped by type, including availability status."""
         # Provides all loaded models; UI can use '_backend_available' flag to filter/disable
         return {mtype: list(defs.values()) for mtype, defs in self._available_models.items()}

    def select_model(self, model_type: str, model_id: str):
        """Selects and initializes a model, using pre-initialized if available."""
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unknown physics model type: '{model_type}'. Valid: {self.MODEL_TYPES}")

        available_for_type = self._available_models.get(model_type, {})
        if model_id not in available_for_type:
             raise ValueError(f"Unknown model ID '{model_id}' for type '{model_type}'. Available: {list(available_for_type.keys())}")

        model_def = available_for_type[model_id]

        # --- Check 1: Use Pre-initialized Model ---
        pre_init_model = self._pre_initialized_models.get(model_type)
        # Ensure the pre-initialized model ID matches the requested ID
        if pre_init_model and model_def.get('id') == model_id:
            print(f"Using pre-initialized {model_type} model: '{model_id}'")
            old_model = self._active_models.get(model_type)
            if old_model and old_model is not pre_init_model and hasattr(old_model, 'cleanup'):
                try: old_model.cleanup(); print(f"  Cleaned up old '{self._active_model_ids.get(model_type)}' instance.")
                except Exception as e_clean: print(f"  Warn: Error cleaning up old model: {e_clean}")

            self._active_models[model_type] = pre_init_model
            self._active_model_ids[model_type] = model_id
            # Ensure its config is up-to-date
            if hasattr(pre_init_model, 'update_config'): pre_init_model.update_config(self._config)
            # Remove from pre-initialized dict after use
            del self._pre_initialized_models[model_type]
            return # Model selected successfully

        # --- Check 2: Model Already Active? ---
        if self._active_model_ids.get(model_type) == model_id and self._active_models.get(model_type) is not None:
            # Ensure config is up-to-date even if already active
            try: self._active_models[model_type].update_config(self._config)
            except Exception as e_upd: print(f"Warn: Error updating config for active {model_id}: {e_upd}")
            return # No change needed

        # --- Check 3: Backend Available? ---
        if not model_def.get('_backend_available', False):
            required = model_def.get("required_backend", "N/A")
            raise ValueError(f"Cannot select '{model_id}': Required backend '{required}' unavailable.")

        # --- Dynamic Load and Setup ---
        module_name = model_def['module']
        class_name = model_def['class']
        print(f"Selecting {model_type} model: '{model_def.get('name', model_id)}' ({model_id}) [Dynamic Load]...")
        try:
             ModelClass = dynamic_import(module_name, class_name)
             new_model = ModelClass(config=self._config)
             new_model.setup(self._pd)

             # Cleanup the previously active model for this type
             old_model = self._active_models.get(model_type)
             if old_model and hasattr(old_model, 'cleanup'):
                  try: old_model.cleanup(); print(f"  Cleaned up old '{self._active_model_ids.get(model_type)}' instance.")
                  except Exception as e_clean: print(f"Warn: Error cleaning up old model: {e_clean}")

             # Activate the new model
             self._active_models[model_type] = new_model
             self._active_model_ids[model_type] = model_id
             print(f"  Model '{model_id}' for {model_type} selected successfully (Dynamic).")

        except Exception as e:
             print(f"ERROR: Failed to select/setup model '{model_id}': {e}"); traceback.print_exc()
             self._active_models[model_type] = None # Reset on failure
             self._active_model_ids[model_type] = None
             raise # Signal failure

    def compute_all_physics(self, compute_forces: bool = True, compute_thermo: bool = True, compute_color: bool = True, debug_particle_idx: int = -1):
        """Executes active physics models in the correct physical sequence."""
        pd = self._pd # Local shorthand

        # Helper to call compute method if model exists and check for debug arg
        def _call_model_method(model_type, method_name, *args, **kwargs):
            model = self._active_models.get(model_type)
            if model and hasattr(model, method_name):
                try:
                    method = getattr(model, method_name)
                    # Check if method accepts debug_particle_idx
                    takes_debug = 'debug_particle_idx' in method.__code__.co_varnames
                    if takes_debug: kwargs['debug_particle_idx'] = debug_particle_idx
                    elif 'debug_particle_idx' in kwargs: del kwargs['debug_particle_idx'] 
                    # Call the method
                    method(pd, *args, **kwargs)
                except Exception as e:
                    print(f"ERROR during {model_type}.{method_name}: {e}"); traceback.print_exc()
          
        # --- Physics Sequence ---
        # 1. Gravity (Sets initial forces or adds to existing)
        if compute_forces:
            # Ensure forces are zeroed *before* gravity if it's the first force calc
            try: f_write = pd.get("forces", "cpu", writeable=True); f_write.fill(0.0); pd.release_writeable("forces")
            except Exception as e_zero: print(f"Error zeroing forces: {e_zero}")
            _call_model_method("gravity", "compute_forces")

        # 2. SPH Density (Needed before EOS and SPH forces)
        _call_model_method("sph", "compute_density")

        # 3. Thermodynamics EOS Update (T, P from rho, u)
        if compute_thermo: _call_model_method("thermo", "update_eos_quantities")

       # 4. SPH Forces (Pressure + Viscosity) & Work/Heating Rates
        # Adds SPH forces to any existing forces (gravity). Calculates work terms.
        if compute_forces: _call_model_method("sph", "compute_pressure_force")

        # 5. Thermodynamics Energy Change Rates (Cooling, Fusion)
        if compute_thermo: _call_model_method("thermo", "compute_energy_changes")

        # 6. Color Calculation (Based on current state, e.g., T or rho)
        if compute_color: _call_model_method("color", "compute_colors")

    def get_active_models_info(self) -> Dict[str, Optional[Dict]]:
        """Returns definition dictionaries for currently active models, including status."""
        return {
            mtype: self._available_models.get(mtype, {}).get(mid)
            for mtype, mid in self._active_model_ids.items() if mid # Only include if ID is set
        }

    def update_config(self, new_config: dict):
         """Updates internal config and propagates relevant changes to active models."""
         self._config.update(new_config)
         for model_instance in self._active_models.values():
              if model_instance and hasattr(model_instance, 'update_config'):
                   try: model_instance.update_config(self._config)
                   except Exception as e: print(f"Warn: Error updating config for {type(model_instance).__name__}: {e}")

    def cleanup_models(self):
         """Calls cleanup on all currently active model instances."""
         print("Cleaning up active physics models...")
         cleaned_count = 0
         for model_type, model_instance in self._active_models.items():
              if model_instance and hasattr(model_instance, 'cleanup'):
                   try: model_instance.cleanup(); cleaned_count += 1
                   except Exception as e: print(f"Warn: Error cleaning up '{model_type}' model: {e}")
         self._active_models = {mtype: None for mtype in self.MODEL_TYPES}
         self._active_model_ids = {mtype: None for mtype in self.MODEL_TYPES}
         print(f"Cleaned up {cleaned_count} active models.")