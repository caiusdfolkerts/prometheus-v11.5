"""
This code manages the selection and execution of time integration schemes.

It loads available integrator definitions, allows the simulator to select
an active integrator (e.g., leapfrog, yoshida4), and provides an interface to
advance the simulation state by one time step using the chosen integrator.
it handles dynamic importing and setup of the integrator instances.
"""

from typing import Dict, List, Optional, Callable
import traceback

# absolute imports from project structure
from sim.integrators.base import Integrator
from sim.particle_data import ParticleData
from sim.utils import dynamic_import
from config.available_integrators import AVAILABLE_INTEGRATORS

class IntegratorManager:
    """handles selection and execution of the active time integrator."""

    def __init__(self, pd: ParticleData, forces_cb: Callable[[], None], config: dict):
        """
        initializes the IntegratorManager
        """
        self._pd = pd # particle data instance
        self._recompute_forces_callback = forces_cb # callback to recompute forces
        self._config = config # simulation configuration
        self._available_integrators: Dict[str, Dict] = {} # dictionary of available integrator definitions
        self._active_integrator: Optional[Integrator] = None # currently active integrator instance
        self._active_integrator_id: Optional[str] = None # id of the currently active integrator

        self._load_available_integrators() # load definitions on initialization
        print(f"IntegratorManager Initialized. Available: {list(self._available_integrators.keys())}")

    def get_available_integrators(self) -> List[Dict]:
        """returns the list of available integrator definitions."""
        # return a list copy to prevent external modification
        return list(self._available_integrators.values())

    def _load_available_integrators(self):
        """Loads integrator definitions from available_integrators."""
        print("IntegratorManager: Loading integrator definitions...")
        try:
            # create a dictionary mapping integrator id to its definition
            self._available_integrators = {
                integrator_def['id']: integrator_def
                for integrator_def in AVAILABLE_INTEGRATORS
            }
            if not self._available_integrators:
                print("Warning: No integrators found in AVAILABLE_INTEGRATORS config.")
        except KeyError as e_key:
            print(f"ERROR: Integrator definition missing 'id' key: {e_key}")
            raise ValueError("Invalid integrator definition (missing 'id')") from e_key
        except Exception as e:
            print(f"ERROR processing AVAILABLE_INTEGRATORS: {e}")
            traceback.print_exc()
            self._available_integrators = {} # clear on error

    def select_integrator(self, integrator_id: str):
        """selects and initializes the specified integrator."""
        if integrator_id not in self._available_integrators:
            available_ids = list(self._available_integrators.keys())
            raise ValueError(f"Unknown integrator ID: '{integrator_id}'. Available: {available_ids}")

        # avoid re-selecting the same integrator unnecessarily
        if self._active_integrator_id == integrator_id and self._active_integrator is not None:
            # optionally update config even if already active
            if hasattr(self._active_integrator, 'update_config'): # check before calling
                 self._active_integrator.update_config(self._config)
            return

        integrator_def = self._available_integrators[integrator_id]
        module_name = integrator_def['module']
        class_name = integrator_def['class']

        print(f"Selecting integrator: '{integrator_def['name']}' ({integrator_id})...")
        try:
            # dynamically import the class
            IntegratorClass = dynamic_import(module_name, class_name)
            # instantiate the new integrator
            new_integrator = IntegratorClass()

            # always call setup, passing particledata and the *current* config.
            # the base integrator class handles the config argument gracefully.
            new_integrator.setup(self._pd, self._config)

            # cleanup the previous integrator instance if it exists
            if self._active_integrator and hasattr(self._active_integrator, 'cleanup'):
                try: self._active_integrator.cleanup()
                except Exception as e_clean: print(f"Warn: Error cleaning up old integrator: {e_clean}")

            # activate the new integrator
            self._active_integrator = new_integrator
            self._active_integrator_id = integrator_id
            print(f"Integrator '{integrator_id}' selected successfully.")

        except Exception as e:
            print(f"ERROR: Failed to select/setup integrator '{integrator_id}': {e}")
            traceback.print_exc() # print detailed error traceback
            # reset to no active integrator on failure
            self._active_integrator = None
            self._active_integrator_id = None
            raise # re-raise the exception to signal failure

    def get_active_integrator_info(self) -> Optional[Dict]:
         """returns the definition dictionary of the currently active integrator, or none."""
         return self._available_integrators.get(self._active_integrator_id) if self._active_integrator_id else None

    def advance(self, dt: float, current_step: int, debug_particle_idx: int = -1):
        """advances the simulation by one timestep using the active integrator."""
        if not self._active_integrator:
            raise RuntimeError("Attempted to advance simulation without an active integrator.")

        # call the step method of the active integrator
        # ensure the order matches the definition in integrator.step and leapfrog.step:
        # (self, pd, dt, forces_callback, config, current_step, debug_particle_idx)
        self._active_integrator.step(
            self._pd,                       # pd
            dt,                             # dt
            self._recompute_forces_callback,# forces_cb
            self._config,                   # config
            current_step,                   # current_step
            debug_particle_idx              # debug_particle_idx
        )


    def update_config(self, new_config: dict):
        """updates the manager's config and propagates to the active integrator."""
        self._config.update(new_config)
        # re-run setup on the active integrator to apply config changes
        if self._active_integrator and hasattr(self._active_integrator, 'setup'):
            # the setup method should handle receiving the updated config
            self._active_integrator.setup(self._pd, self._config)