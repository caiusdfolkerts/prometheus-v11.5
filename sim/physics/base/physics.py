# prometheus/sim/physics/base/physics.py
"""
Defines the Abstract Base Class (ABC) for all physics models.

Provides a consistent interface for different physics components like gravity, SPH,
thermodynamics, etc., ensuring they can be managed and executed uniformly by the
PhysicsManager.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

# Absolute import from project structure
from sim.particle_data import ParticleData

class PhysicsModel(ABC):
    """
    Abstract Base Class for physics calculation components.

    All specific physics models (e.g., GravityBHNumba, SPHTaichi) should inherit
    from this class and implement the abstract `setup` method. They should also
    implement relevant compute methods (e.g., `compute_forces`, `compute_density`).
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the physics model with configuration.

        Args:
            config: Simulation-wide configuration dictionary. Stored internally.
        """
        # Store a copy or an empty dict if config is None
        self.config: Dict = config.copy() if config is not None else {}
        self._is_setup: bool = False # Flag to track if setup has been called

    @abstractmethod
    def setup(self, pd: ParticleData):
        """
        Perform model-specific setup and precomputation.

        Called when the model is selected or during simulation restarts/resets.
        Implementations should use this method to:
        - Validate required configuration parameters in `self.config`.
        - Retrieve necessary initial data from ParticleData (`pd`).
        - Precompute constants or allocate model-specific resources (if any
          are not managed directly within ParticleData).
        - Set the `_is_setup` flag to True upon successful completion.

        Args:
            pd: The particle data manager.
        """
        # Base implementation just sets the flag; subclasses MUST call super().setup(pd)
        # or set self._is_setup = True themselves after successful setup.
        self._is_setup = True
        pass

    def is_ready(self) -> bool:
        """Checks if the model's setup method has been successfully completed."""
        return self._is_setup

    def update_config(self, config: Dict):
         """
         Updates the model's internal configuration dictionary.

         Called by the PhysicsManager when live parameters or settings change.
         Subclasses should override this method if they need to perform specific
         actions (e.g., recalculate constants) in response to config updates,
         after calling `super().update_config(config)`.

         Args:
            config: Dictionary containing updated configuration parameters.
         """
         if config: # Update internal config if provided
              self.config.update(config)
         # Base implementation only updates the dictionary.

    def cleanup(self):
         """
         Optional method to release resources specific to this model instance.

         Called by the PhysicsManager when the model is deselected or during
         shutdown. Useful for releasing GPU memory allocated outside of
         ParticleData, closing file handles, etc.
         """
         # print(f"Cleaning up {self.__class__.__name__}...") # Optional log
         pass # Base implementation does nothing

    # __del__ is generally avoided for complex cleanup. Explicit cleanup() is preferred.