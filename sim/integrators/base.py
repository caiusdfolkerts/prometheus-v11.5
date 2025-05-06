# prometheus/sim/integrators/base.py
"""
Defines the Abstract Base Class (ABC) for all time integrators.
"""
# ... (imports remain the same) ...
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict
from sim.particle_data import ParticleData

class Integrator(ABC):
    """abstract base class for time integration schemes."""

    def setup(self, pd: ParticleData, config: Optional[Dict] = None):
        """optional setup method."""
        pass

    @abstractmethod
    def step(self, pd: ParticleData, dt: float, forces_cb: Callable[[], None],
             config: Optional[Dict] = None, current_step: int = -1, debug_particle_idx: int = -1):
        """
        one integration step
        """
        pass

    def cleanup(self):
        pass