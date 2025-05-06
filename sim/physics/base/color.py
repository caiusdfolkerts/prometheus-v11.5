# prometheus/sim/physics/base/color.py
"""
defines the Abstract Base Class (ABC) for particle color calculation models.
"""

from abc import abstractmethod
from typing import Optional, Dict

# absolute imports from project structure
from .physics import PhysicsModel
from sim.particle_data import ParticleData

class ColorModel(PhysicsModel):
    """ABC for models that compute particle colors for visualization."""

    @abstractmethod
    def compute_colors(self, pd: ParticleData):
        pass