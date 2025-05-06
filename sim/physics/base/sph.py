# prometheus/sim/physics/base/sph.py
"""
Defines the Abstract Base Class (ABC) for Smoothed Particle Hydrodynamics (SPH) models.
"""

from abc import abstractmethod
from .physics import PhysicsModel
from sim.particle_data import ParticleData

class SPHModel(PhysicsModel):
    """ABC for SPH calculation models (density, forces, energy terms)."""

    @abstractmethod
    def compute_density(self, pd: ParticleData):
        pass

    @abstractmethod
    def compute_pressure_force(self, pd: ParticleData):
        pass

    def setup(self, pd: ParticleData):
        super().setup(pd) # calls PhysicsModel.setup()
        if 'h' not in self.config or not isinstance(self.config['h'], (float, int)) or self.config['h'] <= 0:
             raise ValueError(f"SPHModel ({self.__class__.__name__}) requires a positive smoothing length 'h' in configuration.")