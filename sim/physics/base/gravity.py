# prometheus/sim/physics/base/gravity.py
"""
defines the ABC for gravitational interaction models
"""

from abc import abstractmethod
from .physics import PhysicsModel
from sim.particle_data import ParticleData

class GravityModel(PhysicsModel):
    """ABC for gravitational force and potential energy calculation models."""

    @abstractmethod
    def compute_forces(self, pd: ParticleData):

        pass

    @abstractmethod
    def compute_potential_energy(self, pd: ParticleData) -> float:
        pass

    def setup(self, pd: ParticleData):
        super().setup(pd) # calls PhysicsModel.setup()
        if 'G' not in self.config:
             print(f"Warning ({self.__class__.__name__}): Gravitational constant 'G' not found in config.")