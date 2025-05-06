# prometheus/sim/physics/base/thermo.py
"""
Defines the ABC for thermodynamics models.
"""

from abc import abstractmethod

# Absolute imports from project structure
from .physics import PhysicsModel
from sim.particle_data import ParticleData

class ThermodynamicsModel(PhysicsModel):
    @abstractmethod
    def update_eos_quantities(self, pd: ParticleData):
        pass

    @abstractmethod
    def compute_energy_changes(self, pd: ParticleData):
        pass

    def setup(self, pd: ParticleData):
        super().setup(pd) # calls PhysicsModel.setup()
