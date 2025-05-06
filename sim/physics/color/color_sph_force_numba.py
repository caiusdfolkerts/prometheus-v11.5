# prometheus/sim/physics/color/color_sph_force_numba.py
"""
Color model based on SPH force magnitude using a logarithmic scale and Numba kernel.
SPH force is determined as F_total - F_gravity.
"""

import numpy as np
import traceback
from typing import Dict, Optional

# Absolute imports from project structure
from sim.physics.base.color import ColorModel
from sim.particle_data import ParticleData
from sim.utils import timing_decorator

# --- Numba Import and Kernel Import ---
try:
    # Import the shared log-scale color kernel from color_temp_numba
    from .color_temp_numba import calculate_colors_log_scale_numba_kernel as map_log_value_to_color
    HAVE_NUMBA = True # Assumes color_temp_numba also checked this
except ImportError:
    HAVE_NUMBA = False
    # define dummy decorator if Numba is not available for map_log_value_to_color
    # this path shouldn't be taken if color_temp_numba exists and works.
    if 'map_log_value_to_color' not in globals(): # check if it was successfully imported
        njit = lambda *args, **kwargs: lambda func: func
        #  if import failed, define default to avoid crash
        def map_log_value_to_color(*args, **kwargs):
             print("Warning: Dummy map_log_value_to_color called. Numba or color_temp_numba.py might be missing.")
             if len(args) > 0 and hasattr(args[0], 'shape'):
                 N_kernel = args[0].shape[0]
                 return np.full((N_kernel, 3), 0.5, dtype=np.float32) # Default gray as with other modules
             return np.empty((0,3), dtype=np.float32)


# --- Python Class ---
class ColorSPHForceNumba(ColorModel):
    """Colors particles based on SPH force magnitude using a dynamic log scale."""

    def setup(self, pd: ParticleData):
        """Sets up an absolute minimum force floor for coloring."""
        if not HAVE_NUMBA: raise ImportError("Numba is required for ColorSPHForceNumba.")
        super().setup(pd)
        self.N = pd.get_n()
        # define an absolute minimum force magnitude for dynamic range calculation
        # this prevents issues if all forces are zero or extremely small.
        self.min_force_floor = float(self.config.get('color_sph_force_min_floor', 1e-9))
        print(f"ColorSPHForceNumba Setup: N={self.N}, MinForceFloor={self.min_force_floor:.2e}")

    @timing_decorator
    def compute_colors(self, pd: ParticleData):
        """Computes particle colors based on SPH force magnitude."""
        target_dtype = np.float32 # colors are typically f32
        if self.N == 0:
            pd.set("colors", np.empty((0, 3), dtype=target_dtype)); return

        try:
            # ensure total forces and gravitational forces are on CPU
            pd.ensure("forces forces_grav", "cpu")
            forces_total_orig = pd.get("forces", "cpu")
            forces_grav_orig = pd.get("forces_grav", "cpu")

            # calculate SPH force vector: F_sph = F_total - F_gravity
            # ensure consistent dtypes for subtraction, use float64 for intermediate calc
            forces_total_f64 = forces_total_orig.astype(np.float64, copy=False)
            forces_grav_f64 = forces_grav_orig.astype(np.float64, copy=False)
            sph_force_vector_f64 = forces_total_f64 - forces_grav_f64

            # Calculate magnitude of the SPH force
            sph_force_magnitudes_f64 = np.linalg.norm(sph_force_vector_f64, axis=1)

            # Call the shared log kernel for color mapping
            colors_f32 = map_log_value_to_color(
                sph_force_magnitudes_f64, # Values to map (expects float64)
                self.min_force_floor,     # Absolute minimum floor for values
                0.0,                      # log_vmin_fixed = 0.0 (dynamic range)
                0.0                       # log_vmax_fixed = 0.0 (dynamic range)
            )
            pd.set("colors", colors_f32.astype(target_dtype, copy=False), source_device="cpu")

        except Exception as e:
            print(f"ERROR during Numba SPH force color kernel: {e}")
            traceback.print_exc()
            default_colors = np.full((self.N, 3), 0.5, dtype=target_dtype)
            pd.set("colors", default_colors, source_device="cpu")