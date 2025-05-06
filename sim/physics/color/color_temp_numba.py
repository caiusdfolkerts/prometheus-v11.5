# prometheus/sim/physics/color/color_temp_numba.py
"""
Color model based on particle temperature using a logarithmic scale and Numba kernel.
"""

import numpy as np
import traceback
from math import log10
from typing import Dict, Optional

# Absolute imports from project structure
from sim.physics.base.color import ColorModel
from sim.particle_data import ParticleData
from sim.utils import timing_decorator

# --- Numba Import and Kernel Definition ---
try:
    from numba import njit, prange, float64, float32
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    # Define dummy decorator if Numba is not available to aviod crash
    njit = lambda *args, **kwargs: lambda func: func

# This kernel is also used by ColorDensityNumba
@njit(float32[:,:](float64[:], float64, float64, float64), cache=True, fastmath=True, parallel=True)
def calculate_colors_log_scale_numba_kernel(
    values_f64: np.ndarray,       # input values (temperature or density)
    min_value_floor: float,       # absolute minimum floor for the physical value
    log_vmin_fixed: float,        # pre-calculated log10(fixed_min) or 0.0 if dynamic
    log_vmax_fixed: float         # pre-calculated log10(fixed_max) or 0.0 if dynamic
    ) -> np.ndarray:
    """(Numba Kernel) Maps values to colors using a log scale and fixed gradient."""
    N = values_f64.shape[0]
    colors = np.zeros((N, 3), dtype=np.float32)
    # color gradient points (normalized scale s, color c)
    s_pts = np.array([0.0,  0.25, 0.5,  0.75, 1.0 ], dtype=np.float32)
    c_pts = np.array([ # Dark Blue -> Red -> Orange -> Yellow -> White
                       [0.0, 0.0, 0.5], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0],
                       [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]
                     ], dtype=np.float32)
    num_segments = len(s_pts) - 1
    # precompute inverse segment lengths for interpolation
    inv_seg_len = np.zeros(num_segments, dtype=np.float32)
    for i in range(num_segments):
        diff = s_pts[i+1] - s_pts[i]
        inv_seg_len[i] = 1.0 / diff if diff > 1e-9 else 0.0

    # determine mapping range (fixed or dynamic)
    log_min = log_vmin_fixed
    log_max = log_vmax_fixed
    use_fixed = (log_vmax_fixed > log_vmin_fixed + 1e-9)

    if not use_fixed: # Calculate dynamic range if fixed range wasn't valid
        finite_mask = np.isfinite(values_f64)
        valid_values = values_f64[finite_mask & (values_f64 > min_value_floor)]
        if valid_values.size > 0:
            v_min_data = np.min(valid_values); v_max_data = np.max(valid_values)
            log_min = log10(max(1e-15, v_min_data))
            log_max = log10(max(1e-15, v_max_data))
        else: # Fallback if no valid data > floor
            log_min = log10(max(1e-15, min_value_floor)); log_max = log_min + 4.0 

    log_range = log_max - log_min
    inv_log_range = 1.0 / log_range if log_range > 1e-9 else 0.0

    # Map values to colors
    for i in prange(N):
        val = values_f64[i]
        # Handle non-finite values or values below floor -> map to start color
        if not np.isfinite(val) or val <= min_value_floor:
            colors[i] = c_pts[0]; continue

        # Normalize logarithm of value to [0, 1] range
        log_val = log10(max(1e-15, val)) # Avoid log10(0)
        norm_val = np.float32(max(0.0, min(1.0, (log_val - log_min) * inv_log_range)))

        # Find correct segment in color gradient
        seg_idx = 0
        while seg_idx < num_segments and norm_val > s_pts[seg_idx + 1]:
            seg_idx += 1
        # Clamp index just in case
        seg_idx = min(seg_idx, num_segments - 1)

        # Interpolate within the segment
        t = (norm_val - s_pts[seg_idx]) * inv_seg_len[seg_idx]
        t = max(0.0, min(1.0, t)) # Clamp interpolation factor
        colors[i] = c_pts[seg_idx] * (1.0 - t) + c_pts[seg_idx + 1] * t

    return colors

# --- Python Class ---
class ColorTempNumba(ColorModel):
    """Colors particles based on temperature using a logarithmic scale."""

    def setup(self, pd: ParticleData):
        """Sets up temperature ranges for color mapping."""
        if not HAVE_NUMBA: raise ImportError("Numba is required for ColorTempNumba.")
        super().setup(pd)
        self.N = pd.get_n()
        # Read fixed color range parameters from config
        try:
            self.color_temp_min = float(self.config['color_temp_min'])
            self.color_temp_max = float(self.config['color_temp_max'])
            self.min_temp_floor = float(self.config['min_temperature']) # Absolute floor
        except KeyError as e: raise ValueError(f"Missing required config key for ColorTempNumba: {e}") from e
        except (TypeError, ValueError) as e: raise ValueError(f"Invalid config value for ColorTempNumba: {e}") from e

        # Pre-calculate log range (use float64)
        self.log_t_min = log10(max(1e-15, self.color_temp_min))
        self.log_t_max = log10(max(1e-15, self.color_temp_max))
        if self.log_t_max <= self.log_t_min: # Ensure max > min
             print(f"Warn: color_temp_max <= min. Adjusting max for 4 orders of magnitude.")
             self.log_t_max = self.log_t_min + 4.0

        print(f"ColorTempNumba Setup: N={self.N}, Temp Range=[{self.color_temp_min:.1f} K, {self.color_temp_max:.1f} K]")

    def compute_colors(self, pd: ParticleData):
        """Computes particle colors based on temperature using the Numba kernel."""
        if self.N == 0: pd.set("colors", np.empty((0, 3), dtype=np.float32)); return

        try:
            pd.ensure("temperatures", "cpu")
            temps_orig = pd.get("temperatures", "cpu")
            temps_f64 = temps_orig.astype(np.float64, copy=False)

            # Call the kernel with pre-calculated fixed log range
            colors_f32 = calculate_colors_log_scale_numba_kernel(
                temps_f64,
                self.min_temp_floor, # Absolute floor for clipping low values
                self.log_t_min,      # Fixed log min for normalization range
                self.log_t_max       # Fixed log max for normalization range
            )
            pd.set("colors", colors_f32, source_device="cpu") # Set f32 colors

        except Exception as e:
            print(f"ERROR during Numba temperature color kernel: {e}")
            traceback.print_exc()
            # give default gray colors on error
            default_colors = np.full((self.N, 3), 0.5, dtype=np.float32)
            pd.set("colors", default_colors, source_device="cpu")
