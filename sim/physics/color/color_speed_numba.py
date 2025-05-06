# prometheus/sim/physics/color/color_speed_numba.py
"""
color model based on particle speed magnitude using a linear scale and Numba kernel.
"""

import numpy as np
import traceback
from typing import Dict, Optional

from sim.physics.base.color import ColorModel
from sim.particle_data import ParticleData
from sim.utils import timing_decorator

# --- Numba Import and Kernel Definition ---
try:
    from numba import njit, prange, float64, float32
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    njit = lambda *args, **kwargs: lambda func: func # type: ignore
    prange = range 

@njit(float32[:,:](float64[:], float64, float64), cache=True, fastmath=True, parallel=True)
def map_value_to_color_linear_numba(
    values_f64: np.ndarray, # Input values (speed) - expects float64
    v_min: float,           # Minimum value for normalization range
    v_max: float            # Maximum value for normalization range
    ) -> np.ndarray:
    """(Numba Kernel) Maps values to colors using a linear scale and fixed gradient."""
    N = values_f64.shape[0]
    colors = np.zeros((N, 3), dtype=np.float32)
    # Color gradient points (same as log kernel)
    s_pts = np.array([0.0,  0.25, 0.5,  0.75, 1.0 ], dtype=np.float32)
    c_pts = np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0],
                      [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    num_segments = len(s_pts) - 1
    inv_seg_len = np.zeros(num_segments, dtype=np.float32)
    for i in range(num_segments):
        diff = s_pts[i+1] - s_pts[i]
        inv_seg_len[i] = 1.0 / diff if diff > 1e-9 else 0.0

    # calculate range and inverse range for normalization (use float64)
    val_range_f64 = float64(v_max) - float64(v_min)
    inv_val_range_f64 = 1.0 / val_range_f64 if val_range_f64 > 1e-9 else 0.0

    # ,ap values to colors
    for i in prange(N):
        val_f64 = values_f64[i]
        # handle non-finite values -> map to start color
        if not np.isfinite(val_f64):
            colors[i] = c_pts[0]; continue

        # normalize linearly: (value - min) / (max - min) -> clamp to [0, 1]
        norm_val = (val_f64 - float64(v_min)) * inv_val_range_f64
        norm_val_f32 = np.float32(max(0.0, min(1.0, norm_val))) # Clamp and cast

        # find correct segment and interpolate (same as log kernel)
        seg_idx = 0
        while seg_idx < num_segments and norm_val_f32 > s_pts[seg_idx + 1]: seg_idx += 1
        seg_idx = min(seg_idx, num_segments - 1)
        t = max(0.0, min(1.0, (norm_val_f32 - s_pts[seg_idx]) * inv_seg_len[seg_idx]))
        colors[i] = c_pts[seg_idx] * (1.0 - t) + c_pts[seg_idx + 1] * t

    return colors

# --- python Class ---
class ColorSpeedNumba(ColorModel):
    """Colors particles based on speed magnitude using a linear scale."""

    def setup(self, pd: ParticleData):
        """Sets up speed range parameters."""
        if not HAVE_NUMBA: raise ImportError("Numba is required for ColorSpeedNumba.")
        super().setup(pd)
        self.N = pd.get_n()
        # read config parameters
        try:
            self.speed_percentile = float(self.config['color_speed_percentile'])
            self.min_speed_floor = float(self.config['color_speed_min_floor'])
            if not (0 <= self.speed_percentile <= 100): raise ValueError("Percentile must be 0-100.")
            if self.min_speed_floor < 0: raise ValueError("Min speed floor cannot be negative.")
        except KeyError as e: raise ValueError(f"Missing required config key for ColorSpeedNumba: {e}") from e
        except (TypeError, ValueError) as e: raise ValueError(f"Invalid config value for ColorSpeedNumba: {e}") from e

        print(f"ColorSpeedNumba Setup: N={self.N}, MaxPercentile={self.speed_percentile:.1f}%, MinFloor={self.min_speed_floor:.2f}")

    # @timing_decorator # uncomment for timing
    def compute_colors(self, pd: ParticleData):
        """Computes particle colors based on speed using the Numba kernel."""
        target_dtype = np.float32 # colors are f32
        if self.N == 0: pd.set("colors", np.empty((0, 3), dtype=target_dtype)); return

        try:
            pd.ensure("velocities", "cpu")
            vel_orig = pd.get("velocities", "cpu")

            # calculate speeds (use float64 for precision)
            speeds_f64 = np.linalg.norm(vel_orig, axis=1).astype(np.float64)

            # determine normalization range [vmin, vmax] using float64
            vmin = float64(self.min_speed_floor)
            vmax = float64(self.min_speed_floor + 1e-6) # default max if no valid speeds
            valid_speeds = speeds_f64[np.isfinite(speeds_f64) & (speeds_f64 >= 0)] # filter finite, non-negative

            if valid_speeds.size > 0:
                 # use actual data min if > floor, otherwise use floor
                 vmin = max(self.min_speed_floor, np.min(valid_speeds))
                 vmax_perc = np.percentile(valid_speeds, self.speed_percentile)
                 vmax = max(vmin + 1e-6, vmax_perc) 

            # Call the linear mapping kernel (expects f64 inputs/range)
            colors_f32 = map_value_to_color_linear_numba(speeds_f64, vmin, vmax)
            pd.set("colors", colors_f32.astype(target_dtype, copy=False), source_device="cpu")

        except Exception as e:
            print(f"ERROR during Numba speed color kernel: {e}")
            traceback.print_exc()
            default_colors = np.full((self.N, 3), 0.5, dtype=target_dtype)
            pd.set("colors", default_colors, source_device="cpu")