# prometheus/config/available_models.py
"""
Defines the physics models available for selection.
The structure allows dynamic loading by the PhysicsManager.
"""

# a dictionary where keys are model types (gravity, sph, thermo and color)
# and values are lists of available implementations for that type.
AVAILABLE_MODELS = {
    "gravity": [
        {
            "id": "gravity_bh_numba",
            "name": "Barnes-Hut (Numba)",
            "description": "Optimized N-body gravity using Barnes-Hut algorithm accelerated with Numba (CPU).",
            "module": "sim.physics.gravity.gravity_bh_numba",
            "class": "GravityBHNumba",
            "required_backend": "numpy",
            "notes": "Good performance for large N on CPU.",
            "_backend_available": True
        },
        {
            "id": "gravity_pp_cpu",
            "name": "Direct PP (NumPy)",
            "description": "Direct particle-particle N^2 gravity calculation using pure NumPy (CPU).",
            "module": "sim.physics.gravity.gravity_pp_cpu",
            "class": "GravityPPCpu",
            "required_backend": "numpy",
            "notes": "Simple, but very slow for large N.",
            "_backend_available": True
        },
        {
             "id": "gravity_pp_gpu",
             "name": "Direct PP (Taichi GPU)",
             "description": "Direct particle-particle N^2 gravity using Taichi kernel on GPU.",
             "module": "sim.physics.gravity.gravity_pp_gpu",
             "class": "GravityPPGpu",
             "required_backend": "gpu:ti",
             "notes": "Fast for moderate N on GPU, scales as O(N^2)."
        },
    ],
    "sph": [
        {
            "id": "sph_taichi",
            "name": "SPH (Taichi Grid)",
            "description": "Standard SPH calculations using Taichi's built-in grid neighbor search and Taichi kernels (GPU/CPU).",
            "module": "sim.physics.sph.sph_taichi",
            "class": "SPHTaichi",
            "required_backend": "gpu:ti",
            "notes": "Potentially faster on GPU, uses Taichi native neighbor search.",
        },
    ],
    "thermo": [
        {
            "id": "thermo_numba",
            "name": "Thermo (Numba)",
            "description": "Thermodynamics (EOS, cooling, fusion) using Numba-accelerated kernels (CPU).",
            "module": "sim.physics.thermo.thermo_numba",
            "class": "ThermoNumba",
            "required_backend": "numpy",
            "notes": "Standard CPU implementation.",
            "_backend_available": True
        },
    ],
     "color": [
        {
            "id": "color_temp_numba",
            "name": "Color (Temperature)",
            "description": "Calculates particle colors based on temperature using Numba (CPU). Log scale. Outputs float32.",
            "module": "sim.physics.color.color_temp_numba",
            "class": "ColorTempNumba",
            "required_backend": "numpy",
            "notes": "Log scale color mapping based on temperature.",
            "_backend_available": True
        },
        {
            "id": "color_speed_numba",
            "name": "Color (Speed)",
            "description": "Calculates particle colors based on speed magnitude using Numba (CPU). Linear scale. Outputs float32.",
            "module": "sim.physics.color.color_speed_numba",
            "class": "ColorSpeedNumba",
            "required_backend": "numpy",
            "notes": "Linear color mapping using speed percentiles.",
            "_backend_available": True
        },
        {
             "id": "color_sph_force_numba",
             "name": "Color (SPH Work)",
             "description": "Calculates particle colors based on SPH work term magnitude using Numba (CPU). Log scale. Outputs float32.",
             "module": "sim.physics.color.color_sph_force_numba",
             "class": "ColorSPHForceNumba",
             "required_backend": "numpy",
             "notes": "Log scale color mapping based on absolute SPH force  magnitude.",
             "_backend_available": True
         }
    ]
}

# --- validate the models ---
def _validate_models():
    for model_type, model_list in AVAILABLE_MODELS.items():
        if not isinstance(model_list, list):
            raise TypeError(f"AVAILABLE_MODELS entry for '{model_type}' must be a list.")
        if not model_list and model_type in ["gravity", "sph", "thermo"]: 
             print(f"WARNING: No models defined for essential type '{model_type}' in AVAILABLE_MODELS.")
        for model_def in model_list:
            required_keys = ["id", "name", "module", "class", "required_backend"]
            if not all(key in model_def for key in required_keys):
                raise ValueError(f"Model definition in '{model_type}' is missing required keys: {model_def}")
_validate_models()