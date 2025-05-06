# prometheus/config/available_integrators.py
"""
Defines a list of all integrators available for selection.
"""

AVAILABLE_INTEGRATORS = [
    {
        "id": "leapfrog",          
        "name": "Leapfrog (KDK)",   
        "description": "Standard second-order Kick-Drift-Kick Leapfrog integrator.",
        "module": "sim.integrators.leapfrog",
        "class": "Leapfrog",        
        "order": 2,
        "notes": "Requires 1 force evaluation per step. Energy drift depends on dt^2."
    },
    {
        "id": "yoshida4",
        "name": "Yoshida 4th-Order",
        "description": "Fourth-order symplectic integrator by Yoshida (1990).",
        "module": "sim.integrators.yoshida4",
        "class": "Yoshida4",
        "order": 4,
        "notes": "Requires 3 force evaluations per step. More accurate for larger dt. Energy drift depends on dt^4."
    },
]

# --- integrator validation ---
def _validate_integrators():
    for integrator_def in AVAILABLE_INTEGRATORS:
        required_keys = ["id", "name", "module", "class", "order"]
        if not all(key in integrator_def for key in required_keys):
            raise ValueError(f"Integrator definition is missing required keys: {integrator_def}")
_validate_integrators()
