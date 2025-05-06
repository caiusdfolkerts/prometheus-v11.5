# prometheus/sim/constants.py
"""
Physical constants used in the simulation.
"""
import numpy as np

# --- Fundamental Constants (CGS) ---
CONST_k_B   = 1.380649e-16  # boltzmann constant (erg/K)
CONST_m_p   = 1.6726219e-24 # proton mass (g)
CONST_a_rad = 7.5657e-15    # radiation constant (erg cm^-3 K^-4)
CONST_pi    = np.pi         # pi

# Note that some other constants such as G are defined in default simulation units