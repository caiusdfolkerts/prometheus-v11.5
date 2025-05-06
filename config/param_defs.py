# prometheus/config/param_defs.py
"""
UI Parameter Definitions (for sliders, toggles etc.) in the frontend.
Matches the structure used in the original setup_ui_controls,
using standard Python types.
"""

PARAM_DEFS = {
      # --- Basic Physics & Initial Conditions ---
      'N':         {'label':'Particles (N)', 'min':10,'max':1000000,'step':100,'val':10000, 'fmt':"{:d}", 'live':False},
      'radius':    {'label':'Init Radius',   'min':1.0,'max':50.0,  'step':0.5,'val':10.0,  'fmt':"{:.1f}", 'live':False},
      'v_scale':   {'label':'Init Vel Scale','min':0.0,'max':20.0,  'step':0.1,'val':5.0,   'fmt':"{:.1f}", 'live':False},
      'rotation_strength': {'label':'Init Rotation Str.', 'min':0.0, 'max':10.0, 'step':0.1, 'val':0.0, 'fmt':"{:.2f}", 'live':False}, # Included with val:0.0
      'mass_min':  {'label':'Mass Min',      'min':0.1,'max':5.0,   'step':0.1,'val':0.8,   'fmt':"{:.1f}", 'live':False},
      'mass_max':  {'label':'Mass Max',      'min':0.1,'max':10.0,  'step':0.1,'val':1.2,   'fmt':"{:.1f}", 'live':False},
      'initial_temp':{'label':'Initial Temp (K)','min':1.0, 'max':100000.0,'step':100.0,'val':2000.0,  'fmt':"{:.1f}", 'live':False},

      # --- Gravity Parameters ---
      'G':         {'label':'Gravity (G)',   'min':0.0,'max':5000.0,'step':1.0,  'val':60.0, 'fmt':"{:.1f}", 'live':True},
      'softening': {'label':'Softening ε',   'min':0.01,'max':5.0,  'step':0.01,'val':1.0,   'fmt':"{:.3f}", 'live':True},
      'bh_theta':  {'label':'BH Theta θ',    'min':0.0,'max':2.0,   'step':0.05,'val':0.5,   'fmt':"{:.2f}", 'live':True},
      'MAX_NODES_FACTOR': {'label':'BH Node Factor','min':2,'max':50,'step':1,'val':10,    'fmt':"{:d}", 'live':False},

      # --- SPH Parameters ---
      'h':         {'label':'SPH Radius (h)','min':0.1,'max':10.0,  'step':0.05,'val':2.0,   'fmt':"{:.2f}", 'live':True},
      'sph_visc_alpha': {'label':'SPH Visc α', 'min':0.0, 'max':2.0, 'step':0.05, 'val':0.0, 'fmt':"{:.2f}", 'live':True},
      'sph_visc_beta':  {'label':'SPH Visc β', 'min':0.0, 'max':2.0, 'step':0.05, 'val':0.0, 'fmt':"{:.2f}", 'live':True},

      # --- Thermodynamics Parameters ---
      'mu':        {'label':'Mean Mol Wt μ', 'min':0.5, 'max':2.5,   'step':0.05,'val':0.6,   'fmt':"{:.2f}", 'live':False}, # Not live as it affects Cv calculation, requiring restart
      'min_temp':  {'label':'Min Temp Floor (K)','min':1.0, 'max':1000.0,'step':1.0,'val':10.0,   'fmt':"{:.1f}", 'live':True},
      'cooling_coeff':{'label':'Cool Strength','min':0.0,'max':0.1,   'step':1e-5,'val':1e-3,  'fmt':"{:.2e}", 'live':True},
      'cooling_beta':{'label':'Cool Exp β',    'min':-2.0,'max':2.0,   'step':0.1, 'val':0.5,   'fmt':"{:.2f}", 'live':True},
      'fusion_thresh':{'label':'Fusion Thresh (K)','min':1e2,'max':1e8,'step':1e2,'val':1e4, 'fmt':"{:.1e}", 'live':True},
      'fusion_coeff':{'label':'Fusion Strength','min':0.0,'max':1.0,   'step':1e-26,'val':1e-24,'fmt':"{:.2e}", 'live':True},
      'fusion_power':{'label':'Fusion Exp p',  'min':0.0,'max':15.0,  'step':0.5,'val':5.0,   'fmt':"{:.1f}", 'live':True},
      'FUSION_MAX_SPECIFIC_ENERGY_INCREASE_PER_DT': {'label':'Fusion Max Δu/step', 'min':0.001, 'max':1.0, 'step':0.001, 'val':0.1, 'fmt':"{:.3f}", 'live':True},


      # --- Numerical / Simulation Control ---
      'dt':          {'label':'Timestep (dt)', 'min':1e-7,'max':0.01,  'step':1e-6,'val':5e-5,  'fmt':"{:.1e}", 'live':False},

      # --- UI / Visualization Color Scale Parameters ---
      'color_temp_min': {'label':'Color T Min (K)', 'min':1.0, 'max':1e5, 'step':1.0, 'val':100.0, 'fmt':"{:.1f}", 'live':False},
      'color_temp_max': {'label':'Color T Max (K)', 'min':10.0,'max':1e7, 'step':10.0,'val':1e6,'fmt':"{:.1f}", 'live':False},
      'color_speed_percentile': {'label':'Color Spd %Max', 'min':80.0, 'max':100.0, 'step':0.1, 'val':99.5, 'fmt':"{:.1f}", 'live':False},
      'color_speed_min_floor': {'label':'Color Spd Min Floor', 'min':0.0, 'max':100.0, 'step':0.1, 'val':0.0, 'fmt':"{:.1f}", 'live':False},
}

# --- VALIDATION ---
if len(PARAM_DEFS.keys()) != len(set(PARAM_DEFS.keys())):
    raise ValueError("Duplicate keys found in PARAM_DEFS!")
