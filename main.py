# prometheus/main.py
# ==================================================
#      <<< SERVER IMPLEMENTATION >>>
# ==================================================
"""
main.py acts as the main application entry point for the simulation server.

This script acts as a central administrator for the simulation, taking
an input from the frontend and delegating relevant tasks to the relevant 
parts. It performs the following key functions:

1. Initialization Sets up Python paths, determines effective compute precision
    (based on defaults and hardware capabilities), initializes required libraries
    (especially Taichi, attempting various configurations), and checks for optional
    libraries (PyTorch, CuPy). Updates default settings based on effective precision.
2.  Threading Setup: Establishes the core threading model:
    - Main Thread: Handles initial setup, runs the task processor loop.
    - Server Thread (daemon): Runs the Flask/SocketIO web server.
    - Simulation Worker Thread (daemon): Executes the main simulation loop.
    - Synchronization Primitives: Uses `threading.Lock` (sim_lock) for safe access
      to the shared Simulator instance and `threading.Event` (stop_simulation_flag)
      for controlling the worker thread.
3.  Main Thread Task Queue: Creates a ;queue.Queue' (main_thread_task_queue)
    to delegate tasks that *must* run on the main thread (like Taichi field
    allocations during restarts) from other threads.
4.  Flask & SocketIO Server: Configures and runs a Flask web server with
    Flask-SocketIO for handling HTTP API requests ('/api/...') and WebSocket
    communication ('connect', 'disconnect', 'send_command', etc.) with the
    frontend UI. Serves static UI files.
5.  Backend Helpers: Provides utility functions for serializing data for JSON
    responses (`_to_serializable`), fetching initial configuration (`get_initial_config`),
    and getting the current simulation state (`get_simulation_state`).
6.  **Main Thread Task Execution:** Implements `_execute_main_thread_task` which
    processes tasks from the queue, specifically handling `ParticleData` allocation
    and pre-initialization of Taichi-based physics models during restarts.
7.  **Command Handling:** The `handle_command` function processes commands received
    from the UI (via HTTP or WebSocket) to control the simulation (toggle run, step,
    reset, restart, set parameters/flags, select models/integrators), interacting
    with the `Simulator` instance under lock protection.
8.  Simulation Loop: The 'simulation_loop_worker' function runs in its own
    thread, continuously advancing the simulation (if running) by calling
    'sim.advance_one_step()', collecting graph data,
    and emitting periodic state updates to the UI via SocketIO.
9.  PDF Generation: Provides an API endpoint ('/api/generate_pdf') to trigger
    the generation of summary plots based on collected simulation data.
10. Application Orchestration: The 'if __name__ == "__main__":' block
    coordinates the entire startup sequence: calls `main_backend_setup` to
    initialize the 'Simulator' instance (passing the task queue reference), starts
    the web server thread, and then runs the 'main_thread_task_processor' loop
    in the main thread. It also handles graceful shutdown on Ctrl+C.
    
"""
import time
import traceback
import threading
import numpy as np
import sys
import os
import queue

# --- For plotting graphs
import matplotlib
matplotlib.use('Agg')
from sim.plotting import generate_plots_pdf

# --- Project Setup ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Simulation Core Components ---
from sim.particle_data import ParticleData
from sim.simulator import Simulator, SIM_VERSION
from sim.utils import dynamic_import #

# --- Configuration ---
from config.param_defs import PARAM_DEFS
from config.default_settings import DEFAULT_SETTINGS as initial_default_settings
from config.available_models import AVAILABLE_MODELS
from config.available_integrators import AVAILABLE_INTEGRATORS

# --- Server Components ---
from flask import Flask, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit

# --- Output Directory ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Reduced print: print(f"Output directory for PDFs: {os.path.normpath(OUTPUT_DIR)}")

# ============================================
# --- Library Initialization and Detection ---
# ============================================
print("\n--- Initializing Libraries ---")

# --- User Preferences ---
USER_REQUESTED_DOUBLE_PRECISION = initial_default_settings.get('USE_DOUBLE_PRECISION', True)
DEFAULT_TAICHI_ARCH = initial_default_settings.get('default_taichi_arch', "gpu")

# --- State Flags ---
TAICHI_INITIALIZED = False
TAICHI_BACKEND_SUPPORTS_F64 = False
TAICHI_EFFECTIVE_FP_TYPE = None
TAICHI_EFFECTIVE_IP_TYPE = None
TAICHI_ACTUAL_ARCH = None
CUPY_AVAILABLE = False

# --- Taichi Initialization ---
try:
    import taichi as ti
    print("Taichi: Attempting Initialization...")
    preferred_ti_fp = ti.f64 if USER_REQUESTED_DOUBLE_PRECISION else ti.f32
    preferred_ti_ip = ti.i32

    try:
        preferred_arch_enum = getattr(ti, DEFAULT_TAICHI_ARCH)
    except AttributeError:
        print(f"  Warning: Invalid DEFAULT_TAICHI_ARCH '{DEFAULT_TAICHI_ARCH}'. Falling back to cpu.")
        DEFAULT_TAICHI_ARCH = "cpu" 
        preferred_arch_enum = ti.cpu

    init_kwargs = {
        "default_ip": preferred_ti_ip,
        "device_memory_GB": initial_default_settings.get('taichi_device_memory_gb', 4.0)
    }
    init_attempts = [
        {"arch": preferred_arch_enum, "default_fp": preferred_ti_fp},
        {"arch": preferred_arch_enum, "default_fp": ti.f32} if preferred_ti_fp == ti.f64 else None,
        {"arch": ti.cpu, "default_fp": preferred_ti_fp} if preferred_arch_enum != ti.cpu else None,
        {"arch": ti.cpu, "default_fp": ti.f32} if preferred_arch_enum != ti.cpu else None,
    ]

    for attempt_cfg in init_attempts:
        if TAICHI_INITIALIZED: break
        if attempt_cfg is None: continue
        arch_name_str = next(k for k, v in ti.__dict__.items() if v == attempt_cfg['arch'])
        fp_name_str = 'f64' if attempt_cfg['default_fp'] == ti.f64 else 'f32'
        try:
            ti.init(**{**init_kwargs, **attempt_cfg})
            TAICHI_INITIALIZED = True
        except Exception: 
            pass

    if TAICHI_INITIALIZED:
        cfg = ti.lang.impl.current_cfg()
        TAICHI_ACTUAL_ARCH = cfg.arch
        TAICHI_EFFECTIVE_FP_TYPE = cfg.default_fp
        TAICHI_EFFECTIVE_IP_TYPE = cfg.default_ip
        actual_arch_name_str = next((k for k, v in ti.__dict__.items() if v == TAICHI_ACTUAL_ARCH), 'Unknown')
        print(f"  Taichi: OK (Arch={actual_arch_name_str}, FP={TAICHI_EFFECTIVE_FP_TYPE})")  
        if TAICHI_ACTUAL_ARCH in [ti.metal, ti.opengl] and TAICHI_EFFECTIVE_FP_TYPE != ti.f64:
            TAICHI_BACKEND_SUPPORTS_F64 = False
        else:
            TAICHI_BACKEND_SUPPORTS_F64 = (TAICHI_EFFECTIVE_FP_TYPE == ti.f64)
    else:
        print("  Taichi: Initialization FAILED.")

except ImportError:
    print("Taichi: Library not found. Features disabled.")
except Exception as e_global_ti:
    print(f"Taichi: ERROR during import or initialization: {e_global_ti}")
    traceback.print_exc()

# --- Determine Effective NumPy Precision ---
effective_precision_is_f64 = False
if TAICHI_INITIALIZED:
    if USER_REQUESTED_DOUBLE_PRECISION and TAICHI_BACKEND_SUPPORTS_F64:
        effective_precision_is_f64 = True
    elif USER_REQUESTED_DOUBLE_PRECISION and not TAICHI_BACKEND_SUPPORTS_F64:
        effective_precision_is_f64 = False
        print("  WARNING: User requested f64, but Taichi backend lacks support. Using f32.")
    else:
        effective_precision_is_f64 = False
else:
    effective_precision_is_f64 = USER_REQUESTED_DOUBLE_PRECISION
effective_np_float_type = np.float64 if effective_precision_is_f64 else np.float32


# --- Update Default Settings ---
from config.default_settings import DEFAULT_SETTINGS
DEFAULT_SETTINGS['USE_DOUBLE_PRECISION'] = effective_precision_is_f64
DEFAULT_SETTINGS['_effective_np_float_type'] = effective_np_float_type
DEFAULT_SETTINGS['_effective_ti_float_type'] = TAICHI_EFFECTIVE_FP_TYPE
DEFAULT_SETTINGS['_effective_ti_int_type'] = TAICHI_EFFECTIVE_IP_TYPE

# --- Taichi Kernel Pre-compilation for Apply and restart---
if TAICHI_INITIALIZED:
    try:
        _np_dtype = DEFAULT_SETTINGS['_effective_np_float_type']
        _ti_dtype = TAICHI_EFFECTIVE_FP_TYPE
        if _ti_dtype is None:
             print("  Skipping pre-compilation: Effective Taichi type unknown.")
        else:
            dummy_scalar_ti = ti.field(dtype=_ti_dtype, shape=1)
            dummy_vector_ti = ti.Vector.field(3, dtype=_ti_dtype, shape=1)
            dummy_scalar_np = np.zeros(1, dtype=_np_dtype)
            dummy_vector_np = np.zeros((1, 3), dtype=_np_dtype)
            dummy_scalar_ti.from_numpy(dummy_scalar_np)
            dummy_vector_ti.from_numpy(dummy_vector_np)
            @ti.kernel
            def _dummy_kernel_precompile(f_s: ti.template(), f_v: ti.template()):
                one_val: _ti_dtype = 1.0
                f_s[0] = one_val
                f_v[0] = ti.Vector([1.0, 1.0, 1.0], dt=_ti_dtype)
            _dummy_kernel_precompile(dummy_scalar_ti, dummy_vector_ti)
            ti.sync()
            del dummy_scalar_ti, dummy_vector_ti, dummy_scalar_np, dummy_vector_np, _dummy_kernel_precompile
    except Exception: 
        pass

# --- Other Library Checks (CuPy) --- 
try:
    import cupy as cp
    gpu_id = cp.cuda.Device().id
    CUPY_AVAILABLE = True
    print(f"CuPy: Found. Using GPU ID: {gpu_id}")
except ImportError:
    print("CuPy: Not found. CuPy features disabled.")
except Exception as e_cupy:
     print(f"CuPy: Found but failed to initialize GPU: {e_cupy}. CuPy features disabled.")
     CUPY_AVAILABLE = False

# --- Final Library Status ---
print("-" * 30)
print("Library Status Summary:")
print(f"  Taichi:       {'Available' if TAICHI_INITIALIZED else 'Unavailable'}")
if TAICHI_INITIALIZED:
    actual_arch_name_str = next((k for k, v in ti.__dict__.items() if v == TAICHI_ACTUAL_ARCH), 'Unknown')
    print(f"    Backend:    {actual_arch_name_str}, FP={TAICHI_EFFECTIVE_FP_TYPE}, IP={TAICHI_EFFECTIVE_IP_TYPE}, f64 Support={TAICHI_BACKEND_SUPPORTS_F64}")
print(f"  CuPy:         {'Available' if CUPY_AVAILABLE else 'Unavailable'}")
print("-" * 30)


# ===========================================
# --- Global State & Server Setup ---
# ===========================================

sim: Simulator | None = None
sim_lock = threading.Lock()
simulation_thread: threading.Thread | None = None
stop_simulation_flag = threading.Event()
main_thread_task_queue = queue.Queue()
backend_flags = {} 

# --- Flask & SocketIO App ---
UI_FOLDER = os.path.join(PROJECT_ROOT, 'ui')
print(f"Serving UI static files from: {os.path.normpath(UI_FOLDER)}")
if not os.path.isdir(UI_FOLDER): print(f"ERROR: UI folder not found at {UI_FOLDER}")
app = Flask(__name__, static_folder=None) # serve static files via routes
app.config['SECRET_KEY'] = os.urandom(24) or 'dev-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ==================================
# --- Backend Helper Functions ---
# ==================================

def _to_serializable(data):
    """Recursively converts NumPy/Taichi types to JSON-serializable Python types."""
    if isinstance(data, (np.int64, np.int32, np.uint64, np.uint32)): return int(data)
    if isinstance(data, (np.float64, np.float32)): return float(data)
    if isinstance(data, np.ndarray): return data.tolist() if data.size > 0 else []
    if isinstance(data, dict): return {k: _to_serializable(v) for k, v in data.items()}
    if isinstance(data, list): return [_to_serializable(item) for item in data]
    # handle Taichi Fields 
    if TAICHI_INITIALIZED and ti is not None and isinstance(data, ti.Field):
        if data.shape is None or any(s == 0 for s in data.shape): return []
        try: return _to_serializable(data.to_numpy()) # convert via numpy
        except Exception: return "<ti.Field>"
    # handle basic types and fallbacks
    if isinstance(data, (str, int, float, bool, type(None))): return data
    try: return str(data) #attempt string conversion
    except Exception: return f"<unserializable:{type(data).__name__}>"


def get_initial_config():
    """Returns parameter definitions and current default settings for the UI."""
    # import current settings dynamically to ensure updates are reflected
    from config.default_settings import DEFAULT_SETTINGS as current_settings
    return _to_serializable({
        "PARAM_DEFS": PARAM_DEFS,
        "DEFAULT_SETTINGS": current_settings,
        "AVAILABLE_MODELS": AVAILABLE_MODELS,
        "AVAILABLE_INTEGRATORS": AVAILABLE_INTEGRATORS,
        "SIM_VERSION": SIM_VERSION
    })

def get_simulation_state():
    """Returns a snapshot of the current simulation state for the UI."""
    default_state = {
        "time": 0.0, "steps_taken": 0, "status_msg": "Simulation not initialized",
        "running": False, "ended": True, "N": 0, "positions": [], "colors": [],
        "stats": {}, "current_models": {}, "current_integrator": "-", "graph_settings": {}
    }
    if not sim: return default_state
    with sim_lock:
        try: return _to_serializable(sim.get_current_state_for_ui())
        except Exception as e:
            print(f"ERROR fetching simulation state: {e}"); traceback.print_exc()
            # Return default but try to preserve basic status
            error_state = default_state.copy()
            try: error_state["status_msg"] = sim.get_status_message() or "Error State"
            except Exception: error_state["status_msg"] = "Error State"
            try: error_state["time"] = sim.get_time()
            except Exception: pass
            try: error_state["graph_settings"] = sim._config.get('GRAPH_SETTINGS', {})
            except Exception: pass
            return error_state


# ========================================
# --- Main Thread Task Processing ---
# ========================================

def _execute_main_thread_task(task_details):
    """(Runs on Main Thread) Executes tasks requiring main context (Taichi alloc)."""
    task_type = task_details.get('type')
    params = task_details.get('params', {})
    response_queue = task_details.get('response_queue')
    result = {'success': False, 'data': None, 'error': 'Unknown task'}
    # print(f"Main Thread: Executing task '{task_type}'...") # Runtime print

    if not response_queue:
        print("  ERROR: No response queue provided for task."); return

    try:
        if task_type == 'create_pd_for_restart':
            N = params['N']
            numpy_precision = params['numpy_precision']
            taichi_is_active = params['taichi_is_active']
            initial_config = params['initial_config'] # Config needed for model setup

            # 1. Allocate ParticleData
            # print(f"  Allocating ParticleData(N={N}, prec={numpy_precision.__name__}, ti={taichi_is_active})...") # Runtime print
            new_pd = ParticleData(N, allow_implicit_transfers=True,
                                  numpy_precision=numpy_precision,
                                  taichi_is_active=taichi_is_active)
            # print(f"  ParticleData allocation successful.") # Runtime print

            # 2. Pre-Setup Default Taichi Models (if needed)
            pre_initialized_models = {}
            if taichi_is_active:
                # print(f"  Pre-setting up default Taichi models...") # Runtime print
                # Identify which default models *are* Taichi models
                models_to_setup = []
                for m_type in ["sph", "thermo", "gravity"]: # Add other Taichi types if needed
                    default_id = initial_config.get(f'default_{m_type}_model')
                    if default_id and "taichi" in default_id.lower():
                        models_to_setup.append((m_type, default_id))

                if models_to_setup:
                    from sim.utils import dynamic_import # Local import fine here
                    from config.available_models import AVAILABLE_MODELS # Need definitions

                    for model_type, model_id in models_to_setup:
                        model_def = next((m for m in AVAILABLE_MODELS.get(model_type, []) if m['id'] == model_id), None)
                        if model_def:
                             # print(f"    Setting up {model_type} model: {model_id}...") # Runtime print
                             try:
                                 ModelClass = dynamic_import(model_def['module'], model_def['class'])
                                 # Pass the relevant config slice
                                 new_model_instance = ModelClass(config=initial_config)
                                 # *** CRITICAL: Call setup on the main thread ***
                                 new_model_instance.setup(new_pd)
                                 pre_initialized_models[model_type] = new_model_instance
                                 # print(f"    {model_id} setup successful.") # Runtime print
                             except Exception as e_model_setup:
                                  print(f"    ERROR pre-setting up {model_id}: {e_model_setup}")
                                  raise RuntimeError(f"Failed main thread setup for {model_id}") from e_model_setup
                        else: print(f"    Warning: Could not find definition for Taichi model {model_id}")
                # else: print("  No default Taichi models specified in config to pre-setup.") # Runtime print

            # 3. Prepare Response
            result = {'success': True, 'data': {
                'particle_data': new_pd,
                'initialized_models': pre_initialized_models
            }, 'error': None}
        # Add other task types here if needed ('shutdown' is handled by None signal)

    except Exception as e:
        print(f"ERROR in Main Thread Task '{task_type}': {e}")
        traceback.print_exc()
        result = {'success': False, 'data': None, 'error': str(e)}

    # Send response back to the requesting thread
    try: response_queue.put(result); # print(f"Main Thread: Sent response for '{task_type}'.") # Runtime print
    except Exception as e_resp: print(f"ERROR: Failed sending response for task {task_type}: {e_resp}")


def main_thread_task_processor():
    """(Runs on Main Thread) Loop that waits for and executes tasks."""
    print("Main Thread: Task processor started. Waiting for tasks...")
    while True:
        try:
            task = main_thread_task_queue.get(block=True) # wait indefinitely until appply and restart button pressed
            if task is None: #shutdown signal
                print("Main Thread: Received shutdown signal. Exiting task processor.")
                break
            _execute_main_thread_task(task)
            main_thread_task_queue.task_done()
        except queue.Empty: continue #wont happen if block=True
        except Exception as e:
             print(f"ERROR in main thread task processor loop: {e}")
             traceback.print_exc()
             # Try to notify worker if possible
             response_queue = task.get('response_queue') if isinstance(task, dict) else None
             if response_queue:
                  try: response_queue.put({'success': False, 'data': None, 'error': f'Main thread loop error: {e}'})
                  except Exception: pass
    print("Main Thread: Task processor finished.")


# =====================================
# --- Command Handling (Sim Thread) ---
# =====================================

def handle_command(command_data):
    """(Runs on Sim Thread/Server Thread) Handles commands from the frontend."""
    global simulation_thread, sim # Use global sim
    command = command_data.get('command')

    # allow restart even if sim failed init, but other commands need sim
    if command != 'restart' and not sim:
         return {"success": False, "message": "Simulation not initialized"}

    response = {"success": True, "message": f"Command '{command}' received."}
    needs_state_update = False

    try:
        with sim_lock: # lock during command processing
            if command == 'toggle_run':
                if sim.is_ended(): response = {"success": False, "message": "Cannot run/pause, simulation has ended."}
                else:
                    sim.toggle_run()
                    if sim.is_running():
                        # start worker thread if not running
                        if simulation_thread is None or not simulation_thread.is_alive():
                            stop_simulation_flag.clear()
                            simulation_thread = threading.Thread(target=simulation_loop_worker, daemon=True)
                            simulation_thread.start()
                            # print("Simulation worker thread started.") # Runtime print
                        response["isRunning"] = True
                    else:
                        # print("Simulation paused.") # Runtime print
                        response["isRunning"] = False
                    needs_state_update = True

            elif command == 'step':
                if not sim.is_running() and not sim.is_ended():
                    sim.step_forward()
                    response["message"] = f"Stepped forward to time {sim.get_time():.4f}"
                    needs_state_update = True
                elif sim.is_running(): response = {"success": False, "message": "Cannot step while running."}
                else: response = {"success": False, "message": "Cannot step, simulation ended."}

            elif command == 'reset':
                # stop worker thread first
                if simulation_thread and simulation_thread.is_alive():
                    stop_simulation_flag.set()
                    simulation_thread.join(timeout=1.0)
                    if simulation_thread.is_alive(): print("Warning: Sim worker did not stop cleanly on reset.")
                    simulation_thread = None
                    stop_simulation_flag.clear()
                # reset simulator state
                sim.reset_to_initial()
                response["message"] = "Simulation reset to initial state."
                response["isRunning"] = False
                needs_state_update = True

            elif command == 'restart':
                # stop worker thread first
                if simulation_thread and simulation_thread.is_alive():
                    stop_simulation_flag.set()
                    simulation_thread.join(timeout=1.0)
                    if simulation_thread.is_alive(): print("Warning: Sim worker did not stop cleanly on restart.")
                    simulation_thread = None
                    stop_simulation_flag.clear()

                # prepare settings
                settings_from_ui = command_data.get('settings', {})
                from config.default_settings import DEFAULT_SETTINGS as current_defaults
                merged_settings = current_defaults.copy()
                merged_settings.update(settings_from_ui)
                # so effective types aren't overridden by UI settings
                merged_settings['_effective_np_float_type'] = current_defaults['_effective_np_float_type']
                merged_settings['_effective_ti_float_type'] = current_defaults['_effective_ti_float_type']
                merged_settings['_effective_ti_int_type'] = current_defaults['_effective_ti_int_type']
                merged_settings['USE_DOUBLE_PRECISION'] = current_defaults['USE_DOUBLE_PRECISION']

                # get selections
                gravity_id = command_data.get('gravity')
                sph_id = command_data.get('sph')
                thermo_id = command_data.get('thermo')
                color_id = command_data.get('color')
                integrator_id = command_data.get('integrator')

                try:
                    if not sim:
                        sim = Simulator(
                            initial_settings=merged_settings, 
                            backend_availability_flags=backend_flags,
                            main_thread_task_queue_ref=main_thread_task_queue 
                        )

                    # delegate simulator instance for restart
                    sim.restart(merged_settings, gravity_id, sph_id, thermo_id, color_id, integrator_id)
                    response["message"] = "Simulation restarted successfully."
                    response["isRunning"] = False
                except Exception as e_restart:
                     print(f"ERROR during simulation restart: {e_restart}"); traceback.print_exc()
                     response = {"success": False, "message": f"Restart Failed: {e_restart}"}
                     if sim: sim._status_msg = "Restart Failed"; sim._ended = True; sim._running = False
                needs_state_update = True

            elif command == 'set_param':
                key, value = command_data.get('key'), command_data.get('value')
                if key and value is not None:
                    try: sim.set_live_parameter(key, value); response["message"] = f"Param '{key}' set."
                    except Exception as e: response = {"success": False, "message": f"Failed: {e}"}
                else: response = {"success": False, "message": "Missing key/value."}

            elif command == 'set_flag':
                 key, value = command_data.get('key'), command_data.get('value')
                 if key and value is not None:
                     try: sim.set_thermo_flag(key, value); response["message"] = f"Flag '{key}' set."
                     except Exception as e: response = {"success": False, "message": f"Failed: {e}"}
                 else: response = {"success": False, "message": "Missing key/value."}

            elif command == 'select_model':
                 m_type, m_id = command_data.get('type'), command_data.get('id')
                 if m_type and m_id:
                     try: sim.select_model(m_type, m_id); response["message"] = f"{m_type.capitalize()} model set."; needs_state_update = True
                     except ValueError as e: response = {"success": False, "message": str(e)}
                 else: response = {"success": False, "message": "Missing type/id."}

            elif command == 'select_integrator':
                 i_id = command_data.get('id')
                 if i_id:
                     try: sim.select_integrator(i_id); response["message"] = f"Integrator set."; needs_state_update = True
                     except ValueError as e: response = {"success": False, "message": str(e)}
                 else: response = {"success": False, "message": "Missing integrator id."}

            else: response = {"success": False, "message": f"Unknown command: {command}"}

    except Exception as e:
        print(f"ERROR handling command '{command}': {e}"); traceback.print_exc()
        response = {"success": False, "message": f"Internal Server Error: {e}"}
        if sim:
             try: sim._status_msg = "Command Error"; sim._ended = True; sim._running = False; needs_state_update = True
             except Exception: pass

    response["_needs_immediate_update"] = needs_state_update
    return response

# ===========================================
# --- Simulation Worker Thread ---
# ===========================================

def simulation_loop_worker():
    """(Runs on Worker Thread) Main loop for advancing the simulation."""
    global sim
    last_state_send_time = time.time()
    state_send_interval = 1.0 / 30.0 # ~30 FPS UI updates

    while not stop_simulation_flag.is_set():
        loop_start_time = time.perf_counter()
        sim_state = {} 

        # --- Safely Get Sim State Under Lock ---
        with sim_lock:
            if sim is None: time.sleep(0.2); continue # Wait if sim not ready
            sim_state['running'] = sim.is_running()
            sim_state['ended'] = sim.is_ended()
            if sim_state['ended']: break # kill loop if done
            if not sim_state['running']: time.sleep(0.1); continue

            # parameters needed for this loop iteration
            sim_state['step'] = sim.get_steps_taken()
            sim_state['N'] = sim.get_particle_count()
            sim_state['dt'] = sim.get_dt()
            sim_state['G_config'] = 0.0
            if hasattr(sim, '_config') and sim._config:
                 sim_state['graph_interval'] = sim._config.get('GRAPH_SETTINGS', {}).get('plot_interval_steps', 10)
                 sim_state['graph_enabled'] = sim._config.get('GRAPH_SETTINGS', {}).get('enable_plotting', False)
                 sim_state['G_config'] = sim._config.get('G', 0.0)
            else: 
                 sim_state['graph_interval'] = 10
                 sim_state['energy_interval'] = 5
                 sim_state['graph_enabled'] = False


        # --- Advance Simulation Step ---
        advance_success = False
        try:
            advance_success = sim.advance_one_step() # core work
        except Exception as e_step:
            print(f"\n!!! ERROR during sim.advance_one_step() at step {sim_state['step'] + 1}: {e_step} !!!")
            traceback.print_exc()
            # set error state and emit final update
            try:
                with sim_lock:
                    if sim: sim._status_msg = "Error During Step"; sim._running = False; sim._ended = True
                final_state = get_simulation_state()
                if final_state and final_state.get("status_msg") != "Sim not initialized":
                     socketio.emit('state_update', final_state)
            except Exception as e_final_err: print(f"Error setting/sending final error state: {e_final_err}")
            break 

        # --- Post-Step Processing (if successful) ---
        if advance_success:
            step_just_completed = sim_state['step'] + 1

            # --- Collect Graph Data ---
            should_collect_graph = (sim_state['graph_enabled'] and step_just_completed > 0 and
                                    step_just_completed % sim_state['graph_interval'] == 0)
            if should_collect_graph:
                try:
                    with sim_lock:
                        if sim: sim.collect_graph_data(force_collect=False)
                except Exception as e_graph: print(f"ERROR collecting graph data: {e_graph}")

        # --- Send Periodic State Update to UI ---
        current_time = time.time()
        if current_time - last_state_send_time >= state_send_interval:
            try: socketio.emit('state_update', get_simulation_state()) # Fetches state under lock
            except Exception as e_emit: print(f"Warning: Error emitting state update: {e_emit}")
            last_state_send_time = current_time
        # --- Loop Rate Control ---
        loop_duration = time.perf_counter() - loop_start_time
        time.sleep(max(0.0001, 0.002 - loop_duration)) # rate limit to stop inf loop

    # --- End of Worker Loop ---
    # time to collect final snapshot data if loop exited due to sim ending
    with sim_lock:
        if sim and sim.is_ended():
            if hasattr(sim, '_graph_data') and sim._graph_data is not None and not sim._graph_data.get('final_snapshot'):
                 try: sim._collect_final_snapshot_data()
                 except Exception as e_final: print(f"ERROR collecting final snapshot: {e_final}")

# ============================================
# --- Flask Routes & SocketIO Handlers ---
# ============================================

@app.route('/')
def route_index():
    """Serves the main UI page."""
    return send_from_directory(UI_FOLDER, 'index.html')

@app.route('/<path:filename>')
def route_static_files(filename):
    """Serves static files (JS, CSS) from the UI folder."""
    return send_from_directory(UI_FOLDER, filename)

@app.route('/api/config')
def route_config_http():
    """HTTP endpoint to get initial simulation configuration."""
    return jsonify(get_initial_config())

@app.route('/api/state')
def route_state_http():
    """HTTP endpoint to get current simulation state."""
    return jsonify(get_simulation_state())

@app.route('/api/command', methods=['POST'])
def route_command_http():
    """HTTP endpoint to send commands to the simulation."""
    data = request.get_json()
    if not data: return jsonify({"success": False, "message": "Invalid request"}), 400
    response_dict = handle_command(data)
    response_dict.pop("_needs_immediate_update", None) 
    return jsonify(response_dict)

@app.route('/api/generate_pdf', methods=['POST'])
def route_generate_pdf():
    """HTTP endpoint to trigger PDF plot generation."""
    global sim
    print("Received request to generate plots PDF...")
    if not sim: return jsonify({"success": False, "message": "Sim not initialized."}), 400
    try:
        with sim_lock:
             graph_data = sim.get_graph_data()
             graph_settings = sim._config.get('GRAPH_SETTINGS', {}) if hasattr(sim, '_config') else {}
        if not graph_data: return jsonify({"success": False, "message": "No graph data."}), 400
        # Generate PDF
        pdf_filepath = generate_plots_pdf(graph_data, graph_settings, OUTPUT_DIR)
        if pdf_filepath:
            pdf_url = f"output/{os.path.basename(pdf_filepath)}"
            return jsonify({"success": True, "message": "PDF generated.", "filepath": pdf_url})
        else: return jsonify({"success": False, "message": "Failed PDF generation."}), 500
    except Exception as e:
        print(f"ERROR during PDF generation request: {e}"); traceback.print_exc()
        return jsonify({"success": False, "message": f"Server Error: {e}"}), 500

# --- SocketIO Handlers ---
@socketio.on('connect')
def handle_connect():
    """Handles new client WebSocket connection."""
    sid = request.sid
    print(f'Client connected: {sid}')
    try:
        emit('config', get_initial_config()) 
        emit('state_update', get_simulation_state()) 
        print(f"Sent initial config and state to {sid}")
    except Exception as e: print(f"Error sending initial data to {sid}: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client WebSocket disconnection."""
    print(f'Client disconnected: {request.sid}')

@socketio.on('send_command')
def handle_command_ws(command_data):
    """Handles commands received via WebSocket."""
    sid = request.sid
    print(f'Received command via WS from {sid}: {command_data}')
    response_dict = handle_command(command_data)
    needs_update = response_dict.pop("_needs_immediate_update", False)
    emit('command_response', response_dict, room=sid) 
    if needs_update: 
        print(f"Command requires immediate state broadcast.")
        socketio.emit('state_update', get_simulation_state())

@socketio.on('request_state')
def handle_request_state():
     """Handles explicit request for state update from a client."""
     sid = request.sid
     print(f"Received explicit state request from {sid}.")
     emit('state_update', get_simulation_state(), room=sid) # TODO: ADD COMMENT TO EXPLAIN THIS BIT


# ==================================
# --- Main Execution Logic ---
# ==================================

def main_backend_setup():
    """(Runs on Main Thread) Initializes the backend components, including the Simulator."""
    global sim, backend_flags, main_thread_task_queue
    print("Setting up initial simulation instance...")
    backend_flags.update({
            "taichi": TAICHI_INITIALIZED,
            "cupy": CUPY_AVAILABLE,
            "taichi_arch": TAICHI_ACTUAL_ARCH,
            "taichi_supports_f64": TAICHI_BACKEND_SUPPORTS_F64,
            "effective_np_float_type": DEFAULT_SETTINGS['_effective_np_float_type'],
            "effective_ti_float_type": DEFAULT_SETTINGS['_effective_ti_float_type'],
            "effective_ti_int_type": DEFAULT_SETTINGS['_effective_ti_int_type'],
    })

    try:
        initial_sim_settings = DEFAULT_SETTINGS.copy()
        initial_sim_settings['start_running'] = False

        sim = Simulator(
            initial_settings=initial_sim_settings,
            backend_availability_flags=backend_flags,
            main_thread_task_queue_ref=main_thread_task_queue
        )

        print(f"Simulation instance created (v{SIM_VERSION}).") 
        return True

    except Exception as e:
        print(f"\n!!! FATAL ERROR during initial simulation setup: {e} !!!")
        traceback.print_exc()
        sim = None
        print("Backend FAILED to Initialize.")
        return False

# --- Application Entry Point ---
if __name__ == "__main__":
    if main_backend_setup():
        print("\n--- Starting Server and Task Processor ---") 
        server_thread = threading.Thread(
            target=lambda: socketio.run(
                app, host='0.0.0.0', port=7847, debug=False, use_reloader=False, allow_unsafe_werkzeug=True
            ), daemon=True
        )
        server_thread.start()
        print(f"   - Server running. Access UI at: http://localhost:7847/") # Concise
        print(   "   - Main thread processing tasks. Press Ctrl+C to stop.")
        try:
             main_thread_task_processor()
        except KeyboardInterrupt:
             print("\nCtrl+C received, shutting down gracefully...")
        finally:
             print("Initiating shutdown sequence...")
             main_thread_task_queue.put(None)
             stop_simulation_flag.set()
             if simulation_thread is not None and simulation_thread.is_alive():
                 # redundant block
                 print("  Waiting for simulation worker thread...")
                 simulation_thread.join(timeout=1.5)
                 if simulation_thread.is_alive(): print("  Warning: Simulation thread did not exit cleanly.")
             with sim_lock:
                 if sim and hasattr(sim, '__del__'):
                      try: sim.__del__()
                      except Exception: pass
             print("Shutdown sequence finished.")
    else:
        print("\nBackend setup failed. Server not started.")
        sys.exit(1)
