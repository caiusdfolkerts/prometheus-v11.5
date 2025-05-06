# prometheus/sim/utils.py
"""General utility functions for the simulation backend."""

import numpy as np
import time
import importlib

# Cache for backend availability check results
_backend_availability_cache = {}

def get_memory_usage_gb():
    """Returns current process memory usage in GiB (cross-platform). Returns 0.0 on failure."""
    try:
        # import psutil locally to avoid hard dependency if not used elsewhere frequently
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / (1024**3) # Resident Set Size in GiB
    except ImportError:
        # Silently return 0.0 if psutil not installed
        return 0.0
    except Exception as e:
        print(f"Warning: Error getting memory usage: {e}")
        return 0.0

def timing_decorator(func):
    """Decorator to print the execution time of a function (useful for debugging)."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Timing: {func.__name__:<25} executed in {(end_time - start_time) * 1000:.3f} ms")
        return result
    return wrapper

def dynamic_import(module_name, class_name):
    """Dynamically imports a class from a specified module."""
    try:
        module = importlib.import_module(module_name)
        imported_class = getattr(module, class_name)
        return imported_class
    except ImportError:
        print(f"ERROR: Module '{module_name}' not found.")
        raise # Re-raise the error for calling code to handle
    except AttributeError:
        print(f"ERROR: Class '{class_name}' not found in module '{module_name}'.")
        raise # Re-raise the error
    except Exception as e:
        print(f"ERROR: Failed dynamic import of {module_name}.{class_name}: {e}")
        raise # Re-raise the error

def check_backend_availability(
    backend_id: str,
    taichi_init_flag: bool = False,
    torch_avail_flag: bool = False,
    cupy_avail_flag: bool = False
) -> bool:
    """
    Checks if a required backend is available, using flags determined during startup.
    """
    global _backend_availability_cache
    # Create a cache key based on the backend ID and the state of the flags
    cache_key = (backend_id, taichi_init_flag, torch_avail_flag, cupy_avail_flag)
    if cache_key in _backend_availability_cache:
        return _backend_availability_cache[cache_key]

    is_available = False
    try:
        if backend_id == "numpy":
            is_available = True # NumPy is assumed always available as a core dependency

        elif backend_id == "gpu:ti":
            is_available = taichi_init_flag # Availability depends solely on successful init

        elif backend_id == "gpu:torch":
            # Requires both PyTorch library and a usable GPU device (CUDA or MPS)
            if torch_avail_flag:
                try:
                    import torch # Import only when needed for the check
                    is_available = torch.cuda.is_available() or \
                                   (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
                except ImportError: is_available = False # Should not happen if flag is true, but check
                except Exception as e_torch: print(f"Warn: Error checking torch GPU: {e_torch}"); is_available = False
            else: is_available = False

        elif backend_id == "gpu:cuda": # Specific request for CUDA via CuPy or Taichi
            cupy_ok = cupy_avail_flag # CuPy availability is direct check
            ti_cuda_ok = False
            if taichi_init_flag: # Only check Taichi backend if it initialized
                try:
                    import taichi as ti # Import only when needed
                    ti_cuda_ok = ti.lang.impl.current_cfg().arch == ti.cuda
                except ImportError: ti_cuda_ok = False # Should not happen, but check
                except AttributeError: ti_cuda_ok = False # If Taichi config missing
                except Exception as e_ti: print(f"Warn: Error checking taichi CUDA: {e_ti}"); ti_cuda_ok = False
            is_available = cupy_ok or ti_cuda_ok

        else:
            print(f"Warning: Unknown backend_id '{backend_id}' requested for check.")

    except Exception as e_check:
         # Catch-all for unexpected errors during check
         print(f"Warning: Error in check_backend_availability for '{backend_id}': {e_check}")
         # traceback.print_exc() # Uncomment for debugging if needed
         is_available = False

    _backend_availability_cache[cache_key] = is_available
    return is_available

def format_value_scientific(value, precision=2):
    """Formats a number into scientific notation string, handling non-finite values."""
    if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        return "-" # Return dash for NaN, Inf, or non-numeric types
    if abs(value) < 1e-15: # Handle zero or very small numbers cleanly
        return "0.0e+00"
    return f"{value:.{precision}e}"