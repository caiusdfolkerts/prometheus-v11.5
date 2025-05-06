# prometheus/sim/particle_data.py

"""
Manages particle arrays across different compute devices (CPU numpy, GPU taichi, NVIDIA CUDA cupy).

This Class is central to handling data consistency and transfers between the CPU
and the GPU governed by Taichi or CuPy 

It tracks the location of each particle attribute and performs data transfers
implicitly (if enabled) or explicitly via the 'ensure; method.
"""

import numpy as np
import time
import gc
import traceback
from typing import Dict, List, Optional, Union, Set, Tuple

# conditional imports for GPU libraries
try:
    import taichi as ti
    from taichi.lang.util import to_taichi_type # Helper for type conversion
    HAVE_TAICHI = True
except ImportError:
    ti = None # define ti as none if import fails
    to_taichi_type = None # define helper as none
    HAVE_TAICHI = False

try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    cp = None
    HAVE_CUPY = False

# type definitions
DeviceStr = str
AttrName = str      # e.g., "positions", "masses"
NpArray = np.ndarray
# placeholders for types that require the library to be imported (will result in error and likely exit down the line, TODO: implement defaulting to CPU if taichi not found, requires SPH numba method)
TiField = object
CpArray = object


class ParticleData:
    """
    Manages particle attribute arrays across cpu/gpu, handling transfers + sync.

    tracks data location ('cpu', 'gpu:ti', etc.) and handles data transfers
    between numpy arrays and backend-specific formats (taichi fields, cupy arrays).
    """
    VALID_DEVICES = {"cpu", "gpu:ti", "gpu:cupy"}

    def __init__(self, N: int, allow_implicit_transfers: bool = True, numpy_precision: type = np.float64, taichi_is_active: bool = False):
        if not isinstance(N, int) or N < 0:
            raise ValueError(f"N must be a non-negative integer, got {N}")
        self.N = N # number of particles
        self._allow_implicit_transfers = allow_implicit_transfers # flag for implicit data transfers
        self._locked_writeable: Dict[AttrName, bool] = {} # tracks if an attribute is locked for writing on cpu
        self._internal_numpy_fp_type = numpy_precision # default floating point precision for numpy arrays
        self._internal_numpy_int_type = np.int32 # default integer precision for numpy arrays
        self._taichi_is_active = taichi_is_active and HAVE_TAICHI # flag indicating if taichi is active and available

        _default_backends = self.VALID_DEVICES # default set of supported backends for attributes
        # definitions for all particle attributes: (shape_suffix, dtype, supported_backends)
        self._attr_definitions = {
            "positions":            ((3,), self._internal_numpy_fp_type, _default_backends),
            "velocities":           ((3,), self._internal_numpy_fp_type, _default_backends),
            "forces":               ((3,), self._internal_numpy_fp_type, _default_backends),
            "forces_grav":          ((3,), self._internal_numpy_fp_type, _default_backends),
            "accelerations":        ((3,), self._internal_numpy_fp_type, _default_backends),
            "masses":               ((),   self._internal_numpy_fp_type, _default_backends),
            "internal_energies":    ((),   self._internal_numpy_fp_type, _default_backends),
            "densities":            ((),   self._internal_numpy_fp_type, _default_backends),
            "pressures":            ((),   self._internal_numpy_fp_type, _default_backends),
            "temperatures":         ((),   self._internal_numpy_fp_type, _default_backends),
            "work_terms":           ((),   self._internal_numpy_fp_type, _default_backends),
            "cooling_rates":        ((),   self._internal_numpy_fp_type, _default_backends),
            "fusion_rates":         ((),   self._internal_numpy_fp_type, _default_backends),
            "visc_heating_terms":   ((),   self._internal_numpy_fp_type, _default_backends),
            "v_half_temp":          ((3,), self._internal_numpy_fp_type, _default_backends),
            "work_terms_prev":      ((),   self._internal_numpy_fp_type, _default_backends),
            "visc_heating_terms_prev":((),  self._internal_numpy_fp_type, _default_backends),
            "cooling_rates_prev":   ((),   self._internal_numpy_fp_type, _default_backends),
            "fusion_rates_prev":    ((),   self._internal_numpy_fp_type, _default_backends),
            "colors":               ((3,), np.float32,                 _default_backends),
        }
        self._data_cpu: Dict[AttrName, Optional[NpArray]] = {} # stores numpy arrays on cpu
        self._location: Dict[AttrName, DeviceStr] = {} # tracks the current authoritative location of each attribute
        self._gpu_copies: Dict[AttrName, Dict[DeviceStr, object]] = {} # stores gpu copies (e.g., taichi fields, cupy arrays)
        self._gpu_dirty: Dict[AttrName, bool] = {} # flags if a gpu copy is stale relative to the cpu version

        # initialize cpu data arrays and metadata for each attribute
        for name, (shape_suffix, dtype, _) in self._attr_definitions.items():
            full_shape = (self.N,) + shape_suffix
            try:
                 self._data_cpu[name] = np.zeros(full_shape, dtype=dtype)
            except MemoryError:
                 print(f"ERROR: ParticleData Failed allocating NumPy '{name}' {full_shape}. Not enough memory.")
                 self._data_cpu = {}; gc.collect(); raise
            except Exception as e:
                 print(f"ERROR: ParticleData Failed allocating NumPy '{name}' {full_shape}: {e}")
                 self._data_cpu = {}; gc.collect(); raise
            self._location[name] = "cpu"
            self._gpu_copies[name] = {}
            self._gpu_dirty[name] = False
            self._locked_writeable[name] = False

        self._cupy_device = self._determine_cupy_device() # determine the primary CuPy device


    def _determine_cupy_device(self) -> Optional[int]:
        """determines the primary available cupy cuda device id."""
        if HAVE_CUPY:
            try: return cp.cuda.Device().id
            except cp.cuda.runtime.CUDARuntimeError: return None # no CUDA device found
            except Exception as e: print(f"Warn: Error getting CuPy device: {e}"); return None
        return None

    def _validate_attribute(self, name: AttrName):
        """raises keyerror if attribute name is invalid."""
        if name not in self._attr_definitions:
            raise KeyError(f"Unknown attribute: '{name}'. Valid: {list(self._attr_definitions.keys())}")

    def _validate_device(self, device: DeviceStr):
        """raises valueerror or runtimeerror if the device is invalid or unavailable."""
        if device not in self.VALID_DEVICES:
            raise ValueError(f"Invalid device: '{device}'. Valid: {self.VALID_DEVICES}")
        if device == "gpu:ti" and not HAVE_TAICHI: raise RuntimeError("Taichi required but unavailable.")
        if device == "gpu:cupy" and not HAVE_CUPY: raise RuntimeError("CuPy required but unavailable.")

    # public info getters
    def get_n(self) -> int: return self.N
    def get_dtype(self, name: AttrName) -> np.dtype:
        self._validate_attribute(name)
        return self._attr_definitions[name][1]
    def get_shape(self, name: AttrName) -> Tuple[int, ...]:
        self._validate_attribute(name)
        return (self.N,) + self._attr_definitions[name][0]
    def get_attribute_names(self) -> List[AttrName]: return list(self._attr_definitions.keys())

    # core data access & management

    def set(self, name: AttrName, data: NpArray, source_device: DeviceStr = "cpu"):
        """updates an attribute's data (from cpu), invalidating gpu copies."""
        self._validate_attribute(name)
        if source_device != "cpu": raise NotImplementedError("Setting from non-CPU not implemented.")
        expected_shape = self.get_shape(name)
        expected_dtype = self.get_dtype(name)
        if not isinstance(data, np.ndarray): raise TypeError(f"set('{name}') data must be NumPy array")
        if data.shape != expected_shape: raise ValueError(f"Shape mismatch set('{name}'): {data.shape} vs {expected_shape}")
        if data.dtype != expected_dtype:
             try: data = data.astype(expected_dtype, copy=False)
             except (TypeError, ValueError): raise TypeError(f"Dtype mismatch set('{name}'): {data.dtype} vs {expected_dtype}")
        self._data_cpu[name] = np.copy(data)
        self._location[name] = "cpu"
        self._gpu_dirty[name] = True
        self._locked_writeable[name] = False

    def get(self, name: AttrName, device: DeviceStr = "cpu", writeable: bool = False) -> Union[NpArray, TiField, CpArray]:
        """gets attribute data on the specified device, handling transfers if needed."""
        self._validate_attribute(name)
        self._validate_device(device)
        current_location = self._location.get(name, 'cpu')

        if writeable: # request for writeable cpu access
            if device != "cpu": raise ValueError("Writeable access only for device='cpu'.")
            if self._locked_writeable.get(name, False): raise RuntimeError(f"'{name}' already locked for writing.")
            if current_location != "cpu": # if data is on gpu, transfer it back to cpu
                 gpu_data = self._get_gpu_data_internal(name, current_location)
                 if gpu_data is None: print(f"Warn: Authoritative GPU data '{name}' missing from '{current_location}'. Writeable get might return stale CPU data.")
                 else: self._data_cpu[name] = self._transfer_gpu_to_cpu(name, gpu_data, current_location)
            self._locked_writeable[name] = True
            self._location[name] = "cpu"
            self._gpu_dirty[name] = True # mark gpu copies as dirty
            return self._data_cpu[name]

        if current_location == device: # data already where needed
            if device == "cpu": return self._data_cpu.get(name)
            gpu_data = self._get_gpu_data_internal(name, device)
            if gpu_data is not None: return gpu_data
            print(f"Warn: Location '{device}' but GPU copy missing for '{name}'. Forcing transfer.") # fallback if gpu copy is unexpectedly missing

        elif device == "cpu": # target cpu, data on gpu -> transfer gpu->cpu
            if not self._allow_implicit_transfers: raise RuntimeError(f"GPU->CPU needed for '{name}', implicit disabled.")
            gpu_data = self._get_gpu_data_internal(name, current_location)
            if gpu_data is None: raise RuntimeError(f"Cannot transfer '{name}': Authoritative GPU data missing from '{current_location}'.")
            cpu_data = self._transfer_gpu_to_cpu(name, gpu_data, current_location)
            self._data_cpu[name] = cpu_data
            self._location[name] = "cpu"
            self._gpu_dirty[name] = False # cpu data is now up-to-date
            return cpu_data

        elif device != "cpu": # target gpu, needs transfer (cpu->gpu or gpu_source->cpu->gpu_target)
            if not self._allow_implicit_transfers: raise RuntimeError(f"Transfer needed for '{name}' to '{device}', implicit disabled.")
            source_np_data = None
            if current_location == "cpu": source_np_data = self._data_cpu.get(name)
            else: # transfer via cpu: gpu_source -> cpu -> gpu_target
                gpu_data_source = self._get_gpu_data_internal(name, current_location)
                if gpu_data_source is None: raise RuntimeError(f"Cannot transfer '{name}': Source GPU data missing from '{current_location}'.")
                source_np_data = self._transfer_gpu_to_cpu(name, gpu_data_source, current_location)
                self._data_cpu[name] = source_np_data # update cpu cache
            if source_np_data is None: raise RuntimeError(f"Cannot transfer '{name}': Source NumPy data unavailable.")

            target_gpu_data = self._transfer_data(name, source_np_data, device)
            if name not in self._gpu_copies: self._gpu_copies[name] = {}
            self._gpu_copies[name][device] = target_gpu_data
            self._location[name] = device
            self._gpu_dirty[name] = False # gpu copy is now up-to-date
            return target_gpu_data

        raise RuntimeError(f"Unhandled state get('{name}', device='{device}', loc='{current_location}')")

    def _get_gpu_data_internal(self, name: AttrName, device: DeviceStr) -> Optional[object]:
        """safely retrieves a gpu object from cache."""
        return self._gpu_copies.get(name, {}).get(device)

    def ensure(self, names: Union[AttrName, List[AttrName], str], target_device: DeviceStr):
        """ensures attribute(s) are present and up-to-date on the target gpu device."""
        self._validate_device(target_device)
        if target_device == "cpu": return # no action needed for cpu
        if isinstance(names, str): names = names.split() # allow space-separated string of names
        if not isinstance(names, list): names = [names]

        for name in names:
            self._validate_attribute(name)
            current_location = self._location.get(name, 'cpu')
            gpu_copy_exists = target_device in self._gpu_copies.get(name, {})
            is_stale = self._gpu_dirty.get(name, True) or current_location != target_device
            needs_transfer = not gpu_copy_exists or is_stale

            if needs_transfer:
                source_np_data = self._data_cpu.get(name) # always transfer from cpu for simplicity/consistency
                if source_np_data is None: raise RuntimeError(f"Cannot ensure '{name}': Source CPU data missing.")
                target_gpu_data = self._transfer_data(name, source_np_data, target_device)
                if name not in self._gpu_copies: self._gpu_copies[name] = {}
                self._gpu_copies[name][target_device] = target_gpu_data
                # mark gpu copy as up-to-date relative to cpu (simplification)
                if current_location == "cpu": self._gpu_dirty[name] = False

        self.synchronize(target_device) # ensure all transfers are complete

    def _transfer_gpu_to_cpu(self, name: AttrName, gpu_data: object, source_device: DeviceStr) -> NpArray:
        """internal helper: transfers data from gpu to cpu numpy array."""
        target_np_dtype = self.get_dtype(name)
        np_array = None
        try:
            if source_device == "gpu:ti":
                if not (HAVE_TAICHI and isinstance(gpu_data, ti.Field)): raise TypeError("Expected Taichi Field.")
                ti.sync(); np_array = gpu_data.to_numpy()
            elif source_device == "gpu:cupy":
                if not (HAVE_CUPY and isinstance(gpu_data, cp.ndarray)): raise TypeError("Expected CuPy ndarray.")
                np_array = gpu_data.get()
            else: raise ValueError(f"Unsupported GPU->CPU source device: {source_device}")

            if np_array is None: raise RuntimeError(f"Transfer resulted in None for {name} from {source_device}")
            return np_array.astype(target_np_dtype, copy=False) if np_array.dtype != target_np_dtype else np_array
        except Exception as e:
            print(f"ERROR GPU->CPU transfer '{name}' from '{source_device}': {e}"); traceback.print_exc()
            raise RuntimeError("GPU->CPU transfer failed.") from e

    def synchronize(self, device: DeviceStr):
        """synchronizes the execution stream/queue for the specified device."""
        if device == "cpu": return
        elif device == "gpu:ti":
             if HAVE_TAICHI and ti.lang.impl.current_cfg().arch != ti.cpu: # only sync if taichi is on a gpu backend
                 try: ti.sync()
                 except Exception as e: print(f"Warn: ti.sync() failed: {e}")
        elif device == "gpu:cupy":
             if HAVE_CUPY and self._cupy_device is not None:
                 try: cp.cuda.get_current_stream().synchronize()
                 except Exception as e: print(f"Warn: cupy sync failed: {e}")

    def release_writeable(self, name: AttrName):
         """releases a write lock obtained via `get(..., writeable=true)`."""
         self._validate_attribute(name)
         self._locked_writeable[name] = False # okay to call even if not locked

    def get_numpy_dtype(self, name: AttrName) -> type:
        """returns the internal numpy dtype used for the attribute."""
        self._validate_attribute(name)
        return self._attr_definitions[name][1]

    def get_effective_precision_is_f64(self) -> bool:
        """returns true if the internal numpy float precision is float64."""
        return self._internal_numpy_fp_type == np.float64

    def _transfer_data(self, name: AttrName, source_np_array: NpArray, target_device: DeviceStr) -> object:
        """internal helper: transfers numpy data to a target accelerator device."""
        try:
            # taichi transfer
            if target_device == "gpu:ti":
                if not HAVE_TAICHI: raise RuntimeError("Taichi not available.")
                target_ti_dtype = ti.lang.impl.current_cfg().default_fp # use taichi's default float precision
                expected_np_dtype = self._attr_definitions[name][1]
                if source_np_array is None: raise ValueError(f"Source NumPy for '{name}' is None.")
                if source_np_array.dtype != expected_np_dtype: source_np_array = source_np_array.astype(expected_np_dtype)

                existing_field = self._gpu_copies.get(name, {}).get(target_device)
                field_is_valid = False # check if existing field can be reused
                if existing_field is not None:
                    shape_suffix = self._attr_definitions[name][0]; is_vector = len(shape_suffix) > 0
                    if existing_field.dtype == target_ti_dtype and existing_field.shape == (self.N,):
                        if is_vector: field_is_valid = hasattr(existing_field, 'n') and existing_field.n == shape_suffix[0]
                        else: field_is_valid = True # scalar field matches

                if field_is_valid: # reuse existing field
                    existing_field.from_numpy(source_np_array); return existing_field
                else: # create new field
                    shape_suffix = self._attr_definitions[name][0]
                    if not shape_suffix: field = ti.field(dtype=target_ti_dtype, shape=self.N) # scalar
                    else: field = ti.Vector.field(shape_suffix[0], dtype=target_ti_dtype, shape=self.N, layout=ti.Layout.SOA) # vector
                    field.from_numpy(source_np_array); return field


            # cupy transfer
            elif target_device == "gpu:cupy":
                 if not HAVE_CUPY: raise RuntimeError("CuPy not available.")
                 if self._cupy_device is None: raise RuntimeError("No suitable CuPy device found.")
                 expected_np_dtype = self._attr_definitions[name][1]
                 if source_np_array is None: raise ValueError(f"Source NumPy for '{name}' is None.")
                 if source_np_array.dtype != expected_np_dtype: source_np_array = source_np_array.astype(expected_np_dtype)
                 with cp.cuda.Device(self._cupy_device): return cp.asarray(source_np_array)
            else:
                raise ValueError(f"Invalid target device '{target_device}' for transfer.")

        except Exception as e:
             print(f"ERROR during data transfer of '{name}' to '{target_device}': {e}")
             if HAVE_TAICHI and "FieldsBuilder finalized" in str(e): # specific taichi error
                  print("\n *** FATAL: Taichi FieldsBuilder finalized error. Ensure fields allocated before kernel launch. ***\n")
             traceback.print_exc()
             raise RuntimeError(f"Data transfer failed for {name} to {target_device}.") from e

    def get_state_for_ui(self, precision: str = 'f32') -> Dict[str, list]:
        """prepares positions and colors as flat lists for ui rendering."""
        out_dtype = np.float32 if precision == 'f32' else np.float64
        ui_state = {'positions': [], 'colors': []}
        if self.N == 0: return ui_state
        try:
            pos_np = self.get('positions', 'cpu') # ensure data is on cpu
            col_np = self.get('colors', 'cpu')
            if pos_np is None or col_np is None: raise ValueError("Position/Color data missing")
            ui_state['positions'] = pos_np.astype(out_dtype, copy=False).flatten().tolist()
            ui_state['colors'] = col_np.astype(out_dtype, copy=False).flatten().tolist()
            return ui_state
        except Exception as e:
             print(f"Error preparing UI state: {e}"); traceback.print_exc()
             return {'positions': [], 'colors': []} # return empty on error

    def cleanup_gpu_resources(self):
        """attempts to explicitly release gpu memory held by this instance."""
        print("Cleaning up ParticleData GPU resources...")
        # destroy taichi fields
        if HAVE_TAICHI:
            for name in self._gpu_copies:
                ti_field = self._gpu_copies[name].get("gpu:ti")
                if ti_field is not None and hasattr(ti_field, 'destroy'):
                     try: ti_field.destroy()
                     except Exception as e: print(f"  Warn: Non-fatal error destroying Ti field '{name}': {e}")
        # clear internal caches
        self._gpu_copies = {name: {} for name in self._attr_definitions}

        # clear cupy memory pool
        if HAVE_CUPY:
             try: cp.get_default_memory_pool().free_all_blocks()
             except Exception as e: print(f"Warn: Error clearing CuPy memory pool: {e}")
        gc.collect() # run garbage collection
        print("GPU resource cleanup finished.")

    def __del__(self):
        """Destructor - prefer explicit cleanup via 'cleanup_gpu_resources'."""
        pass