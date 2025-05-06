"""
simulation orchestrator.

this class manages the main simulation lifecycle, including:
- holding the simulation configuration (`_config`).
- managing core components: particledata, physicsmanager, integratormanager.
- controlling simulation state (running, paused, ended, time, steps).
- advancing the simulation step-by-step via the integratormanager.
- handling user commands like run, pause, step, reset, restart, parameter changes.
- delegating computationally intensive or backend-specific tasks (like particledata
  reallocation during restart) to the main thread via a task queue.
- collecting and providing simulation state data for ui updates.
- collecting and providing data for diagnostic plots.
"""

import numpy as np
import time
import traceback
from typing import Dict, Any, Optional, List
import queue # used for response queue in restart

# simulation components (absolute imports from project root)
from sim.particle_data import ParticleData
from sim.physics_manager import PhysicsManager
from sim.integrator_manager import IntegratorManager
from sim.constants import CONST_k_B, CONST_m_p # for default cv calculation
from sim.utils import get_memory_usage_gb, format_value_scientific
from scipy.spatial.distance import pdist # for min/max separation calculation

# import the main thread task queue reference passed during initialization
# this is safer than relying on `from __main__ import ...` inside methods
# the actual queue instance is created in main.py and passed to the simulator constructor.

# simulation version
SIM_VERSION = "11.5"

# status messages
STATUS_READY = "Ready"
STATUS_RUNNING = "Running"
STATUS_PAUSED = "Paused"
STATUS_ENDED = "Ended"
STATUS_INITIALIZING = "Initializing..."
STATUS_RESETTING = "Resetting..."
STATUS_RESTARTING = "Restarting..."
STATUS_ERROR = "Error"


class Simulator:
    """
    main simulation orchestrator. manages particle data, physics, integration,
    state, commands, and data collection.
    """
    _initial_energy_components_for_log: Optional[Dict[str, float]] = None # stores initial energy components for logging

    def __init__(
        self,
        initial_settings: Dict[str, Any],
        backend_availability_flags: Dict[str, bool],
        main_thread_task_queue_ref: queue.Queue # accept queue reference
        ):
        """
        initializes the simulator.
        """
        print(f"\n===== Initializing Simulator v{SIM_VERSION} =====")
        self._start_time_init = time.perf_counter()

        # core state
        self._config: Dict[str, Any] = {}
        self._status_msg: str = STATUS_INITIALIZING
        self._running: bool = False
        self._ended: bool = False
        self._backend_flags = backend_availability_flags.copy()
        self._main_thread_task_queue = main_thread_task_queue_ref # store queue reference

        # effective precision (determined in main.py)
        self._effective_np_float_type = initial_settings.get('_effective_np_float_type', np.float32)
        self._effective_ti_float_type = initial_settings.get('_effective_ti_float_type', None)
        self._effective_ti_int_type = initial_settings.get('_effective_ti_int_type', None)
        initial_settings['USE_DOUBLE_PRECISION'] = (self._effective_np_float_type == np.float64)

        # simulation time/step tracking
        self._time: float = 0.0
        self._dt: float = 0.001 # default, overridden by config
        self._steps_taken: int = 0
        self._max_steps: Optional[int] = None
        self._max_time: Optional[float] = None
        self._last_step_duration: float = 0.0 # duration of the last simulation step
        self._bh_node_count_for_plot: int = 0 # for barnes-hut node count plotting

        # graphing state
        self._graph_data: Optional[Dict[str, List]] = None
        self._graph_settings: Dict = {}
        self._graph_log_interval_steps: int = 10
        self._graph_initial_total_energy: Optional[float] = None
        self._graph_accumulated_expected_delta_e: float = 0.0

        # component placeholders
        self._pd: Optional[ParticleData] = None
        self._physics_manager: Optional[PhysicsManager] = None
        self._integrator_manager: Optional[IntegratorManager] = None

        try:
            # step 1: apply initial configuration
            print("1. Processing Initial Configuration...")
            self._configure(initial_settings) # sets self._config, graph settings, etc.
            if self._graph_data is None or 'time' not in self._graph_data:
                 raise AssertionError("_graph_data not initialized by _configure.")

            # step 2: validate particle count & create components
            N = self.get_particle_count()
            if N <= 0: raise ValueError("N must be positive.")
            print(f"2. Creating Core Components (N={N})...")
            self._pd = ParticleData(
                N,
                allow_implicit_transfers=True,
                numpy_precision=self._effective_np_float_type,
                taichi_is_active=self._backend_flags.get("taichi", False)
            )
            self._physics_manager = PhysicsManager(self._pd, self._config, self._backend_flags)
            self._integrator_manager = IntegratorManager(self._pd, self._recompute_forces_and_physics, self._config)

            # step 3: set initial particle state
            print("3. Initializing Particle Distribution...")
            self._initialize_particles()

            # step 4: select & setup initial models/integrator
            print("4. Selecting Initial Models & Integrator...")
            self._select_initial_models_and_integrator()

            # step 5: perform initial physics calculation
            print("5. Performing Initial Physics Calculation (Density, EOS, Forces)...")
            self._perform_initial_calculation()

            # step 6: store initial energy states for graphing
            print("6. Calculating Initial Energy Components (for graphing)...")
            try:
                # store total energy baseline specifically for graphing drift
                # force pe calculation for this initial graphing baseline if g != 0
                force_pe_for_graph_baseline = self._config.get('G', 0.0) != 0.0
                energy_components_for_graph_baseline = self._get_energy_components(force_pe_for_logging=force_pe_for_graph_baseline)
                self._graph_initial_total_energy = energy_components_for_graph_baseline.get("Total")

                self._graph_accumulated_expected_delta_e = 0.0 # reset graph accumulator
                if self._graph_initial_total_energy is not None:
                     print(f"   Initial Energy (for graph): Total={self._graph_initial_total_energy:.6e}")
                else: print("   Warning: Could not calculate initial total energy for graphing.")
            except Exception as e_init_eng:
                print(f"   ERROR calculating initial energy for graphing: {e_init_eng}")

            # step 7: collect initial graph data & finalize
            self.collect_graph_data(force_collect=True) # collect the t=0 state
            self._status_msg = STATUS_READY if not self._running else STATUS_RUNNING
            init_duration = time.perf_counter() - self._start_time_init
            print(f"===== Simulator Initialization Complete ({init_duration:.3f} s) =====")
            print(f"  N={N}, dt={self._dt:.2e}, Precision={self._effective_np_float_type.__name__}")

        except Exception as e:
             self._status_msg = STATUS_ERROR
             self._ended = True
             print(f"\n!!! FATAL ERROR during Simulator Initialization: {e} !!!")
             traceback.print_exc()
             self._pd = None; self._physics_manager = None; self._integrator_manager = None
             raise

    def _configure(self, settings: Dict[str, Any]):
        """applies settings dictionary to internal configuration and related states."""
        self._config = settings.copy()
        self._dt = float(self._config.get('dt', 0.001))
        max_steps_cfg = int(self._config.get('max_steps', -1))
        max_time_cfg = float(self._config.get('max_time', -1.0))
        self._max_steps = max_steps_cfg if max_steps_cfg > 0 else None
        self._max_time = max_time_cfg if max_time_cfg > 0.0 else None
        self._running = bool(self._config.get('start_running', False))
        required_keys = ['N', 'dt', 'G', 'h', 'cv', 'gas_constant_sim']
        if not all(key in self._config for key in required_keys):
            missing = [k for k in required_keys if k not in self._config]
            raise ValueError(f"Missing required config keys: {missing}")
        self._graph_settings = self._config.get('GRAPH_SETTINGS', {})
        interval_cfg = self._graph_settings.get('log_interval_steps', 10)
        self._graph_log_interval_steps = max(1, int(interval_cfg))
        self._initialize_graph_data_lists()
        self._ensure_thermo_params()


    def _get_plot_setting(self, graph_settings: Dict, key: str, default: bool = False) -> bool:
        """safely retrieves a boolean plot setting from the graph settings dict."""
        settings_dict = graph_settings if isinstance(graph_settings, dict) else {}
        return settings_dict.get(key, default)

    def _get_total_energy_for_graphing(self) -> Optional[float]:
        """calculates total energy (ke+u+pe) for graphing baseline."""
        if not self._pd or self.get_particle_count() == 0: return None
        try:
            # use the detailed energy component calculation
            energy_components = self._get_energy_components()
            # sum components, returning none if any are missing
            ke = energy_components.get("KE")
            u = energy_components.get("U")
            pe = energy_components.get("PE")
            if ke is None or u is None or pe is None:
                print("Warning: Missing energy component for graphing total.")
                return None
            return float(ke + u + pe) # ensure result is float
        except Exception as e:
            print(f"Warning: Error calculating total energy for graphing: {e}")
            return None

    def _initialize_graph_data_lists(self):
        """creates/resets lists in _graph_data based on graph_settings."""
        # always include time and step
        self._graph_data = {'time': [], 'step': []}
        gs = self._graph_settings
        if not isinstance(gs, dict): gs = {} # ensure it's a dict

        # add list for each enabled plot type
        if self._get_plot_setting(gs, 'plot_total_energy'): self._graph_data['total_energy'] = []
        if self._get_plot_setting(gs, 'plot_energy_components'):
            self._graph_data['total_ke'] = []; self._graph_data['total_pe'] = []; self._graph_data['total_u'] = []
        if self._get_plot_setting(gs, 'plot_energy_drift'): self._graph_data['energy_drift_percent'] = []
        if self._get_plot_setting(gs, 'plot_linear_momentum'):
            self._graph_data['total_px'] = []; self._graph_data['total_py'] = []; self._graph_data['total_pz'] = []
        if self._get_plot_setting(gs, 'plot_angular_momentum'):
            self._graph_data['total_lx'] = []; self._graph_data['total_ly'] = []; self._graph_data['total_lz'] = []
        if self._get_plot_setting(gs, 'plot_com_velocity'):
            self._graph_data['com_vx'] = []; self._graph_data['com_vy'] = []; self._graph_data['com_vz'] = []
        if self._get_plot_setting(gs, 'plot_virial_ratio'): self._graph_data['virial_ratio'] = []
        if self._get_plot_setting(gs, 'plot_max_rho'): self._graph_data['max_rho'] = []
        if self._get_plot_setting(gs, 'plot_max_temp'): self._graph_data['max_temp'] = []
        if self._get_plot_setting(gs, 'plot_avg_temp'): self._graph_data['avg_temp'] = []
        if self._get_plot_setting(gs, 'plot_energy_rates'):
            self._graph_data['total_cooling_rate'] = []; self._graph_data['total_fusion_rate'] = []
            self._graph_data['total_sph_work_rate'] = []

        # new graphs
        if self._get_plot_setting(gs, 'plot_com_position'):
            self._graph_data['com_x'] = []; self._graph_data['com_y'] = []; self._graph_data['com_z'] = []
        if self._get_plot_setting(gs, 'plot_min_max_separation'):
            self._graph_data['min_separation'] = []; self._graph_data['max_separation'] = []
        if self._get_plot_setting(gs, 'plot_avg_density'): self._graph_data['avg_rho'] = []
        if self._get_plot_setting(gs, 'plot_max_pressure'): self._graph_data['max_pressure'] = []
        if self._get_plot_setting(gs, 'plot_avg_pressure'): self._graph_data['avg_pressure'] = []
        if self._get_plot_setting(gs, 'plot_sph_visc_heat_rate'): self._graph_data['total_visc_heat_rate'] = []
        if self._get_plot_setting(gs, 'plot_num_fusing_particles'): self._graph_data['num_fusing_particles'] = []
        if self._get_plot_setting(gs, 'plot_avg_spec_int_energy'): self._graph_data['avg_spec_u'] = []
        if self._get_plot_setting(gs, 'plot_bh_nodes'): self._graph_data['bh_num_nodes'] = []
        if self._get_plot_setting(gs, 'plot_step_timing'): self._graph_data['step_duration_ms'] = []

        # placeholder for final state data (histograms, profiles)
        self._graph_data['final_snapshot'] = {}


    def _get_min_max_separation(self) -> tuple[float, float]:
        """calculates min and max inter-particle separation (n^2, expensive)."""
        if not self._pd or self.get_particle_count() < 2: return 0.0, 0.0
        # this is very expensive. only call if enabled.
        if not self._get_plot_setting(self._graph_settings, 'plot_min_max_separation'):
            return -1.0, -1.0 # indicate not calculated

        pos = self._pd.get("positions", "cpu").astype(np.float64)
        N = pos.shape[0]
        min_dist_sq = np.inf
        max_dist_sq = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dr = pos[i] - pos[j]
                dist_sq = np.dot(dr, dr)
                min_dist_sq = min(min_dist_sq, dist_sq)
                max_dist_sq = max(max_dist_sq, dist_sq)
        min_sep = np.sqrt(min_dist_sq) if min_dist_sq != np.inf else 0.0
        max_sep = np.sqrt(max_dist_sq)
        return float(min_sep), float(max_sep)


    def _calculate_momentum(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """calculates total linear momentum, com position, and com velocity."""
        zeros = np.zeros(3, dtype=np.float64)
        if not self._pd or self.get_particle_count() == 0: return zeros, zeros, zeros
        try:
            self._pd.ensure("positions velocities masses", "cpu")
            pos = self._pd.get("positions", "cpu").astype(np.float64)
            vel = self._pd.get("velocities", "cpu").astype(np.float64)
            mass = self._pd.get("masses", "cpu").astype(np.float64)
            total_mass = np.sum(mass)
            if total_mass <= 1e-30: return zeros, zeros, zeros
            # calculations use float64 for precision
            total_momentum = np.sum(mass[:, np.newaxis] * vel, axis=0)
            com_pos = np.sum(mass[:, np.newaxis] * pos, axis=0) / total_mass
            com_vel = np.sum(mass[:, np.newaxis] * vel, axis=0) / total_mass
            return total_momentum, com_pos, com_vel
        except Exception as e: print(f"Warn: Error calculating momentum/CoM: {e}"); return zeros, zeros, zeros


    def _calculate_angular_momentum(self, com_pos: np.ndarray) -> np.ndarray:
        """calculates total angular momentum relative to the center of mass (com)."""
        zeros = np.zeros(3, dtype=np.float64)
        if not self._pd or self.get_particle_count() == 0: return zeros
        try:
            # assumes caller (collect_graph_data) ensured cpu data
            pos = self._pd.get("positions", "cpu").astype(np.float64)
            vel = self._pd.get("velocities", "cpu").astype(np.float64)
            mass = self._pd.get("masses", "cpu").astype(np.float64)
            # position relative to com
            rel_pos = pos - com_pos
            # linear momentum p = m*v
            mom = mass[:, np.newaxis] * vel
            # angular momentum l = sum(r_rel x p)
            angular_momenta = np.cross(rel_pos, mom)
            return np.sum(angular_momenta, axis=0)
        except Exception as e: print(f"Warn: Error calculating angular momentum: {e}"); return zeros

    def _calculate_min_max_separation(self) -> tuple[Optional[float], Optional[float]]:
        """calculates min and max inter-particle separation. expensive o(n^2)."""
        N = self.get_particle_count()
        if N < 2 or not self._pd: return None, None
        try:
            pos = self._pd.get("positions", "cpu")
            if N > 5000: # limit for pdist to avoid excessive memory/time
                print(f"Warn: N={N} too large for min/max separation plot. Skipping.")
                return None, None
            distances = pdist(pos) # condensed distance matrix
            if distances.size == 0: return None, None
            return np.min(distances), np.max(distances)
        except Exception as e:
            print(f"Warn: Error calculating min/max separation: {e}")
            return None, None

    def collect_graph_data(self, force_collect: bool = False):
         """calculates and stores snapshot data for enabled graphs."""
         if self._graph_data is None: return # plotting disabled
         if self._ended and not force_collect: return # don't collect after end unless forced
         if not self._pd: print("Error: collect_graph_data called before ParticleData setup."); return
         if self.get_particle_count() == 0: return # skip if no particles

         # check interval unless forcing collection (e.g., at t=0 or end)
         if not force_collect and (self._steps_taken == 0 or self._steps_taken % self._graph_log_interval_steps != 0):
              return

         gs = self._graph_settings # active graph settings
         gd = self._graph_data    # data dictionary to append to

         try:
             # always log time and step
             gd['time'].append(self._time)
             gd['step'].append(self._steps_taken)

             # ensure data is on cpu for most calculations
             # list all potentially needed arrays here for one ensure call
             all_needed_stats = {"masses", "velocities", "internal_energies", "positions",
                                 "densities", "temperatures", "pressures",
                                 "work_terms", "visc_heating_terms", "cooling_rates", "fusion_rates"}
             self._pd.ensure(list(all_needed_stats), "cpu")

             # calculate & store energies
             energy_comps = self._get_energy_components() # {"ke":..., "u":..., "pe":...}
             ke_tot = energy_comps.get("KE", 0.0)
             u_tot = energy_comps.get("U", 0.0)
             pe_tot = energy_comps.get("PE", 0.0)
             e_tot = ke_tot + u_tot + pe_tot
             if self._get_plot_setting(gs, 'plot_total_energy'): gd['total_energy'].append(e_tot)
             if self._get_plot_setting(gs, 'plot_energy_components'):
                 gd['total_ke'].append(ke_tot); gd['total_pe'].append(pe_tot); gd['total_u'].append(u_tot)

             # calculate & store momenta & com
             lin_mom, com_pos, com_vel = self._calculate_momentum()
             ang_mom = self._calculate_angular_momentum(com_pos)
             if self._get_plot_setting(gs, 'plot_linear_momentum'):
                 gd['total_px'].append(lin_mom[0]); gd['total_py'].append(lin_mom[1]); gd['total_pz'].append(lin_mom[2])
             if self._get_plot_setting(gs, 'plot_angular_momentum'):
                 gd['total_lx'].append(ang_mom[0]); gd['total_ly'].append(ang_mom[1]); gd['total_lz'].append(ang_mom[2])
             if self._get_plot_setting(gs, 'plot_com_velocity'):
                 gd['com_vx'].append(com_vel[0]); gd['com_vy'].append(com_vel[1]); gd['com_vz'].append(com_vel[2])
             if self._get_plot_setting(gs, 'plot_com_position'): # new
                 gd['com_x'].append(com_pos[0]); gd['com_y'].append(com_pos[1]); gd['com_z'].append(com_pos[2])


             # calculate & store other stats
             mass_np = self._pd.get("masses", "cpu")
             total_mass = np.sum(mass_np)
             if total_mass <= 1e-30: total_mass = 1.0 # avoid div by zero for averages

             densities_np = self._pd.get("densities", "cpu")
             temps_np = self._pd.get("temperatures", "cpu")
             pressures_np = self._pd.get("pressures", "cpu")
             internal_energies_np = self._pd.get("internal_energies", "cpu")

             if self._get_plot_setting(gs, 'plot_max_rho'): gd['max_rho'].append(np.max(densities_np) if densities_np.size > 0 else 0)
             if self._get_plot_setting(gs, 'plot_avg_density'): gd['avg_rho'].append(np.mean(densities_np) if densities_np.size > 0 else 0) # new
             if self._get_plot_setting(gs, 'plot_max_temp'): gd['max_temp'].append(np.max(temps_np) if temps_np.size > 0 else 0)
             if self._get_plot_setting(gs, 'plot_avg_temp'):
                 avg_temp_val = np.sum(mass_np * temps_np) / total_mass if temps_np.size > 0 else 0.0
                 gd['avg_temp'].append(avg_temp_val)
             if self._get_plot_setting(gs, 'plot_max_pressure'): gd['max_pressure'].append(np.max(pressures_np) if pressures_np.size > 0 else 0) # new
             if self._get_plot_setting(gs, 'plot_avg_pressure'): gd['avg_pressure'].append(np.mean(pressures_np) if pressures_np.size > 0 else 0) # new

             if self._get_plot_setting(gs, 'plot_virial_ratio'):
                 virial = (2.0 * ke_tot / abs(pe_tot)) if abs(pe_tot) > 1e-30 else 0.0
                 gd['virial_ratio'].append(virial)

             if self._get_plot_setting(gs, 'plot_min_max_separation'): # new
                 min_sep, max_sep = self._calculate_min_max_separation()
                 gd['min_separation'].append(min_sep if min_sep is not None else np.nan)
                 gd['max_separation'].append(max_sep if max_sep is not None else np.nan)

             # calculate & store energy rates
             if self._get_plot_setting(gs, 'plot_energy_rates') or self._get_plot_setting(gs, 'plot_sph_visc_heat_rate'):
                cr = self._pd.get("cooling_rates", "cpu")
                fr = self._pd.get("fusion_rates", "cpu")
                wr = self._pd.get("work_terms", "cpu")
                vr = self._pd.get("visc_heating_terms", "cpu") # new for its own plot
                if self._get_plot_setting(gs, 'plot_energy_rates'):
                    gd['total_cooling_rate'].append(np.sum(cr * mass_np))
                    gd['total_fusion_rate'].append(np.sum(fr * mass_np))
                    gd['total_sph_work_rate'].append(np.sum(wr * mass_np))
                if self._get_plot_setting(gs, 'plot_sph_visc_heat_rate'): # new
                    gd['total_visc_heat_rate'].append(np.sum(vr * mass_np))


             # thermo/fusion particle counts & averages
             if self._get_plot_setting(gs, 'plot_num_fusing_particles'): # new
                 num_fusing = np.sum(temps_np >= self._config.get('fusion_thresh', 1e10)) if temps_np.size > 0 else 0
                 gd['num_fusing_particles'].append(num_fusing)
             if self._get_plot_setting(gs, 'plot_avg_spec_int_energy'): # new
                 avg_spec_u_val = np.sum(internal_energies_np) / total_mass if internal_energies_np.size > 0 else 0.0
                 gd['avg_spec_u'].append(avg_spec_u_val)


             # calculate & store energy drift
             if self._get_plot_setting(gs, 'plot_energy_drift'):
                drift_percent = 0.0
                initial_e_graph = self._graph_initial_total_energy # baseline set at init/restart
                if initial_e_graph is not None and abs(initial_e_graph) > 1e-25:
                    # actual change since start = current e - initial e
                    delta_e_actual_total = e_tot - initial_e_graph
                    # expected change since start (accumulated in advance_one_step)
                    # drift = actual change - expected change (from non-conservative sources)
                    energy_drift_cumulative = delta_e_actual_total - self._graph_accumulated_expected_delta_e
                    drift_percent = (energy_drift_cumulative / abs(initial_e_graph)) * 100.0
                elif self._steps_taken > 0: print("Warn (Graph): Initial E not set for drift calc.")
                gd['energy_drift_percent'].append(drift_percent)

             # bh tree nodes
             if self._get_plot_setting(gs, 'plot_bh_nodes'): # new
                 gd['bh_num_nodes'].append(self._bh_node_count_for_plot)

             # step timing
             if self._get_plot_setting(gs, 'plot_step_timing'): # new
                 gd['step_duration_ms'].append(self._last_step_duration * 1000.0)


         except KeyError as e: print(f"ERROR graph collect step {self._steps_taken}: Missing key '{e}' for plotting.")
         except Exception as e: print(f"ERROR graph collect step {self._steps_taken}: {e}"); traceback.print_exc()


    def _collect_final_snapshot_data(self):
        """calculates and stores data needed for final state plots (histograms, etc.)."""
        if not self._pd or self.get_particle_count() == 0:
            if self._graph_data: self._graph_data['final_snapshot'] = {}
            return

        print("Collecting final snapshot data for plots...")
        snapshot = {}
        N = self.get_particle_count()
        gs = self._graph_settings
        pd = self._pd

        try:
            # determine needed arrays based on which final plots are enabled
            needed = set()
            if self._get_plot_setting(gs, 'plot_hist_speed'): needed.add("velocities")
            if self._get_plot_setting(gs, 'plot_hist_temp'): needed.add("temperatures")
            if self._get_plot_setting(gs, 'plot_hist_density'): needed.add("densities")
            if self._get_plot_setting(gs, 'plot_profile_temp'): needed.add("temperatures"); needed.add("positions"); needed.add("masses")
            if self._get_plot_setting(gs, 'plot_profile_density'): needed.add("densities"); needed.add("positions"); needed.add("masses")
            if self._get_plot_setting(gs, 'plot_phase_T_rho'): needed.add("temperatures"); needed.add("densities")

            # new final snapshots
            if self._get_plot_setting(gs, 'plot_hist_spec_int_energy'): needed.add("internal_energies"); needed.add("masses")
            if self._get_plot_setting(gs, 'plot_profile_pressure'): needed.add("pressures"); needed.add("positions"); needed.add("masses")
            if self._get_plot_setting(gs, 'plot_profile_spec_int_energy'): needed.add("internal_energies"); needed.add("masses"); needed.add("positions")
            if self._get_plot_setting(gs, 'plot_phase_P_rho'): needed.add("pressures"); needed.add("densities")
            if self._get_plot_setting(gs, 'plot_scatter_cool_temp'): needed.add("cooling_rates"); needed.add("temperatures")
            if self._get_plot_setting(gs, 'plot_scatter_fus_temp'): needed.add("fusion_rates"); needed.add("temperatures")
            if self._get_plot_setting(gs, 'plot_hist_sph_neighbors'): pass # placeholder, data would be calculated elsewhere

            if not needed:
                if self._graph_data: self._graph_data['final_snapshot'] = {}
                return # nothing to collect

            pd.ensure(list(needed), "cpu") # ensure all needed data is on cpu

            # calculate derived quantities for snapshot
            if "velocities" in needed: snapshot['speeds'] = np.linalg.norm(pd.get("velocities", "cpu"), axis=1)
            if "temperatures" in needed: snapshot['temperatures'] = pd.get("temperatures", "cpu").copy()
            if "densities" in needed: snapshot['densities'] = pd.get("densities", "cpu").copy()
            if "pressures" in needed: snapshot['pressures'] = pd.get("pressures", "cpu").copy() # new
            if "cooling_rates" in needed: snapshot['cooling_rates'] = pd.get("cooling_rates", "cpu").copy() # new
            if "fusion_rates" in needed: snapshot['fusion_rates'] = pd.get("fusion_rates", "cpu").copy() # new

            if "internal_energies" in needed and "masses" in needed: # new for specific internal energy
                u_total = pd.get("internal_energies", "cpu")
                m = pd.get("masses", "cpu")
                m_safe = np.maximum(m, 1e-30)
                snapshot['specific_internal_energies'] = u_total / m_safe

            # calculate radii relative to com if needed for profiles
            if "positions" in needed and "masses" in needed and \
               (self._get_plot_setting(gs, 'plot_profile_density') or \
                self._get_plot_setting(gs, 'plot_profile_temp') or \
                self._get_plot_setting(gs, 'plot_profile_pressure') or \
                self._get_plot_setting(gs, 'plot_profile_spec_int_energy')):
                pos_np = pd.get("positions", "cpu"); mass_np = pd.get("masses", "cpu")
                total_mass = np.sum(mass_np)
                if total_mass > 1e-30:
                    com_pos = np.sum(mass_np[:, np.newaxis] * pos_np, axis=0) / total_mass
                    snapshot['radii'] = np.linalg.norm(pos_np - com_pos, axis=1)
                else: snapshot['radii'] = np.zeros(N, dtype=pos_np.dtype) # fallback if no mass
            else: # still provide radii if only one of the conditions met for other plots
                if "positions" in needed and ('radii' not in snapshot): # if not already calculated
                    snapshot['radii'] = np.linalg.norm(pd.get("positions", "cpu"), axis=1) # radii from origin if com not calc

            # placeholder for sph neighbor data
            if self._get_plot_setting(gs, 'plot_hist_sph_neighbors'):
                snapshot['sph_neighbor_counts'] = np.array([]) # empty for now

            # store the collected snapshot data
            if self._graph_data: self._graph_data['final_snapshot'] = snapshot
            print("Final snapshot data collected.")
        except Exception as e:
            print(f"ERROR collecting final snapshot data: {e}"); traceback.print_exc()
            if self._graph_data: self._graph_data['final_snapshot'] = {} # ensure empty dict on error


    def _ensure_thermo_params(self):
     """ensures simulation 'cv' and 'gas_constant_sim' are in config, calculating if needed."""
     # ensure cv (simulation specific heat capacity)
     if 'cv' not in self._config:
         gas_const = self._config.get('gas_constant_sim')
         gamma = self._config.get('sph_eos_gamma')
         if gas_const is not None and gamma is not None and abs(gamma - 1.0) > 1e-9:
             self._config['cv'] = float(gas_const) / (float(gamma) - 1.0)
             print(f"  Derived Sim Cv = {self._config['cv']:.3e} from gas_const_sim and gamma")
         else: raise ValueError("Missing 'cv' and cannot derive it (needs 'gas_constant_sim', 'sph_eos_gamma'!=1)")

     # ensure gas_constant_sim (simulation gas constant r_sim) - less likely to be missing if cv derived
     if 'gas_constant_sim' not in self._config:
         print("Warning: 'gas_constant_sim' missing from config. EOS models might fail.")

     # ensure density floor consistency
     df = self._config.get('density_floor')
     sph_df = self._config.get('sph_density_floor')
     if df is None and sph_df is not None: self._config['density_floor'] = sph_df
     elif sph_df is None and df is not None: self._config['sph_density_floor'] = df
     elif df is None and sph_df is None: raise ValueError("Missing density_floor/sph_density_floor")

    def _get_energy_components(self, force_pe_for_logging: bool = False) -> Dict[str, float]:
        """computes ke, u, pe, returning results as float64. includes 'total'.
           pe is calculated conditionally.
        """
        default_energies = {"KE": 0.0, "U": 0.0, "PE": 0.0, "Total": 0.0}
        if not self._pd or self.get_particle_count() == 0: return default_energies

        target_dtype = np.float64 # use high precision for energy calculations
        try:
            needed = {"velocities", "masses", "internal_energies"}
            G = self._config.get('G', 0.0)
            # determine if pe needs to be calculated
            calculate_pe_actual = False
            if G != 0.0:
                plot_e_comps = self._config.get('GRAPH_SETTINGS', {}).get('plot_energy_components', False)
                plot_v_ratio = self._config.get('GRAPH_SETTINGS', {}).get('plot_virial_ratio', False)
                if plot_e_comps or plot_v_ratio or force_pe_for_logging:
                    calculate_pe_actual = True
            if calculate_pe_actual:
                needed.add("positions")

            self._pd.ensure(list(needed), "cpu")

            vel = self._pd.get("velocities", "cpu").astype(target_dtype, copy=False)
            mass = self._pd.get("masses", "cpu").astype(target_dtype, copy=False)
            u_tot_per_particle = self._pd.get("internal_energies", "cpu").astype(target_dtype, copy=False)

            ke_total = 0.5 * np.sum(mass * np.sum(vel * vel, axis=1))
            u_total = np.sum(u_tot_per_particle)
            pe_total = 0.0

            if calculate_pe_actual:
                gravity_model = self._physics_manager._active_models.get("gravity") if self._physics_manager else None
                can_calc_pe_model = gravity_model and hasattr(gravity_model, 'compute_potential_energy')
                pe_calculated_by_model = False
                if can_calc_pe_model:
                    try:
                        pe_total = float(gravity_model.compute_potential_energy(self._pd))
                        pe_calculated_by_model = True
                    except NotImplementedError: pass
                    except Exception as e_pe: print(f"Warn: PE calc via model failed: {e_pe}")

                if not pe_calculated_by_model:
                    N_local = self.get_particle_count()
                    if N_local < 5000: # fallback to n^2 for small n
                        pos = self._pd.get("positions", "cpu").astype(target_dtype, copy=False)
                        soft_sq = float(self._config.get('softening', 0.1))**2
                        for i in range(N_local):
                            for j in range(i + 1, N_local):
                                dr = pos[i] - pos[j]; dist_sq = np.dot(dr, dr)
                                dist_soft = np.sqrt(dist_sq + soft_sq)
                                if dist_soft > 1e-15: pe_total -= G * mass[i] * mass[j] / dist_soft

            total_energy = ke_total + u_total + pe_total
            return {"KE": float(ke_total), "U": float(u_total), "PE": float(pe_total), "Total": float(total_energy)}
        except Exception as e:
            print(f"Error calculating energy components: {e}"); traceback.print_exc()
            return default_energies.copy() # return a copy to avoid modifying class default


    def _log_energy_and_calculate_drift(self, accumulated_non_grav_delta_e: float,
                                        initial_total_energy: Optional[float],
                                        last_logged_total_energy: Optional[float]
                                        ) -> tuple[Optional[float], Optional[float], float]:
        """computes total energy, calculates drift vs non-conservative work. no printing."""
        if not self._pd or self.get_particle_count() == 0:
            return initial_total_energy, last_logged_total_energy, 0.0

        try:
            current_energy = self._get_energy_components()
            current_total_energy = current_energy.get("Total", 0.0)
            energy_drift_cumulative = 0.0

            if initial_total_energy is None: # first call
                initial_total_energy = current_total_energy
                last_logged_total_energy = current_total_energy
            else:
                delta_e_actual_total = current_total_energy - initial_total_energy
                energy_drift_cumulative = delta_e_actual_total - accumulated_non_grav_delta_e
                last_logged_total_energy = current_total_energy

            return initial_total_energy, last_logged_total_energy, energy_drift_cumulative
        except Exception as e:
             print(f"ERROR calculating/logging energy (step {self._steps_taken}): {e}"); traceback.print_exc()
             return initial_total_energy, last_logged_total_energy, 0.0

    def _initialize_particles(self):
        """sets initial particle positions, velocities, masses, and internal energy."""
        if not self._pd or not self._config: return
        N = self._pd.get_n()
        if N == 0: return

        # get parameters from config
        radius = float(self._config['radius']); v_scale = float(self._config['v_scale'])
        m_min = float(self._config['mass_min']); m_max = float(self._config['mass_max'])
        temp_init = float(self._config['initial_temp']) # in kelvin
        cv_sim = float(self._config['cv']) # simulation specific heat capacity
        min_temp = float(self._config['min_temperature']) # for energy floor
        rotation_strength = float(self._config.get('rotation_strength', 0.0))

        print(f"  Initializing {N} particles: R={radius:.2f}, Vsc={v_scale:.2f}, M=[{m_min:.2f},{m_max:.2f}], T={temp_init:.1f}K, Cv={cv_sim:.2e}, RotStr={rotation_strength:.2f}")

        rng = np.random.default_rng() # modern random number generator

        # positions (random sphere)
        r = radius * np.cbrt(rng.uniform(0.0, 1.0, size=N))
        theta = np.arccos(2.0 * rng.uniform(0.0, 1.0, size=N) - 1.0)
        phi = 2.0 * np.pi * rng.uniform(0.0, 1.0, size=N)
        pos = np.zeros((N, 3), dtype=self._pd.get_dtype('positions'))
        pos[:, 0] = r * np.sin(theta) * np.cos(phi) # x
        pos[:, 1] = r * np.sin(theta) * np.sin(phi) # y
        pos[:, 2] = r * np.cos(theta)               # z
        self._pd.set("positions", pos)

        # velocities (random gaussian, scaled, com corrected)
        vel = rng.standard_normal(size=(N, 3), dtype=self._pd.get_dtype('velocities')) * v_scale
        vel -= np.mean(vel, axis=0) # ensure com velocity is zero initially for the random part

        # add rotational velocity
        if abs(rotation_strength) > 1e-9:
            print(f"    Adding initial rotation (strength = {rotation_strength:.2f}) around z-axis.")
            v_rot = np.zeros_like(vel)
            v_rot[:, 0] = -rotation_strength * pos[:, 1]
            v_rot[:, 1] =  rotation_strength * pos[:, 0]
            vel += v_rot
            print(f"    Max rotational speed added: {np.max(np.linalg.norm(v_rot, axis=1)):.2f}")

        self._pd.set("velocities", vel)

        # masses (uniform random)
        mass = rng.uniform(m_min, m_max, size=N).astype(self._pd.get_dtype('masses'))
        self._pd.set("masses", mass)

        # internal energy (u = mass * cv_sim * t)
        u_tot_initial = mass * cv_sim * temp_init
        min_u_floor = mass * cv_sim * min_temp
        u_tot_initial = np.maximum(min_u_floor, u_tot_initial)
        u_tot_initial = np.maximum(1e-50, u_tot_initial)
        self._pd.set("internal_energies", u_tot_initial.astype(self._pd.get_dtype('internal_energies')))
        print(f"  Set Initial U_total: Min={np.min(u_tot_initial):.3e}, Max={np.max(u_tot_initial):.3e}")

        # zero out other relevant fields
        for name in ["forces", "accelerations", "densities", "pressures",
                     "temperatures", "work_terms", "cooling_rates", "fusion_rates",
                     "visc_heating_terms", "forces_grav", "v_half_temp",
                     "work_terms_prev", "visc_heating_terms_prev", "cooling_rates_prev", "fusion_rates_prev"]:
            arr_ref = self._pd.get(name, "cpu", writeable=True)
            arr_ref.fill(0.0)
            self._pd.release_writeable(name)
        # set default colors
        colors_ref = self._pd.get("colors", "cpu", writeable=True)
        colors_ref.fill(0.5) # mid-gray
        self._pd.release_writeable("colors")

        # ensure all initial data location is marked as cpu
        for name in self._pd.get_attribute_names(): self._pd._location[name] = "cpu"


    def _select_initial_models_and_integrator(self):
         """selects initial models and integrator based on config defaults."""
         if not self._physics_manager or not self._integrator_manager: raise RuntimeError("Managers not initialized")
         print("  Selecting initial models & integrator from config defaults...")
         try:
             self._physics_manager.select_model("gravity", self._config['default_gravity_model'])
             self._physics_manager.select_model("sph", self._config['default_sph_model'])
             self._physics_manager.select_model("thermo", self._config['default_thermo_model'])
             self._physics_manager.select_model("color", self._config['default_color_model'])
             self._integrator_manager.select_integrator(self._config['default_integrator'])
         except KeyError as e: raise ValueError(f"Missing default model/integrator key in config: {e}")
         except Exception as e: raise RuntimeError(f"Error selecting initial models/integrator: {e}") from e

    def _perform_initial_calculation(self):
         """computes initial density, eos, and forces based on the initial particle state."""
         if not self._physics_manager or not self._pd: return
         print("  Calculating initial density, EOS, forces...")
         self._physics_manager.compute_all_physics(compute_forces=True, compute_thermo=True, compute_color=True)

    def advance_one_step(self) -> bool:
        """advances the simulation by one time step dt."""
        if self._ended or not self._pd or not self._integrator_manager: return False

        step_start_time = time.perf_counter() # for step timing plot
        try:
            self._integrator_manager.advance(self._dt, current_step=self._steps_taken, debug_particle_idx=-1)

            # update time/step & check end conditions
            self._time += self._dt
            self._steps_taken += 1
            self._status_msg = STATUS_RUNNING

            # accumulate expected energy change for graphing
            if self._graph_initial_total_energy is not None:
                try:
                    rate_keys = "masses work_terms visc_heating_terms cooling_rates fusion_rates"
                    self._pd.ensure(rate_keys, "cpu")
                    m = self._pd.get("masses", "cpu")
                    wr = self._pd.get("work_terms", "cpu"); vr = self._pd.get("visc_heating_terms", "cpu")
                    cr = self._pd.get("cooling_rates", "cpu"); fr = self._pd.get("fusion_rates", "cpu")
                    rate_total_specific = (np.atleast_1d(wr) + np.atleast_1d(vr) + np.atleast_1d(fr) - np.atleast_1d(cr))
                    m_atleast1d = np.atleast_1d(m)
                    dE_dt_total = np.sum(rate_total_specific * m_atleast1d)
                    self._graph_accumulated_expected_delta_e += dE_dt_total * self._dt
                except Exception as e_acc: print(f"Warn: Error accumulating graph energy delta: {e_acc}")

            # update bh node count for graphing (if model is active)
            if self._physics_manager and self._physics_manager._active_model_ids.get("gravity") == "gravity_bh_numba":
                active_gravity_model = self._physics_manager._active_models.get("gravity")
                if active_gravity_model and hasattr(active_gravity_model, 'n_nodes_status'):
                    self._bh_node_count_for_plot = abs(active_gravity_model.n_nodes_status)
                else:
                    self._bh_node_count_for_plot = 0
            else:
                self._bh_node_count_for_plot = 0


            self._check_end_conditions()
            self._last_step_duration = time.perf_counter() - step_start_time # store duration
            return True

        except Exception as e:
            print(f"ERROR during simulation step {self._steps_taken}: {e}"); traceback.print_exc()
            self._status_msg = STATUS_ERROR; self._ended = True; self._running = False
            self._last_step_duration = time.perf_counter() - step_start_time # store duration even on error
            return False


    def _recompute_forces_and_physics(self, debug_particle_idx: int = -1):
        """
        callback for integrator: calls physicsmanager to compute forces/rates.

        args:
            debug_particle_idx: index for detailed debugging in physics calls.
        """
        if self._physics_manager and not self._ended:
             try:
                 self._physics_manager.compute_all_physics(
                     compute_forces=True,
                     compute_thermo=True,
                     compute_color=True,
                     debug_particle_idx=debug_particle_idx
                 )
             except Exception as e:
                  print(f"ERROR during physics calculation callback: {e}")
                  traceback.print_exc()
                  self._status_msg = STATUS_ERROR; self._ended = True # stop simulation

    # simulation control methods
    def run(self):
        """starts or resumes the simulation execution."""
        if not self._ended: self._running = True; self._status_msg = STATUS_RUNNING
        else: print("Sim Info: Cannot run, simulation has ended.")
    def pause(self):
        """pauses the simulation execution."""
        if not self._ended: self._running = False; self._status_msg = STATUS_PAUSED
    def toggle_run(self):
        """toggles between running and paused states."""
        if not self._ended: self._running = not self._running; self._status_msg = STATUS_RUNNING if self._running else STATUS_PAUSED
    def step_forward(self):
        """advances simulation by one step if paused."""
        if not self._running and not self._ended: self.advance_one_step(); self._status_msg = STATUS_PAUSED
        elif self._running: print("Sim Info: Cannot step forward while running.")
        else: print("Sim Info: Cannot step forward, simulation ended.")

    def reset_to_initial(self):
        """Resets the simulation to t=0 using the original configuration."""
        print("===== Resetting Simulation to Initial State =====")
        if not self._pd or not self._physics_manager or not self._integrator_manager:
            print("ERROR: Cannot reset, core components missing.")
            self._status_msg = STATUS_ERROR; self._ended = True; return

        self._status_msg = STATUS_RESETTING
        self._running = False; self._ended = False
        self._time = 0.0; self._steps_taken = 0
        self._graph_initial_total_energy = None; self._graph_accumulated_expected_delta_e = 0.0

        self._initialize_particles()
        self._physics_manager.setup(self._pd) # re-setup with current config
        self._integrator_manager.setup(self._pd) # re-setup with current config
        self._select_initial_models_and_integrator() # re-select based on current config
        self._perform_initial_calculation()

        force_pe_for_graph_baseline = self._config.get('G', 0.0) != 0.0
        energy_components_for_graph_baseline = self._get_energy_components(force_pe_for_logging=force_pe_for_graph_baseline)
        self._graph_initial_total_energy = energy_components_for_graph_baseline.get("Total")
        self._graph_accumulated_expected_delta_e = 0.0

        self._initialize_graph_data_lists() # reset graph data
        self.collect_graph_data(force_collect=True) # collect t=0 data
        self._status_msg = STATUS_READY
        print("===== Simulation Reset Complete =====")



    def restart(self, new_settings: Dict[str, Any],
                 gravity_id: str, sph_id: str, thermo_id: str, color_id: str, integrator_id: str):
        """Restarts simulation with new settings, potentially reallocating data."""
        new_settings['_effective_np_float_type'] = self._effective_np_float_type
        new_settings['_effective_ti_float_type'] = self._effective_ti_float_type
        new_settings['_effective_ti_int_type'] = self._effective_ti_int_type
        new_settings['USE_DOUBLE_PRECISION'] = (self._effective_np_float_type == np.float64)

        restart_start_time = time.perf_counter()
        print("\n===== Restarting Simulation with New Configuration =====")
        self._status_msg = STATUS_RESTARTING
        self._running = False; self._ended = False
        self._time = 0.0; self._steps_taken = 0
        self._graph_initial_total_energy = None; self._graph_accumulated_expected_delta_e = 0.0
        pre_initialized_taichi_models = {}

        try:
            old_N = self.get_particle_count() if self._pd else 0
            new_N = int(new_settings.get('N', old_N))
            if new_N <= 0: raise ValueError(f"Restart N must be > 0, got {new_N}")
            needs_new_pd = (new_N != old_N) or (self._pd is None)

            self._configure(new_settings)

            if needs_new_pd:
                 if self._pd: self._pd.cleanup_gpu_resources()
                 if self._physics_manager: self._physics_manager.cleanup_models()
                 response_q = queue.Queue(maxsize=1)
                 task = {'type': 'create_pd_for_restart',
                         'params': {'N': new_N, 'numpy_precision': self._effective_np_float_type,
                                    'taichi_is_active': self._backend_flags.get("taichi", False),
                                    'initial_config': self._config},
                         'response_queue': response_q}
                 self._main_thread_task_queue.put(task)
                 try:
                     alloc_result = response_q.get(timeout=20.0)
                     if not alloc_result['success']: raise RuntimeError(f"Main thread task failed: {alloc_result['error']}")
                     result_data = alloc_result['data']
                     self._pd = result_data['particle_data']
                     pre_initialized_taichi_models = result_data.get('initialized_models', {})
                     if not isinstance(self._pd, ParticleData): raise TypeError("Main thread returned invalid PD.")
                 except queue.Empty: raise TimeoutError("Timeout waiting for main thread PD/Model setup.")
                 except Exception as e_wait: raise RuntimeError("Error during PD/Model setup request.") from e_wait

                 self._physics_manager = PhysicsManager(self._pd, self._config, self._backend_flags)
                 self._integrator_manager = IntegratorManager(self._pd, self._recompute_forces_and_physics, self._config)
            else:
                 if self._physics_manager: self._physics_manager.update_config(self._config)
                 if self._integrator_manager: self._integrator_manager.update_config(self._config)

            self._initialize_particles()
            if not self._physics_manager: raise RuntimeError("PhysicsManager missing.")
            self._physics_manager.set_pre_initialized_models(pre_initialized_taichi_models)
            self._physics_manager.select_model("gravity", gravity_id)
            self._physics_manager.select_model("sph", sph_id)
            self._physics_manager.select_model("thermo", thermo_id)
            self._physics_manager.select_model("color", color_id)
            if not self._integrator_manager: raise RuntimeError("IntegratorManager missing.")
            self._integrator_manager.select_integrator(integrator_id)

            self._perform_initial_calculation()
            force_pe_for_graph_baseline = self._config.get('G', 0.0) != 0.0
            energy_components_for_graph_baseline = self._get_energy_components(force_pe_for_logging=force_pe_for_graph_baseline)
            self._graph_initial_total_energy = energy_components_for_graph_baseline.get("Total")
            self._graph_accumulated_expected_delta_e = 0.0
            self.collect_graph_data(force_collect=True)
            self._status_msg = STATUS_READY
            restart_duration = time.perf_counter() - restart_start_time
            print(f"===== Simulation Restart Complete ({restart_duration:.3f} s) =====")
            self._log_current_setup()
        except Exception as e:
             self._status_msg = STATUS_ERROR; self._ended = True
             print(f"\n!!! FATAL ERROR during Simulator Restart: {e} !!!"); traceback.print_exc()
             raise

    def set_live_parameter(self, key: str, value: Any):
        """Updates a live-updatable simulation parameter."""
        if key not in self._config: print(f"Warn: Param '{key}' not found in config."); return
        try:
            current_val = self._config.get(key)
            if current_val is not None: value = type(current_val)(value) # cast to original type
        except Exception as e: print(f"Warn: Could not cast value for '{key}': {e}")
        self._config[key] = value
        if self._physics_manager: self._physics_manager.update_config({key: value})


    def set_thermo_flag(self, key: str, value: bool):
         """Updates a boolean thermodynamics flag (e.g., use_cooling)."""
         valid = ['use_rad_press', 'use_cooling', 'use_fusion']
         if key not in valid: print(f"Warn: Invalid thermo flag '{key}'. Valid: {valid}"); return
         new_val = bool(value);
         self._config[key] = new_val
         if self._physics_manager: self._physics_manager.update_config({key: new_val})


    def select_model(self, model_type: str, model_id: str):
         """Selects a physics model, triggering physics recalculation."""
         if self._physics_manager:
             try:
                  print(f"Simulator: Selecting model type='{model_type}', id='{model_id}'")
                  self._physics_manager.select_model(model_type, model_id)
                  # recompute physics state after model change for consistency
                  self._perform_initial_calculation()
                  print(f"  Model selection successful, physics recalculated.")
             except Exception as e: print(f"ERROR selecting model '{model_id}': {e}"); self._status_msg = STATUS_ERROR

    def select_integrator(self, integrator_id: str):
        """Selects the time integrator."""
        if self._integrator_manager:
             try:
                 print(f"Simulator: Selecting integrator id='{integrator_id}'")
                 self._integrator_manager.select_integrator(integrator_id)
                 print(f"  Integrator selection successful.")
             except Exception as e: print(f"ERROR selecting integrator '{integrator_id}': {e}"); self._status_msg = STATUS_ERROR

    # state query methods
    def is_running(self) -> bool: return self._running
    def is_ended(self) -> bool: return self._ended
    def get_time(self) -> float: return self._time
    def get_dt(self) -> float: return self._dt
    def get_particle_count(self) -> int: return self._config.get('N', 0) if self._config else 0
    def get_steps_taken(self) -> int: return self._steps_taken
    def get_status_message(self) -> str: return self._status_msg

    def get_current_models_info(self) -> Dict[str, Optional[Dict]]:
         """Gets info dicts for currently active models."""
         return self._physics_manager.get_active_models_info() if self._physics_manager else {}

    def get_current_integrator_info(self) -> Optional[Dict]:
          """Gets info dict for the currently active integrator."""
          return self._integrator_manager.get_active_integrator_info() if self._integrator_manager else None

    def get_current_state_for_ui(self) -> Dict[str, Any]:
        """Gathers essential simulation state for ui updates."""
        default_state = { # structure matching ui expectations
            "time": self._time, "steps_taken": self._steps_taken, "status_msg": self._status_msg,
            "running": self._running, "ended": self._ended, "N": 0,
            "positions": [], "colors": [], "stats": {},
            "current_models": {}, "current_integrator": "-", "graph_settings": {} }
        if not self._pd or not self._physics_manager or not self._integrator_manager:
            default_state["status_msg"] = self._status_msg or STATUS_ERROR
            return default_state

        # gather current state
        state = default_state.copy()
        state["N"] = self.get_particle_count()
        state["graph_settings"] = self._config.get('GRAPH_SETTINGS', {})
        # get particle data formatted for ui (positions, colors)
        particle_ui_data = self._pd.get_state_for_ui(precision='f32')
        state["positions"] = particle_ui_data.get('positions', [])
        state["colors"] = particle_ui_data.get('colors', [])
        # compute diagnostic stats
        state["stats"] = self._compute_stats()
        # get info about active models/integrator
        model_info = self.get_current_models_info()
        state["current_models"] = { mtype: (info['id'] if info else '-') for mtype, info in model_info.items() }
        integrator_info = self.get_current_integrator_info()
        state["current_integrator"] = integrator_info['id'] if integrator_info else '-'
        return state

    def _compute_stats(self) -> Dict[str, float]:
        stats = {"avg_KE": 0.0, "avg_vel": 0.0, "core_c": 0, "max_rho": 0.0,
                 "max_T": 0.0, "avg_cool": 0.0, "avg_fus": 0.0}
        N = self.get_particle_count()
        if N == 0 or not self._pd: return stats
        try:
            needed = "velocities masses densities temperatures cooling_rates fusion_rates"
            if self._config.get('calculate_core_concentration', False): needed += " positions"
            self._pd.ensure(needed, "cpu")
            vel = self._pd.get("velocities", "cpu"); mass = self._pd.get("masses", "cpu")
            dens = self._pd.get("densities", "cpu"); temp = self._pd.get("temperatures", "cpu")
            cool = self._pd.get("cooling_rates", "cpu"); fus = self._pd.get("fusion_rates", "cpu")
            vel_sq = np.sum(vel * vel, axis=1); ke = 0.5 * mass * vel_sq
            stats["avg_KE"] = np.sum(ke) / N if N > 0 else 0.0
            stats["avg_vel"] = np.mean(np.sqrt(vel_sq)) if N > 0 else 0.0
            stats["max_rho"] = np.max(dens) if dens.size > 0 else 0.0
            stats["max_T"] = np.max(temp) if temp.size > 0 else 0.0
            total_mass = np.sum(mass)
            if total_mass > 1e-9:
                 stats["avg_cool"] = np.sum(cool * mass) / total_mass
                 stats["avg_fus"] = np.sum(fus * mass) / total_mass
            if self._config.get('calculate_core_concentration', False):
                 try:
                     pos = self._pd.get("positions", "cpu"); center = np.mean(pos, axis=0)
                     core_radius_sq = (self._config.get('radius', 10.0) * 0.1)**2
                     dist_sq = np.sum((pos - center)**2, axis=1)
                     stats["core_c"] = np.sum(dist_sq < core_radius_sq)
                 except Exception: pass
        except Exception as e: print(f"Warn: Error computing stats: {e}")
        return stats


    def _check_end_conditions(self):
        """Checks if simulation max steps or max time have been reached."""
        if self._ended: return
        end_reason = None
        if self._max_steps is not None and self._steps_taken >= self._max_steps:
            end_reason = f"Reached maximum steps ({self._max_steps})"
        elif self._max_time is not None and self._time >= self._max_time:
            end_reason = f"Reached maximum time ({self._max_time:.3f})"
        if end_reason:
            print(f"Simulation ended: {end_reason}")
            self._ended = True; self._running = False; self._status_msg = STATUS_ENDED

    def _log_current_setup(self):
        """Helper to print the current model and integrator setup."""
        if not self._physics_manager or not self._integrator_manager: return
        print(f"  Current Setup:")
        print(f"    Precision: {'f64' if self._config.get('USE_DOUBLE_PRECISION') else 'f32'}")
        print(f"    N: {self.get_particle_count()}")
        models_info = self.get_current_models_info()
        for m_type, m_info in models_info.items():
             status_mark = ""
             if m_info:
                 is_avail = m_info.get('_backend_available', True)
                 status_mark = "" if is_avail else "[Backend Unavailable]"
                 print(f"    {m_type.capitalize():<10}: {m_info['name']} ({m_info['id']}) {status_mark}")
             else:
                 print(f"    {m_type.capitalize():<10}: -")
        integrator_info = self.get_current_integrator_info()
        print(f"    Integrator: {integrator_info['name']} ({integrator_info['id']})") if integrator_info else "    Integrator: -"


    def get_graph_data(self) -> Optional[Dict[str, Any]]:
        """Returns a copy of the collected graph data, ensuring final snapshot is collected if missing."""
        if self._graph_data is None:
            print("Warning: get_graph_data called but _graph_data is None.")
            return None

        # check if the 'final_snapshot' key exists and if its dictionary is empty.
        # if it's missing or empty, collect the data using the current simulation state.
        snapshot_data = self._graph_data.get('final_snapshot')
        if snapshot_data is None or not snapshot_data: # checks for none or empty dict
             if self._pd and self.get_particle_count() > 0:
                 print("Snapshot data missing or empty in get_graph_data, collecting current state...")
                 self._collect_final_snapshot_data() # force collection of current state
             else:
                 print("Warning: Cannot collect snapshot data (No ParticleData or N=0).")
                 # ensure final_snapshot exists as an empty dict if it wasn't there
                 if 'final_snapshot' not in self._graph_data:
                     self._graph_data['final_snapshot'] = {}

        return self._graph_data.copy() # return a copy

    def __del__(self):
         """Suggests cleanup when the simulator object is deleted."""
         print("Simulator.__del__ suggesting cleanup...")
         if self._pd: self._pd.cleanup_gpu_resources()
         if self._physics_manager: self._physics_manager.cleanup_models()

    def get_initial_energy_components_for_log(self) -> Optional[Dict[str, float]]:
        """Returns the detailed energy components calculated at initialization."""
        return self._initial_energy_components_for_log