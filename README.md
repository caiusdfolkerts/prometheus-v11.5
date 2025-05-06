# Prometheus: Simulating Stellar Formation

Version: 0.4.1 

6th May 2025

Caius Folkerts


Prometheus is a numerical simulation server designed for astrophysical N-body and Smoothed Particle Hydrodynamics (SPH) simulations, with a focus on star formation scenarios. It features a Python-based backend for physics calculations and a web-based frontend for real-time visualization and control. The backend supports various computational methods, including Numba for JIT-compiled CPU kernels, Taichi for cross-platform CPU/GPU parallelism, and a framework for CUDA integration.

Note: Requires Python <=3.11

## Table of Contents

1.  [The Computational Physics Prize](#computational-phys-prize)
1.  [Overview](#overview)
2.  [Features](#features)
3.  [Project Structure](#project-structure)
4.  [Physics Implementation](#physics-implementation)
    *   [Gravitational Dynamics](#gravitational-dynamics)
    *   [Smoothed Particle Hydrodynamics (SPH)](#smoothed-particle-hydrodynamics-sph)
    *   [Thermodynamics](#thermodynamics)
    *   [Particle Coloring](#particle-coloring)
5.  [Computational Methods](#computational-methods)
    *   [Taichi for GPU/CPU Acceleration](#taichi-for-gupucpu-acceleration)
    *   [Numba for JIT CPU Acceleration](#numba-for-jit-cpu-acceleration)
    *   [CUDA and GPU Acceleration (Framework)](#cuda-and-gpu-acceleration-framework)
    *   [Parallelism and Threading](#parallelism-and-threading)
    *   [Data Management (`ParticleData`)](#data-management-particledata)
6.  [Time Integration](#time-integration)
7.  [Backend Server (`main.py`)](#backend-server-mainpy)
8.  [Frontend UI (`ui/`)](#frontend-ui-ui)
9.  [Configuration](#configuration)
10. [Output](#output)
11. [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Running the Simulation](#running-the-simulation)
12. [Citations](#citations)



## The Computational Physics Prize

Prometheus and its associated frontend server was written as a submission for the 2025 Computational Physics Prize. The prize (established in 2018) is an independent open ended assignment. The 2025 Computational Physics prize. The 2025 assignment was to model the formation of a star from a spherically symmetric gas cloud.

## Overview

The principle of my code model is to provide a user with a comprehensive view of the early stages of stellar formation - giving a view of the initial collapse under gravity followed by subsequent increase in temperature and pressure. The code also measures the onset of fusion in the core as a certain temperature threshold is reached. 

In order to do this well, it is required that one represent in a comprehensible fashion as many particles as possible. The simulation code runs on a Python backend, which gives the user options between different gravity algorithms (such as Barnes-Hut and direct Particle-Particle interactions). The code allows a user to use both the CPU, using codes that make use of Numba Just In Time compilation (JIT) to optimize compilation and parallelism on CPU, whilst also giving the user the choice to use Taichi Lang to optimize for the added parallelism and acceleration of the GPU of any machine, including with Vulkan, NVIDIA CUDA, OpenGL and Metal backends.  

This Python backend, containing all of the simulation logic, sends simulation data to a JavaScript frontend, which is operated by three.js,  providing a 3 dimensional view of the particles as they are simulated in real time. 

## Features

*   **Modular Physics Engine**: Supports interchangeable models for gravity, SPH, thermodynamics, and particle coloring.
*   **Multiple Computational Backends**:
    *   **Numba**: JIT compilation for CPU-bound kernels.
    *   **Taichi**: Cross-platform (CPU/GPU - CUDA, Vulkan, Metal, OpenGL) acceleration via Python-embedded DSL.
    *   **CUDA (via CuPy/Taichi)**: Framework for NVIDIA CUDA.
*   **Real-time Web UI**:
    *   3D visualization of particles using Three.js.
    *   Live parameter updates and model selection.
    *   Realtime feedback on simulation status and diagnostics.
*   **Flask & SocketIO Backend**: Robust web server for API requests and WebSocket communication.
*   **Flexible Time Integration**: Supports multiple integration schemes including Leapfrog KDK and Yoshida 4th-Order.
*   **Dynamic Configuration**: Simulation parameters and model choices can be adjusted at runtime or restart.
*   **Precision Control**: Attempts to support single (f32) and double (f64) precision, dynamically determining effective precision based on user hardware/preference.
*   **Plotting**: Generates multi page PDF reports giving graphs of various evolving variables.
*   **Threading Model**: Main thread for core tasks, server thread for web requests, and a dedicated worker thread for the simulation loop.

## Project Structure

prometheus/
├── config/                     # Configuration files
│   ├── available_integrators.py  # Definitions of time integrators
│   ├── available_models.py       # Definitions of physics models
│   ├── default_settings.py     # Default simulation parameters
│   └── param_defs.py             # UI parameter definitions
├── sim/                        # Core simulation logic
│   ├── constants.py            # Physical constants
│   ├── integrator_manager.py   # Manages time integrators
│   ├── integrators/            # Integrator implementations (base, leapfrog, yoshida4)
│   ├── particle_data.py        # Manages particle data arrays across CPU/GPU
│   ├── physics_manager.py      # Manages physics models
│   ├── physics/                # Physics model implementations
│   │   ├── base/               # Base classes for physics models
│   │   ├── color/              # Particle coloring models
│   │   ├── gravity/            # Gravity models
│   │   ├── sph/                # SPH models
│   │   └── thermo/             # Thermodynamics models
│   ├── plotting.py             # PDF plot generation
│   ├── simulator.py            # Main simulation orchestrator class
│   └── utils.py                # Utility functions (dynamic import, backend checks)
├── ui/                         # Frontend files
│   ├── index.html              # Main UI page
│   ├── main.js                 # Frontend JavaScript logic (Three.js, SocketIO)
│   └── style.css               # CSS for the UI
├── output/                     # Default directory for generated PDF plots
└── main.py                     # Main application entry point (server, task processor)


## Physics Implementation

The simulation incorporates several key physical processes, each managed by a modular component. These include SPH forces, thermodynamics modules, and Coloring modules

### Gravitational Dynamics

Gravity models compute the gravitational forces between particles.

*   **Direct Particle-Particle (PP)**:
    *   `GravityPPCpu` (`sim/physics/gravity/gravity_pp_cpu.py`): A pure NumPy implementation for CPU. O(N^2) complexity, suitable for small N. Accelerated and parelellised by numba just in time compilation (JIT)
    *   `GravityPPGpu` (`sim/physics/gravity/gravity_pp_gpu.py`): A Taichi-accelerated implementation for CPU/GPU. Also O(N^2) but significantly faster for moderate N on GPUs due to parallelism. The Taichi kernel `compute_forces_n2_ti_kernel` performs the parallel summation.
*   **Barnes-Hut (BH)**:
    *   `GravityBHNumba` (`sim/physics/gravity/gravity_bh_numba.py`): An O(N log N) algorithm accelerated with Numba for CPU execution. It builds an octree to approximate forces from distant particle groups. Key Numba kernels include `parallel_compute_morton_codes_ids_numba` for Morton ordering, `_build_node_iterative` for tree construction, `_bh_compute_node_CoM_iterative` for Center of Mass calculation, and `bh_calculate_forces_parallel_iterative_numba` for force evaluation. The potential energy is currently calculated using a direct N^2 Numba sum (`direct_potential_kernel_numba`) within this class, not via tree traversal.

### Smoothed Particle Hydrodynamics (SPH)

SPH models simulate fluid dynamics by representing the fluid as a collection of particles.

*   `SPHTaichi` (`sim/physics/sph/sph_taichi.py`): Implements SPH calculations using Taichi for CPU/GPU acceleration.
    *   **Density Calculation**: `calculate_density_sph_ti_kernel` uses a grid-based neighbor search and the Poly6 SPH kernel (Monaghan, 1992) to compute particle densities.
        *   $ \rho_i = \sum_j m_j W(\mathbf{r}_i - \mathbf{r}_j, h) $
        *   The Poly6 kernel: $ W_{poly6}(r, h) = \frac{315}{64 \pi h^9} \begin{cases} (h^2 - r^2)^3 & 0 \le r \le h \\ 0 & r > h \end{cases} $
    *   **Equation of State (EOS) and Pressure Term**: This model now *internally* calculates temperature, pressure (gas + optional radiation pressure), and the term $P/\rho^2$ required for symmetric SPH force calculation. This is done in the `_update_eos_p_term_ti_kernel` Taichi kernel. This improves consistency and performance by keeping EOS calculations on the Taichi backend.
        *   Temperature: $ T_i = u_i / C_{v,sim} $ where $u_i$ is specific internal energy.
        *   Gas Pressure: $ P_{gas,i} = \rho_i R_{sim} T_i $
        *   Radiation Pressure (optional): $ P_{rad,i} = \frac{1}{3} a_{rad} T_i^4 $
    *   **Pressure Force & Work Calculation**: `calculate_sph_force_energy_visc_ti_kernel` computes SPH pressure forces and the rate of change of internal energy due to SPH work (PdV work).
        *   Pressure Force: $ \mathbf{F}_{press,i} = - \sum_j m_i m_j \left( \frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2} \right) \nabla_i W(\mathbf{r}_{ij}, h) $ (symmetric form). The Spiky kernel is used for the gradient $ \nabla W $.
        *   Spiky Kernel Gradient: $ \nabla W_{spiky}(r,h) = -\frac{45}{\pi h^6} (h-r)^2 \frac{\mathbf{r}}{r} $ for $0 \le r \le h$.
        *   Work Term (rate of change of specific internal energy): $ (\frac{du_i}{dt})_{work} = \frac{1}{2} \sum_j m_j \left( \frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2} \right) (\mathbf{v}_i - \mathbf{v}_j) \cdot \nabla_i W_{ij} $
    *   **Artificial Viscosity**: The framework for artificial viscosity is present (`alpha_visc`, `beta_visc` parameters), but the full AV calculation in the Taichi kernel is currently a placeholder.

### Thermodynamics

Thermodynamic processes beyond SPH work are handled by dedicated models.

*   `ThermoNumba` (`sim/physics/thermo/thermo_numba.py`): Uses Numba-accelerated kernels for CPU execution.
    *   **Equation of State (EOS)**: `calculate_temperature_numba` and `calculate_pressure_numba` compute temperature and pressure from density and internal energy, similar to the internal EOS in `SPHTaichi` but implemented in Numba for CPU.
    *   **Cooling**: `calculate_cooling_rate_numba` implements a power-law cooling function $ (\frac{du}{dt})_{cool} = -\Lambda_{coeff} \rho T^\beta $.
    *   **Fusion**: `calculate_fusion_rate_numba` implements a power-law fusion heating rate $ (\frac{du}{dt})_{fus} = \epsilon_{coeff} \rho T^p $ above a temperature threshold $T_{thresh}$.

### Particle Coloring

Color models determine particle colors for visualization based on physical properties.

*   `ColorTempNumba` (`sim/physics/color/color_temp_numba.py`): Maps particle temperature to a color gradient using a logarithmic scale. Uses the Numba kernel `calculate_colors_log_scale_numba_kernel`.
*   `ColorSpeedNumba` (`sim/physics/color/color_speed_numba.py`): Maps particle speed to a color gradient using a linear scale, typically based on percentiles. Uses the Numba kernel `map_value_to_color_linear_numba`.
*   `ColorSPHForceNumba` (`sim/physics/color/color_sph_force_numba.py`): Maps the magnitude of the SPH force ($F_{total} - F_{gravity}$) to a color gradient using a logarithmic scale. Reuses `calculate_colors_log_scale_numba_kernel`.

## Computational Methods

Prometheus employs several techniques to accelerate computations and manage data.

### Taichi for GPU/CPU Acceleration

[Taichi Lang](https://www.taichi-lang.org/) is a Python-embedded domain-specific language (DSL) for high-performance numerical computation.

*   **Usage**: Taichi is used in `GravityPPGpu` and `SPHTaichi` to write kernels that can run on various backends (CUDA, Vulkan, Metal, OpenGL, and multi-core CPU).
*   **How it Works**:
    *   **Kernels**: Functions decorated with `@ti.kernel` are compiled by Taichi into efficient backend code.
    *   **Fields**: Taichi uses `ti.field` and `ti.Vector.field` to define dense data structures (similar to arrays) that reside on the target device (CPU/GPU memory). Data transfers between NumPy arrays and Taichi fields are managed by `from_numpy()` and `to_numpy()`.
    *   **Data-Oriented Programming**: Taichi encourages structuring loops over data elements (e.g., `for i in field:`), which it parallelizes.
    *   **Type System**: Taichi has its own type system (e.g., `ti.f32`, `ti.f64`, `ti.i32`). The `main.py` script dynamically attempts to initialize Taichi with a user-preferred precision (f64 or f32) and backend, falling back to alternatives if the preferred setup fails or is unsupported by hardware. The effective Taichi precision is then used by Taichi-based models.
    *   **Synchronization**: `ti.sync()` is used to ensure kernel execution completes before accessing results on the CPU.
    *   **Grid Computations**: `SPHTaichi` utilizes Taichi's built-in grid data structures (`grid_num_particles`, `grid_particle_indices_field`) for efficient neighbor searching in SPH.
*   **Initialization**: `main.py` handles Taichi initialization. It tries a sequence of architectures (e.g., user-defined GPU, CPU) and floating-point precisions (user-defined, f32) to find a working configuration. This ensures the simulation can run on diverse hardware.
*   **Main Thread Requirement**: Some Taichi operations, particularly field allocations or re-allocations (e.g., during simulation restarts with different particle counts), must occur on the same thread where `ti.init()` was called (the main thread). Prometheus addresses this using a `queue.Queue` (`main_thread_task_queue` in `main.py`) to delegate such tasks from worker threads to the main thread. The `_execute_main_thread_task` function processes these tasks.

### Numba for JIT CPU Acceleration

[Numba](https://numba.pydata.org/) is a Just-In-Time (JIT) compiler that translates Python functions to optimized machine code at runtime.

*   **Usage**: Numba is used in `GravityBHNumba`, `ThermoNumba`, and the color models (`ColorTempNumba`, `ColorSpeedNumba`, `ColorSPHForceNumba`) to accelerate CPU-bound Python code.
*   **How it Works**:
    *   `@njit`: The primary decorator used. `cache=True` saves compiled versions, `fastmath=True` allows less precise but faster maths, `parallel=True` enables automatic parallelization with `prange`.
    *   `prange`: Used instead of `range` in Numba-jitted functions to indicate loops that can be parallelized across CPU cores.
    *   **Type Inference**: Numba infers variable types at compile time. Kernels often expect NumPy arrays of specific dtypes (typically `float64` for physics calculations in this project).
*   **Benefits**: Provides significant speedups for numerical loops on the CPU without requiring C/C++/Fortran.

### Parallelism and Threading

*   **Backend Threading (`main.py`)**:
    *   **Main Thread**: Handles initial setup (including Taichi initialization), runs the main thread task processor loop for Taichi-sensitive operations.
    *   **Server Thread (daemon)**: Runs the Flask/SocketIO web server to handle HTTP API requests and WebSocket communication with the UI.
    *   **Simulation Worker Thread (daemon)**: Executes the main simulation loop (`simulation_loop_worker`), advancing the simulation step-by-step.
    *   **Synchronization**: `threading.Lock` (`sim_lock`) protects shared access to the `Simulator` instance. `threading.Event` (`stop_simulation_flag`) controls the simulation worker thread.
*   **Task Queue**: A `queue.Queue` (`main_thread_task_queue`) is used to delegate tasks (like Taichi field (re)allocations during restarts) that must run on the main thread from other threads (e.g., the server thread handling a restart command).
*   **Kernel-Level Parallelism**:
    *   **Numba**: `prange` enables multi-core CPU parallelism.
    *   **Taichi**: Kernels are implicitly parallelized by Taichi across GPU threads or CPU cores.

### Data Management (`ParticleData`)

The `ParticleData` class (`sim/particle_data.py`) is crucial for managing particle attributes across different compute backends.

*   **Attributes**: Stores core particle data like positions, velocities, masses, forces, densities, internal energies, etc., as NumPy arrays on the CPU by default.
*   **Device Tracking**: Tracks the "authoritative" location of each attribute (e.g., "cpu", "gpu:ti", "gpu:cupy").
*   **GPU Copies**: Maintains copies of data on GPU devices (Taichi fields or CuPy arrays) in `_gpu_copies`.
*   **Transfers**:
    *   `get(name, device, writeable)`: Retrieves data. If `device` is different from the current location, it implicitly triggers a transfer (if `allow_implicit_transfers` is true). Writeable access forces data to CPU.
    *   `ensure(names, target_device)`: Explicitly ensures specified attributes are up-to-date on the `target_device`, performing transfers if necessary.
    *   `set(name, data, source_device)`: Updates CPU data and marks GPU copies as "dirty," requiring re-transfer.
*   **Precision**: Initializes NumPy arrays with a precision determined at startup (`_effective_np_float_type`). Taichi fields use the Taichi runtime's default float type.
*   **Synchronization**: Includes a `synchronize(device)` method to wait for device operations (e.g., `ti.sync()`).
*   **Cleanup**: `cleanup_gpu_resources()` attempts to release GPU memory.

## Time Integration

The simulation's time evolution is handled by integrators.

*   **`IntegratorManager` (`sim/integrator_manager.py`)**:
    *   Loads available integrator definitions from `config/available_integrators.py`.
    *   Allows the `Simulator` to select and switch between active integrators.
    *   The `advance()` method calls the `step()` method of the active integrator.
*   **`Integrator` Base Class (`sim/integrators/base.py`)**: Defines the interface for all integrators.
*   **Available Integrators**:
    *   **Leapfrog (KDK)** (`sim/integrators/leapfrog.py`): A second-order symplectic integrator.
        *   Sequence: Kick (velocities by $dt/2$), Drift (positions by $dt$), Kick (velocities by $dt/2$).
        *   Energy Update: Integrates internal energy changes using rates averaged over the timestep ($ (rate(t) + rate(t+dt))/2 $). Requires storing previous step's rates (`_prev` fields in `ParticleData`).
    *   **Yoshida 4th-Order** (`sim/integrators/yoshida4.py`): A fourth-order symplectic integrator.
        *   Involves multiple kick and drift sub-steps within a single $dt$, using specific coefficients (W0, W1, Ck, Dk).
        *   Requires 3 force/rate evaluations per full timestep.
        *   Integrates internal energy changes within its kick substeps.
*   **Force Callback**: Integrators receive a `forces_cb` function from the `Simulator`. This callback (`_recompute_forces_and_physics`) triggers the `PhysicsManager` to compute all active physics (forces, SPH density, EOS, energy rates, colors).

## Backend Server (`main.py`)

`main.py` is the entry point for the simulation server.

*   **Initialization**:
    *   Sets up Python paths.
    *   Initializes Taichi, attempting various configurations to determine effective compute precision and backend.
    *   Checks for optional libraries (PyTorch, CuPy).
    *   Updates default settings based on effective precision.
*   **Threading**: Establishes the main, server, and simulation worker threads, along with synchronization primitives.
*   **Task Queue**: Creates `main_thread_task_queue` for delegating main-thread-only tasks (e.g., Taichi field allocations during restarts).
*   **Flask & SocketIO Server**:
    *   Configures and runs a Flask web server with Flask-SocketIO.
    *   Handles HTTP API requests (e.g., `/api/config`, `/api/state`, `/api/command`, `/api/generate_pdf`).
    *   Manages WebSocket communication for real-time updates (`state_update`) and commands (`send_command`) with the frontend.
    *   Serves static UI files from the `ui/` folder.
*   **Command Handling**: The `handle_command` function processes commands from the UI to control the simulation (toggle run, reset, restart, set parameters, select models/integrators). It interacts with the `Simulator` instance, ensuring thread safety with `sim_lock`.
*   **Simulation Loop (`simulation_loop_worker`)**:
    *   Runs in its own thread.
    *   Continuously advances the simulation (if running) by calling `sim.advance_one_step()`.
    *   Logs energy conservation to the console.
    *   Collects data for graphing at specified intervals.
    *   Emits periodic state updates to the UI via SocketIO.
*   **PDF Generation**: Provides an API endpoint (`/api/generate_pdf`) to trigger the generation of summary plots via `sim.plotting.generate_plots_pdf`.
*   **Application Orchestration**: The `if __name__ == "__main__":` block coordinates startup:
    *   Calls `main_backend_setup` to initialize the `Simulator` instance (passing the task queue reference).
    *   Starts the web server thread.
    *   Runs the `main_thread_task_processor` loop in the main thread.
    *   Handles graceful shutdown on Ctrl+C.

## Frontend UI (`ui/`)

The user interface is a single-page web application.

*   **`index.html`**: The main HTML structure.
*   **`style.css`**: CSS for styling the UI components.
*   **`main.js`**: Handles all client-side logic:
    *   **Three.js**: Initializes a 3D scene for visualizing particles as points. `OrbitControls` allow camera manipulation.
    *   **SocketIO Client**: Connects to the backend server, sends commands, and receives real-time `config` and `state_update` messages.
    *   **Dynamic Controls**: Populates sliders for parameters (defined in `PARAM_DEFS`) and select dropdowns for physics models and integrators (from `AVAILABLE_MODELS`, `AVAILABLE_INTEGRATORS`).
    *   **State Display**: Updates UI elements to reflect the current simulation time, steps, particle count, diagnostic statistics, and active models.
    *   **Command Sending**: Attaches event listeners to buttons and controls to send commands to the backend.
    *   **Graph Checkboxes**: Dynamically creates checkboxes based on `GRAPH_SETTINGS` from the backend, allowing users to (conceptually) select which plots appear in the PDF. The actual plotting logic is backend-side.
    *   **PDF Generation Trigger**: A button initiates a request to the `/api/generate_pdf` endpoint.

## Configuration

Simulation behavior is largely controlled by configuration files in the `config/` directory. To modify the performance of the code when considering stellar formation, it is critical that this be considered

*   **`default_settings.py`**: Defines default values for simulation parameters (e.g., `dt`, `N`, `G`, SPH parameters, thermodynamic constants like `gas_constant_sim`, `cv`). It also calculates some derived defaults (e.g., `L` from `radius`, `cv` from `gas_constant_sim` and `sph_eos_gamma`). This file is updated at startup with effective precision types.
*   **`param_defs.py`**: Defines the properties of UI sliders (label, min, max, step, format, live-updatable status).
*   **`available_models.py`**: A dictionary listing all available physics models, grouped by type (gravity, sph, thermo, color). Each model definition includes its ID, name, description, module path, class name, and required backend. The `_backend_available` flag is set at runtime by the `PhysicsManager`.
*   **`available_integrators.py`**: A list defining all available time integrators, including ID, name, description, module path, class name, and order.

## Output

*   **Real-time Visualization**: The primary output is the 3D visualization in the web UI.
*   **Console Logs**: The backend server prints status messages, timing information, and energy conservation logs to the console.
*   **PDF Plots**: Users can trigger the generation of a multi-page PDF file containing diagnostic plots (e.g., energy evolution, momentum conservation, distributions). This is handled by `sim/plotting.py` and plots are saved to the `output/` directory.

## Getting Started

### Prerequisites

*   Python 3.11
*   Required Python packages (see `requirements.txt`):
*   A modern web browser supporting WebGL and WebSockets.
*   For GPU acceleration with Taichi or CuPy:
    *   A compatible GPU (NVIDIA for CUDA, AMD/Intel for Vulkan/OpenGL, Apple Silicon for Metal).
   
### Running the Simulation

1.  Navigate to the `prometheus` root directory in your terminal.
2.  Run the main server script:
    ```bash
    python main.py
    ```
3.  The server will start, and you should see output indicating the effective precision, library status, and the URL to access the UI (typically `http://localhost:7847/` or `http://<your-ip>:7847/`).
4.  Open the URL in your web browser.
5.  Use the control panel to configure and run simulations.

## Citations

*   **Barnes-Hut Algorithm**:
    *   Barnes, J., & Hut, P. (1986). A hierarchical O(N log N) force-calculation algorithm. *Nature, 324*(6096), 446-449.
*   **Smoothed Particle Hydrodynamics (SPH)**:
    *   Monaghan, J. J. (1992). Smoothed particle hydrodynamics. *Annual Review of Astronomy and Astrophysics, 30*(1), 543-574.
    *   Monaghan, J. J. (2005). Smoothed particle hydrodynamics. *Reports on Progress in Physics, 68*(8), 1703.
    (The specific SPH kernels like Poly6 and Spiky are standard in SPH literature originating from or popularized by Monaghan.)
*   **Yoshida 4th-Order Integrator**:
    *   Yoshida, H. (1990). Construction of higher order symplectic integrators. *Physics Letters A, 150*(5-7), 262-268.
*   **Taichi Programming Language**:
    *   Hu, Y., Ma, K., Liu, X., Wang, M., Li, S., Wang, B., & Han, T. (2019). Taichi: a language for high-performance computation on spatially sparse data structures. *ACM Transactions on Graphics (TOG), 38*(6), 1-16. (SIGGRAPH Asia 2019)
*   **Numba**:
    *   Lam, S. K., Pitrou, A., & Seibert, S. (2015). Numba: A LLVM-based Python JIT compiler. In *Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC* (pp. 1-6).