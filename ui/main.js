// prometheus/ui/main.js
/**
 * prometheus star formation UI client.
 * this script initializes and manages the web user interface for a star formation simulation.
 * it handles 3d visualization using three.js, websocket communication with a backend server
 * for simulation data and control, and dynamic ui elements for user interaction.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

console.log("Prometheus Star Formation UI Initializing...");

// global three.js and simulation state variables
let scene, camera, renderer, controls;
let particlesGeometry, particlesMaterial, particleSystem;
let animationFrameId = null;
let configData = null; // stores configuration received from the server
let currentSimData = { // holds the current state of the simulation
    time: 0.0, status_msg: "Initializing...", running: false, ended: false, N: 0,
    positions: null, colors: null, stats: {},
    current_models: {}, current_integrator: "-"
};
let socket = null; // websocket connection object
const SERVER_URL = `http://${window.location.hostname}:7847`;
let isUIStaticInitialized = false; // flag for successful static ui setup (incl. three.js)
let areDynamicControlsInitialized = false; // flag for successful dynamic ui controls setup (sliders/selectors)

// creates a circular texture for particles
function createCircleTexture(diameter = 16, color = 'white') {
    const canvas = document.createElement('canvas');
    canvas.width = diameter; canvas.height = diameter;
    const context = canvas.getContext('2d');
    context.beginPath();
    context.arc(diameter / 2, diameter / 2, diameter / 2, 0, 2 * Math.PI, false);
    context.fillStyle = color; context.fill();
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
}

// formats a numerical value to a string based on a python-like format specifier
function formatValue(value, fmt) {
    try {
        if (fmt === undefined || fmt === null) fmt = '{:.2f}';
        if (typeof value !== 'number' || !isFinite(value)) { return String(value); }
        const matchExp = fmt.match(/\{:([+ ]?)(\.?)([0-9]*)e\}/i);
        const matchFix = fmt.match(/\{:([+ ]?)(\.?)([0-9]*)f\}/i);
        const matchInt = fmt.match(/\{:[d]\}/i);
        if (matchInt) { return Math.round(value).toString(); }
        else if (matchExp) { const prec = parseInt(matchExp[3] || '2'); return value.toExponential(prec); }
        else if (matchFix) { const prec = parseInt(matchFix[3] || '2'); return value.toFixed(prec); }
        else { return value.toPrecision(3); }
    } catch (e) { console.warn(`Format error ${value} with ${fmt}:`, e); return String(value); }
}

// updates the ui element displaying websocket connection status
function updateConnectionStatus(status) {
    const statusElem = document.getElementById('connection-status');
    if (!statusElem) { return; }
    const parentElem = statusElem.parentElement || document.body;
    parentElem.className = `connection-${status}`;
    switch (status) {
        case 'connected': statusElem.textContent = 'Connected'; statusElem.style.color = 'limegreen'; break;
        case 'disconnected':
            statusElem.textContent = 'Disconnected'; statusElem.style.color = '#f66';
            if (isUIStaticInitialized) { setControlsEnabled(false); }
            const runButton = document.getElementById('run-button');
            if (runButton) runButton.textContent = "Run";
            areDynamicControlsInitialized = false;
            break;
        case 'connecting': statusElem.textContent = 'Connecting...'; statusElem.style.color = 'orange'; break;
        default: statusElem.textContent = 'Unknown'; statusElem.style.color = '';
    }
}

// list of known dom element ids for enabling/disabling controls
const allKnownElementIds = [
    'scene-container', 'loading-indicator', 'control-panel', 'run-button',
    'center-view-button',
    'restart-button',
    'rad-press-toggle', 'cooling-toggle', 'fusion-toggle',
    'live-params-container', 'restart-params-container',
    'feedback-container',
    'status-text', 'time-text', 'steps-text', 'n-particles-text',
    'ke-text', 'vel-text', 'core-text', 'density-text', 'temp-text', 'cool-text',
    'fusion-text', 'connection-status', 'sim-version',
    'gravity-select', 'sph-select', 'thermo-select', 'color-select', 'integrator-select',
    'graph-controls-container', 'generate-pdf-button', 'pdf-feedback-message'
];

// enables or disables ui controls based on the 'enabled' flag
function setControlsEnabled(enabled) {
    if (!areDynamicControlsInitialized && enabled) {
        return;
    }

    let dynamicControlIds = [];
    if (configData?.PARAM_DEFS) {
        Object.keys(configData.PARAM_DEFS).forEach(key => {
            dynamicControlIds.push(`${key}-slider`);
        });
    }
    const allIds = [...allKnownElementIds, ...dynamicControlIds];

    allIds.forEach(id => {
        const elem = document.getElementById(id);
        if (elem) {
            if (!dynamicControlIds.includes(id) || document.getElementById(id)) {
                 elem.disabled = !enabled;
            }
        }
    });

    const runButton = document.getElementById('run-button');
    const restartButton = document.getElementById('restart-button');

    if (runButton) { runButton.disabled = !enabled || currentSimData.ended; }
    if (restartButton) restartButton.disabled = !enabled;
}

// initializes and manages the websocket connection and its event handlers
function connectWebSocket() {
    console.log(`Attempting to connect Socket.IO to: ${SERVER_URL}`);
    updateConnectionStatus('connecting');

    socket = io(SERVER_URL, { transports: ['websocket'], reconnectionAttempts: 5 });

    socket.on('connect', () => {
        console.log("Socket.IO connected:", socket.id);
        updateConnectionStatus('connected');
        if (areDynamicControlsInitialized) {
             setControlsEnabled(true);
        }
    });

    socket.on('config', (message) => {
        console.log("Received 'config':", message);
        const data = (message && typeof message === 'object' && 'data' in message) ? message.data : message;

        if (data && data.PARAM_DEFS && data.DEFAULT_SETTINGS && data.AVAILABLE_MODELS && data.AVAILABLE_INTEGRATORS) {
            configData = data;

            if (!isUIStaticInitialized) {
                 console.error("Config received, but static UI initialization failed. Cannot initialize dynamic controls.");
                 displayFatalError("Static UI Failed. Dynamic controls blocked.");
                 return;
            }

            if (!areDynamicControlsInitialized) {
                 console.log("Config received, initializing dynamic controls, selectors, and graph options...");
                 const controlsSuccess = initUIControls();
                 if (controlsSuccess) {
                     populateModelSelectors();
                     populateGraphCheckboxes();
                     areDynamicControlsInitialized = true;
                     if (socket && socket.connected) {
                         console.log("Dynamic controls initialized, enabling controls.");
                         setControlsEnabled(true);
                     }
                 } else {
                      console.error("Failed to initialize dynamic UI controls (sliders/listeners).");
                 }
             } else {
                 console.log("Re-populating dynamic UI components after receiving config again.");
                 initUIControls();
                 populateModelSelectors();
                 populateGraphCheckboxes();
             }

             const simVersionElem = document.getElementById('sim-version');
             if (simVersionElem) {
                 simVersionElem.textContent = configData.SIM_VERSION ?? '?.?';
             }
        } else {
            console.error("Invalid 'config' structure received:", message);
        }
    });

    socket.on('state_update', (message) => {
        if (!isUIStaticInitialized) return;
        const data = (message && typeof message === 'object' && 'data' in message) ? message.data : message;
        if (data && data.positions !== undefined) {
            updateSimulationState(data);
        }
    });

    socket.on('command_response', (message) => {
        if (!isUIStaticInitialized) return;
        const data = (message && typeof message === 'object' && 'data' in message) ? message.data : message;
        console.log("Command Response:", data);
        if (data && !data.success) { alert(`Command Error: ${data.message}`); }
        const runButton = document.getElementById('run-button');
        if (data && data.isRunning !== undefined && runButton) {
            runButton.textContent = data.isRunning ? "Pause" : "Run";
            if (areDynamicControlsInitialized) {
                setControlsEnabled(socket?.connected);
            }
        }
    });

    socket.on('connect_error', (error) => {
        console.error("Socket.IO connection error:", error);
        updateConnectionStatus('disconnected');
    });
    socket.on('disconnect', (reason) => {
        console.log("Socket.IO closed:", reason);
        updateConnectionStatus('disconnected');
    });
}

// sends a command to the backend via websocket
function sendCommand(command, data = {}) {
    if (socket && socket.connected) {
        const messagePayload = { command: command, ...data };
        console.log("Sending command:", messagePayload);
        socket.emit('send_command', messagePayload);
    } else {
        console.error("Socket.IO not connected. Cannot send command:", command);
        alert("Cannot connect to simulation backend.");
    }
}

// initializes static ui components, including the three.js scene
function initializeStaticUI() {
    console.log("InitializeStaticUI: Attempting direct DOM access...");

    const directSceneContainer = document.getElementById('scene-container');
    if (!directSceneContainer) {
         console.error("InitializeStaticUI FATAL: Cannot find element with id='scene-container'. Check HTML and timing.");
         displayFatalError("UI Initialization Failed: Critical element '#scene-container' not found.");
         isUIStaticInitialized = false;
         return false;
    }
    console.log("InitializeStaticUI: Found #scene-container:", directSceneContainer);

    const directControlPanel = document.getElementById('control-panel');
    const directParamsContainer = document.getElementById('sim-params-container');
    if (!directControlPanel || !directParamsContainer) {
        console.warn("InitializeStaticUI: #control-panel or #sim-params-container missing. Some UI parts may fail.");
    }

    try {
        initThreeJS(directSceneContainer);
        isUIStaticInitialized = true;
        console.log("InitializeStaticUI: Static UI + Three.js initialized successfully.");
        return true;
    } catch (e) {
        console.error("InitializeStaticUI: Error occurred during initThreeJS:", e);
        displayFatalError("Error during 3D scene initialization.");
        isUIStaticInitialized = false;
        return false;
    }
}

// displays a fatal error message overlay on the screen
function displayFatalError(message) {
     try {
         const body = document.querySelector('body');
         if(body) {
             let errorDiv = document.getElementById('fatal-error-message');
             if (!errorDiv) { errorDiv = document.createElement('div'); errorDiv.id = 'fatal-error-message'; body.prepend(errorDiv); }
              errorDiv.style.position = 'fixed'; errorDiv.style.top = '0'; errorDiv.style.left = '0'; errorDiv.style.width = '100%'; errorDiv.style.padding = '20px'; errorDiv.style.backgroundColor = 'darkred'; errorDiv.style.color = 'white'; errorDiv.style.zIndex = '1000'; errorDiv.style.fontSize = '1.2em'; errorDiv.style.textAlign = 'center';
             errorDiv.textContent = message;
         }
     } catch (e) { console.error("Error trying to display fatal error message:", e); }
 }

// updates the simulation state and particle system based on new data from the server
function updateSimulationState(newState) {
    if (!isUIStaticInitialized) return;
    if (!newState || newState.N === undefined) {
       if(newState) updateFeedbackUI(newState);
       return;
    }
   Object.assign(currentSimData, newState);
   try {
       if (!scene) return;
       if (newState.N > 0 && (!particleSystem || particleSystem.geometry.attributes.position.count !== newState.N)) { resizeParticleSystem(newState.N); }
       else if (newState.N === 0 && particleSystem) { particleSystem.visible = false; }

       if (particleSystem && newState.N > 0) {
           if (particleSystem.visible === false) particleSystem.visible = true;
           if (newState.positions && newState.positions.length === newState.N * 3) { particleSystem.geometry.attributes.position.array.set(newState.positions); particleSystem.geometry.attributes.position.needsUpdate = true; }
           else if (newState.positions) { console.warn(`Pos length mismatch: ${newState.positions.length} vs ${newState.N * 3}`); }
           if (newState.colors && newState.colors.length === newState.N * 3) { particleSystem.geometry.attributes.color.array.set(newState.colors); particleSystem.geometry.attributes.color.needsUpdate = true; }
           else if (newState.colors) { console.warn(`Color length mismatch: ${newState.colors.length} vs ${newState.N * 3}`); }
           if (particlesGeometry) particlesGeometry.computeBoundingSphere();
       }
   } catch (e) { console.error("Error updating Three.js buffers:", e); }

   updateFeedbackUI(currentSimData);

   if (areDynamicControlsInitialized) {
       const runButton = document.getElementById('run-button');
       if (runButton) {
           const isRunning = currentSimData.running && !currentSimData.ended;
           runButton.textContent = isRunning ? "Pause" : (currentSimData.ended ? "Ended" : "Run");
           runButton.disabled = currentSimData.ended || !(socket?.connected);
       }
       updateModelSelectorsFromState(currentSimData.current_models, currentSimData.current_integrator);
   }
}

// initializes dynamic ui controls like sliders and attaches event listeners
function initUIControls() {
    console.log("Initializing dynamic UI controls (sliders) and attaching listeners...");
    if (!configData) { console.error("Config data missing for UI controls."); return false; }

    const liveParamsContainer = document.getElementById('live-params-container');
    const restartParamsContainer = document.getElementById('restart-params-container');

    if (!liveParamsContainer || !restartParamsContainer) {
        console.error("CRITICAL: Slider containers ('live-params-container' or 'restart-params-container') not found! Cannot create sliders.");
        return false;
    }
    liveParamsContainer.innerHTML = '';
    restartParamsContainer.innerHTML = '';

    let dynamicControlsCreated = true;
    try {
         for (const [key, pdef] of Object.entries(configData.PARAM_DEFS)) {
             const fullLabel = `${pdef.label}:`;
             const container = document.createElement('div'); container.className = 'slider-container';
             const label = document.createElement('label'); label.htmlFor = `${key}-slider`; label.textContent = fullLabel; label.title = pdef.label;
             const slider = document.createElement('input'); slider.type = 'range'; slider.id = `${key}-slider`; slider.min = pdef.min ?? 0; slider.max = pdef.max ?? 100; slider.step = pdef.step ?? 1; slider.value = configData.DEFAULT_SETTINGS[key] ?? pdef.val ?? 0; slider.dataset.paramKey = key;
             const valueText = document.createElement('span'); valueText.id = `${key}-value`; valueText.textContent = formatValue(slider.value, pdef.fmt);
             container.appendChild(label); container.appendChild(slider); container.appendChild(valueText);

             if (pdef.live) {
                 liveParamsContainer.appendChild(container);
             } else {
                 restartParamsContainer.appendChild(container);
             }
             slider.addEventListener('input', handleSliderInput);
         }
     } catch (e) {
          console.error("Error creating sliders:", e);
          dynamicControlsCreated = false;
     }

    const radToggle = document.getElementById('rad-press-toggle');
    const coolToggle = document.getElementById('cooling-toggle');
    const fusToggle = document.getElementById('fusion-toggle');
    if (radToggle) radToggle.checked = configData.DEFAULT_SETTINGS.use_rad_press ?? false;
    if (coolToggle) coolToggle.checked = configData.DEFAULT_SETTINGS.use_cooling ?? false;
    if (fusToggle) fusToggle.checked = configData.DEFAULT_SETTINGS.use_fusion ?? false;

    document.getElementById('run-button')?.addEventListener('click', () => sendCommand('toggle_run'));
    document.getElementById('center-view-button')?.addEventListener('click', handleCenterViewButton);
    document.getElementById('restart-button')?.addEventListener('click', handleRestartButton);
    radToggle?.addEventListener('change', handleThermoToggles);
    coolToggle?.addEventListener('change', handleThermoToggles);
    fusToggle?.addEventListener('change', handleThermoToggles);
    document.getElementById('gravity-select')?.addEventListener('change', handleModelSelectionChange);
    document.getElementById('sph-select')?.addEventListener('change', handleModelSelectionChange);
    document.getElementById('thermo-select')?.addEventListener('change', handleModelSelectionChange);
    document.getElementById('color-select')?.addEventListener('change', handleModelSelectionChange);
    document.getElementById('integrator-select')?.addEventListener('change', handleModelSelectionChange);
    document.getElementById('generate-pdf-button')?.addEventListener('click', handleGeneratePdf);

    if (dynamicControlsCreated) {
         return true;
    } else {
        return false;
    }
}

// populates model and integrator selection dropdowns based on config data
function populateModelSelectors() {
    console.log("Populating model/integrator selectors...");
    if (!configData || !isUIStaticInitialized) { console.error("Cannot populate selectors: Missing configData or static UI failed."); return; }

    let allSelectorsFound = true;
    for (const modelType of ['gravity', 'sph', 'thermo', 'color']) {
        const selectElement = document.getElementById(`${modelType}-select`);
        if (selectElement) {
            selectElement.innerHTML = '';
            const models = configData.AVAILABLE_MODELS[modelType] || [];
            models.forEach(modelDef => {
                const option = document.createElement('option');
                option.value = modelDef.id;
                const backendStatus = modelDef._backend_available ? "" : " (X)";
                option.textContent = `${modelDef.name}${backendStatus}`;
                option.title = modelDef.description || modelDef.id;
                option.disabled = !modelDef._backend_available;
                selectElement.appendChild(option);
            });
            const defaultModelId = configData.DEFAULT_SETTINGS[`default_${modelType}_model`];
            if (defaultModelId && selectElement.querySelector(`option[value="${defaultModelId}"]:not(:disabled)`)) {
                 selectElement.value = defaultModelId;
            } else {
                const firstAvailableOption = selectElement.querySelector('option:not(:disabled)');
                if (firstAvailableOption) {
                     selectElement.value = firstAvailableOption.value;
                } else if (selectElement.options.length > 0) {
                     selectElement.selectedIndex = 0;
                }
            }
        } else {
            console.warn(`Selector element #${modelType}-select not found.`);
            allSelectorsFound = false;
        }
    }
    const integratorSelect = document.getElementById('integrator-select');
    if (integratorSelect) {
        integratorSelect.innerHTML = ''; const integrators = configData.AVAILABLE_INTEGRATORS || [];
        integrators.forEach(intDef => { const option = document.createElement('option'); option.value = intDef.id; option.textContent = intDef.name; option.title = intDef.description || intDef.id; integratorSelect.appendChild(option); });
        const defaultIntegratorId = configData.DEFAULT_SETTINGS.default_integrator;
        if (defaultIntegratorId && integratorSelect.querySelector(`option[value="${defaultIntegratorId}"]`)) { integratorSelect.value = defaultIntegratorId; }
        else if (integratorSelect.options.length > 0) { integratorSelect.selectedIndex = 0; }
    } else { console.warn(`Selector element #integrator-select not found.`); allSelectorsFound = false; }

    if (allSelectorsFound) console.log("Selectors populated successfully.");
    else console.warn("One or more selector elements missing during population.");
}

// updates the model selector ui elements based on the current simulation state
 function updateModelSelectorsFromState(currentModels, currentIntegrator) {
     if (!currentModels || !areDynamicControlsInitialized) return;
     for (const modelType of ['gravity', 'sph', 'thermo', 'color']) {
         const selectElement = document.getElementById(`${modelType}-select`);
         const backendValue = currentModels[modelType];
         if (selectElement && backendValue && selectElement.value !== backendValue) {
             if ([...selectElement.options].some(opt => opt.value === backendValue)) {
                 selectElement.value = backendValue;
             } else {
                 console.warn(`Backend ${modelType} ('${backendValue}') not in dropdown or is disabled.`);
             }
         }
     }
     const integratorSelect = document.getElementById('integrator-select');
     if (integratorSelect && currentIntegrator && integratorSelect.value !== currentIntegrator) {
         if ([...integratorSelect.options].some(opt => opt.value === currentIntegrator)) {
            integratorSelect.value = currentIntegrator;
        } else {
            console.warn(`Backend integrator ('${currentIntegrator}') not in dropdown.`);
        }
     }
 }

 // handles changes in model or integrator selection dropdowns
 function handleModelSelectionChange(event) {
     const selectElement = event.target;
     const modelType = selectElement.dataset.modelType;
     const selectedId = selectElement.value;
     if (!modelType) return;
     console.log(`UI Selection: ${modelType} = ${selectedId}`);
     if (modelType === 'integrator') {
         sendCommand('select_integrator', { id: selectedId });
     } else {
         sendCommand('select_model', { type: modelType, id: selectedId });
     }
 }

 // applies constraints to parameter values (e.g., min/max)
 function applyConstraints(key, value, allSliderValues) {
     if (!configData?.PARAM_DEFS?.[key]) return value; const pdef = configData.PARAM_DEFS[key]; let cv = Number(value); if (key === 'N') cv = Math.max(10, Math.round(cv)); else if (key === 'mass_max') cv = Math.max(cv, (allSliderValues['mass_min'] ?? pdef.val) + 0.01); else if (key === 'mass_min') { cv = Math.min(cv, (allSliderValues['mass_max'] ?? pdef.val) - 0.01); cv = Math.max(0.01, cv); } else if (['h', 'softening', 'min_temp'].includes(key)) cv = Math.max(1e-3, cv); else if (['cooling_coeff', 'fusion_coeff', 'G'].includes(key)) cv = Math.max(0.0, cv); else if (key === 'bh_theta') cv = Math.max(0.0, Math.min(1.5, cv)); else if (key === 'MAX_NODES_FACTOR') cv = Math.max(1, Math.round(cv)); return Math.max(pdef.min, Math.min(pdef.max, cv));
 }

 // handles input events from parameter sliders
 function handleSliderInput(event) {
     if (!configData || !areDynamicControlsInitialized) return; const slider = event.target; const key = slider.dataset.paramKey;
     const valueText = document.getElementById(`${key}-value`);
     const pdef = configData.PARAM_DEFS[key]; if (!pdef || !valueText) { console.warn(`Missing element for slider ${key}`); return; }
     let value = Number(slider.value); const allSliderValues = {};
     if (key === 'mass_max' || key === 'mass_min') { allSliderValues['mass_min'] = Number(document.getElementById('mass_min-slider')?.value); allSliderValues['mass_max'] = Number(document.getElementById('mass_max-slider')?.value); }
     let constrainedValue = applyConstraints(key, value, allSliderValues);
     if (Math.abs(constrainedValue - value) > (pdef.step / 100 || 1e-9)) { slider.value = constrainedValue; value = constrainedValue; }
     valueText.textContent = formatValue(value, pdef.fmt); if (pdef.live) { sendCommand('set_param', { key: key, value: value }); }
 }

 // handles changes in thermodynamic toggle switches (radiation pressure, cooling, fusion)
 function handleThermoToggles(event) {
     const cb = event.target; const flagKey = cb.id; const isChecked = cb.checked; const backendKeyMap = { 'rad-press-toggle': 'use_rad_press', 'cooling-toggle': 'use_cooling', 'fusion-toggle': 'use_fusion' }; const backendKey = backendKeyMap[flagKey]; if (backendKey) { sendCommand('set_flag', { key: backendKey, value: isChecked }); }
 }

 // centers the 3d view on the particle system
 function handleCenterViewButton() {
    if (!isUIStaticInitialized || !controls || !camera) {
        console.warn("Center view: Static UI or controls not ready.");
        return;
    }
    let targetPosition = new THREE.Vector3(0, 0, 0);

    if (currentSimData?.N > 0 && particleSystem?.geometry?.attributes?.position?.array) {
        const positions = particleSystem.geometry.attributes.position.array;
        const center = new THREE.Vector3(0, 0, 0);
        let count = 0;
        const nToSample = Math.min(currentSimData.N, 10000);
        const step = Math.max(1, Math.floor(currentSimData.N / nToSample));

        for (let i = 0; i < currentSimData.N; i += step) {
            if (i * 3 + 2 < positions.length) {
                center.x += positions[i * 3 + 0];
                center.y += positions[i * 3 + 1];
                center.z += positions[i * 3 + 2];
                count++;
            }
        }
        if (count > 0) {
            center.divideScalar(count);
            targetPosition.copy(center);
        }
    }
    controls.target.copy(targetPosition);
    controls.update();
}

// handles the restart simulation button click
function handleRestartButton() {
    if (!configData || !areDynamicControlsInitialized) { alert("UI not fully initialized or Configuration not loaded."); return; }
    const settings = {}; const allSliderValues = {}; let settingsOk = true;
    for (const key in configData.PARAM_DEFS) {
        const slider = document.getElementById(`${key}-slider`);
        if (!slider) { alert(`Missing slider element for Restart: #${key}-slider`); settingsOk = false; break; }
        allSliderValues[key] = Number(slider.value);
    }
    if (!settingsOk) return;

    for (const key in configData.PARAM_DEFS) {
        const pdef = configData.PARAM_DEFS[key];
        let value = allSliderValues[key];
        let constrainedValue = applyConstraints(key, value, allSliderValues);
        const sliderElem = document.getElementById(`${key}-slider`);
        const valueElem = document.getElementById(`${key}-value`);
        if (sliderElem && valueElem && Math.abs(constrainedValue - value) > (pdef.step / 100 || 1e-9)) {
            sliderElem.value = constrainedValue;
            valueElem.textContent = formatValue(constrainedValue, pdef.fmt);
            allSliderValues[key] = constrainedValue;
        }
        settings[key] = constrainedValue;
    }
    const radToggle = document.getElementById('rad-press-toggle'); const coolToggle = document.getElementById('cooling-toggle'); const fusToggle = document.getElementById('fusion-toggle');
    const gravSelect = document.getElementById('gravity-select'); const sphSelect = document.getElementById('sph-select'); const thermSelect = document.getElementById('thermo-select'); const colorSelect = document.getElementById('color-select'); const intSelect = document.getElementById('integrator-select');
    if (!radToggle || !coolToggle || !fusToggle || !gravSelect || !sphSelect || !thermSelect || !colorSelect || !intSelect) { alert("One or more control elements are missing for Restart!"); return; }
    settings.use_rad_press = radToggle.checked; settings.use_cooling = coolToggle.checked; settings.use_fusion = fusToggle.checked;
    const gravity_id = gravSelect.value; const sph_id = sphSelect.value; const thermo_id = thermSelect.value; const color_id = colorSelect.value; const integrator_id = intSelect.value;
    sendCommand('restart', { settings, gravity: gravity_id, sph: sph_id, thermo: thermo_id, color: color_id, integrator: integrator_id });
}

// updates the feedback ui elements with current simulation data
function updateFeedbackUI(data) {
    if (!isUIStaticInitialized) return;
    const setText = (id, value, fmt) => {
        const elem = document.getElementById(id);
        if (elem) elem.textContent = (value !== undefined && value !== null) ? formatValue(value, fmt) : 'N/A';
    };
    setText('status-text', data.status_msg);
    setText('time-text', data.time, '{:.3f}');
    setText('steps-text', data.steps_taken, '{:d}');
    setText('n-particles-text', data.N, '{:d}');
    const stats = data.stats || {};
    setText('ke-text', stats.avg_KE, '{:.2e}');
    setText('vel-text', stats.avg_vel, '{:.2f}');
    setText('core-text', stats.core_c, '{:d}');
    setText('density-text', stats.max_rho, '{:.2e}');
    setText('temp-text', stats.max_T, '{:.2e}');
    setText('cool-text', stats.avg_cool, '{:.1e}');
    setText('fusion-text', stats.avg_fus, '{:.1e}');
}

 // populates checkboxes for graph plotting options based on config data
function populateGraphCheckboxes() {
    console.log("Populating graph checkboxes...");
    if (!configData?.DEFAULT_SETTINGS?.GRAPH_SETTINGS) {
        console.error("Cannot populate graph checkboxes: Missing configData.DEFAULT_SETTINGS.GRAPH_SETTINGS.");
        return;
    }
    const graphContainer = document.getElementById('graph-controls-container');
    if (!graphContainer) {
        console.error("CRITICAL: Graph container '#graph-controls-container' not found!");
        return;
    }
    graphContainer.innerHTML = '';

    const graphSettings = configData.DEFAULT_SETTINGS.GRAPH_SETTINGS;
    let count = 0;

    const formGroup = document.createElement('div');
    formGroup.style.maxHeight = '200px';
    formGroup.style.overflowY = 'auto';
    formGroup.style.border = '1px solid #444';
    formGroup.style.padding = '5px';
    formGroup.style.marginBottom = '10px';
    formGroup.style.display = 'grid';
    formGroup.style.gridTemplateColumns = '1fr 1fr';
    formGroup.style.gap = '0px 10px';

    for (const key in graphSettings) {
        if (key.startsWith('plot_')) {
            count++;
            const isChecked = graphSettings[key];

            const label = document.createElement('label');
            label.style.display = 'block';
            label.style.marginBottom = '2px';
            label.style.whiteSpace = 'nowrap';
            label.style.overflow = 'hidden';
            label.style.textOverflow = 'ellipsis';
            label.style.fontSize = '0.85rem';
            label.title = formatGraphLabel(key);

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `graph-toggle-${key}`;
            checkbox.name = key;
            checkbox.checked = isChecked;
            checkbox.dataset.graphKey = key;
            checkbox.style.marginRight = '5px';
            checkbox.style.verticalAlign = 'middle';

            checkbox.addEventListener('change', (event) => {
                console.log(`Graph toggle changed: ${event.target.name} = ${event.target.checked}`);
            });

            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(formatGraphLabel(key)));
            formGroup.appendChild(label);
        }
    }

    if (count === 0) {
         graphContainer.innerHTML = '<p style="text-align: center; color: #888;">No graph options found in config.</p>';
    } else {
        graphContainer.appendChild(formGroup);
        console.log(`Populated ${count} graph checkboxes.`);
    }
}

// formats graph checkbox labels from underscore_case to title case
function formatGraphLabel(key) {
    return key.replace('plot_', '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

 // handles the "generate pdf" button click, requesting a pdf report from the backend
async function handleGeneratePdf() {
    const button = document.getElementById('generate-pdf-button');
    const feedback = document.getElementById('pdf-feedback-message');
    if (!button || !feedback) return;

    button.disabled = true;
    feedback.textContent = 'Generating PDF...';
    feedback.style.color = 'orange';

    try {
        const response = await fetch(`${SERVER_URL}/api/generate_pdf`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });

        const result = await response.json();

        if (response.ok && result.success) {
            feedback.textContent = `Success! ${result.message || ''}`;
            feedback.style.color = 'limegreen';
        } else {
            feedback.textContent = `Error: ${result.message || 'Unknown error'}`;
            feedback.style.color = '#f66';
            console.error("PDF Generation Error:", result);
        }
    } catch (error) {
        feedback.textContent = `Error: ${error.message}`;
        feedback.style.color = '#f66';
        console.error("PDF Generation Network/Fetch Error:", error);
    } finally {
        button.disabled = false;
    }
}

// initializes the three.js scene, camera, renderer, and controls
function initThreeJS(container) {
    console.log("Initializing Three.js Scene...");
    const loadingIndicator = document.getElementById('loading-indicator');
    if (!container) { console.error("initThreeJS: Container argument missing!"); throw new Error("Missing scene container"); }

    while (container.firstChild) { container.removeChild(container.firstChild); }
    if (loadingIndicator) { container.appendChild(loadingIndicator); loadingIndicator.style.display = 'block'; }

    try {
         scene = new THREE.Scene(); scene.background = new THREE.Color(0x080810);
         camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.01, 500); camera.position.set(0, 15, 35); camera.up.set(0, 1, 0);
         renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" }); renderer.setSize(container.clientWidth, container.clientHeight); renderer.setPixelRatio(window.devicePixelRatio); container.appendChild(renderer.domElement);
         const ambientLight = new THREE.AmbientLight(0x505055); scene.add(ambientLight);
         const dirLight = new THREE.DirectionalLight(0xffffff, 1.0); dirLight.position.set(8, 15, 10); scene.add(dirLight);
         controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.dampingFactor = 0.07; controls.screenSpacePanning = false; controls.target.set(0, 0, 0); controls.maxDistance = 200; controls.update();
         particleSystem = null;
         window.addEventListener('resize', () => { if (!container || !renderer || !camera) return; camera.aspect = container.clientWidth / container.clientHeight; camera.updateProjectionMatrix(); renderer.setSize(container.clientWidth, container.clientHeight); });
         if (animationFrameId === null) { animate(); }
         console.log("Three.js Initialized.");
         if (loadingIndicator) loadingIndicator.style.display = 'none';
     } catch(e) { console.error("Error setting up Three.js:", e); displayFatalError("Failed to initialize 3D scene."); throw e; }
}

// calculates an appropriate particle size for rendering
function calculateParticleSize(numParticles, baseSize = 20.0, minSize = 0.01, maxSize = 0.3) { if (numParticles <= 0) return baseSize * 0.02; let size = baseSize / Math.sqrt(numParticles + 1.0); size = Math.max(minSize, Math.min(maxSize, size)); return size; }
// initializes the three.js particle system object
function initializeParticleSystem(numParticles) { if (!scene) return; if (numParticles <= 0) return; if (particleSystem) { scene.remove(particleSystem); particlesGeometry?.dispose(); particlesMaterial?.dispose(); } console.log(`Initializing particle system for ${numParticles}...`); particlesGeometry = new THREE.BufferGeometry(); particlesGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(numParticles * 3), 3)); particlesGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(numParticles * 3), 3)); const estRadius = configData?.DEFAULT_SETTINGS?.radius ?? 20.0; particlesGeometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), estRadius * 1.5); const pointTexture = createCircleTexture(32); particlesMaterial = new THREE.PointsMaterial({ size: calculateParticleSize(numParticles), vertexColors: true, sizeAttenuation: true, map: pointTexture, transparent: false, alphaTest: 0.1, blending: THREE.NormalBlending, depthWrite: true, depthTest: true, }); particleSystem = new THREE.Points(particlesGeometry, particlesMaterial); particleSystem.visible = true; scene.add(particleSystem); }
// resizes the particle system buffers if the number of particles changes
function resizeParticleSystem(newNumParticles) { if (!scene) return; if (!particlesGeometry || !particlesMaterial) { initializeParticleSystem(newNumParticles); return; } if (newNumParticles <= 0) { if (particleSystem) particleSystem.visible = false; return; } if (particleSystem) particleSystem.visible = true; particlesMaterial.size = calculateParticleSize(newNumParticles); particlesGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(newNumParticles * 3), 3)); particlesGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(newNumParticles * 3), 3)); particlesGeometry.attributes.color.array.fill(0.5); particlesGeometry.setDrawRange(0, newNumParticles); particlesGeometry.computeBoundingSphere(); console.log(`Resized particle system buffers to N = ${newNumParticles}`); }

// main animation loop for rendering the three.js scene
function animate() {
    animationFrameId = requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) { renderer.render(scene, camera); }
}

// main execution starts after dom content is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOMContentLoaded event fired.");
    setTimeout(() => { // slight delay to ensure all dom elements are reliably available
        console.log("setTimeout callback executing...");
        const staticInitSuccess = initializeStaticUI();
        if (staticInitSuccess) {
             console.log("Static UI initialized successfully, connecting WebSocket...");
             connectWebSocket();
        } else {
             console.error("Aborted WebSocket connection due to static UI initialization failure.");
        }
    }, 50);
});