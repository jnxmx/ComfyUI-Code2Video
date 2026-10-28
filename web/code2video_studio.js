/**
 * ComfyUI-Code2Video Studio Extension
 * Interactive preview (window.seek(t)), live scrubbing, turn refinement & tool tracing.
 */

import { attachTagSuggestions } from "./tag_suggestions.js";
import { SessionContext, createCircularGauge } from "./context.js";

// Dynamically inject stylesheet
const link = document.createElement("link");
link.rel = "stylesheet";
link.type = "text/css";
link.href = new URL("./code2video.css", import.meta.url).href;
document.head.appendChild(link);

// Resolve app and api from ComfyUI global or relative import
const app = window.comfyAPI?.app?.app || (await import("../../scripts/app.js")).app;
const api = window.comfyAPI?.api?.api || (await import("../../scripts/api.js")).api;

app.registerExtension({
    name: "SenseNova.Code2VideoStudio",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SenseNovaCode2VideoStudio") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                initStudioNode(this);
                return r;
            };

            // Dynamically manage autogrow input sockets
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, isConnected, link_info) {
                if (onConnectionsChange) onConnectionsChange.apply(this, arguments);
                updateAutogrowSockets(this);
            };
        }
    }
});

function updateAutogrowSockets(node) {
    if (!node.inputs) return;
    const dynamicPrefixes = ["image_", "video_", "audio_"];
    let maxIndices = { image_: 0, video_: 0, audio_: 0 };

    node.inputs.forEach(input => {
        dynamicPrefixes.forEach(prefix => {
            if (input.name.startsWith(prefix)) {
                const num = parseInt(input.name.slice(prefix.length), 10) || 0;
                if (num > maxIndices[prefix]) maxIndices[prefix] = num;
            }
        });
    });

    // Ensure there is always one free socket available for image, video, audio
    dynamicPrefixes.forEach(prefix => {
        const nextIdx = maxIndices[prefix] + 1;
        const expectedName = `${prefix}${nextIdx}`;
        const hasSlot = node.inputs.some(i => i.name === expectedName);
        if (!hasSlot && nextIdx <= 4) {
            const type = prefix === "image_" ? "IMAGE" : (prefix === "video_" ? "IMAGE" : "AUDIO");
            node.addInput(expectedName, type);
        }
    });
}

function initStudioNode(node) {
    let currentSessionId = "";
    let isPlaying = false;
    let animFrameId = null;
    let playbackTime = 0;
    let maxDuration = 5.0;

    // Find prompt and refinement widgets
    const promptWidget = node.widgets?.find(w => w.name === "prompt");
    const refineWidget = node.widgets?.find(w => w.name === "refinement_prompt");
    const sessionWidget = node.widgets?.find(w => w.name === "session_id");

    if (sessionWidget && sessionWidget.value) {
        currentSessionId = sessionWidget.value;
    }

    // Attach @tag suggestions to prompt input
    if (promptWidget?.inputEl) {
        attachTagSuggestions(promptWidget.inputEl, () => getConnectedSocketNames(node));
    }
    if (refineWidget?.inputEl) {
        attachTagSuggestions(refineWidget.inputEl, () => getConnectedSocketNames(node));
    }

    // Create custom DOM widget
    const container = document.createElement("div");
    container.className = "c2v-studio-container";

    container.innerHTML = `
        <div class="c2v-header">
            <div class="c2v-status-badge">
                <span class="c2v-status-dot idle"></span>
                <span class="c2v-status-text">Idle</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div class="c2v-gauge-slot"></div>
                <span class="c2v-session-id">No Session</span>
            </div>
        </div>

        <div class="c2v-tabs">
            <button class="c2v-tab-btn active" data-tab="preview">Live Preview</button>
            <button class="c2v-tab-btn" data-tab="code">Generated Code</button>
            <button class="c2v-tab-btn" data-tab="logs">Execution Logs</button>
        </div>

        <!-- Tab 1: Preview -->
        <div class="c2v-panel active" data-panel="preview">
            <div class="c2v-preview-wrapper">
                <iframe class="c2v-preview-iframe" sandbox="allow-scripts allow-same-origin"></iframe>
            </div>
            <div class="c2v-scrubber-bar">
                <button class="c2v-btn-icon btn-play">▶ Play</button>
                <button class="c2v-btn-icon btn-step-back">-0.1s</button>
                <input type="range" class="c2v-slider" min="0" max="5.0" step="0.033" value="0">
                <button class="c2v-btn-icon btn-step-fwd">+0.1s</button>
                <span class="c2v-time-display">0.00s / 5.00s</span>
            </div>
        </div>

        <!-- Tab 2: Code -->
        <div class="c2v-panel" data-panel="code">
            <pre class="c2v-code-view">// Generated code will appear here after execution...</pre>
        </div>

        <!-- Tab 3: Logs -->
        <div class="c2v-panel" data-panel="logs">
            <div class="c2v-logs-view">
                <div class="c2v-log-entry">Studio initialized. Ready for prompt.</div>
            </div>
        </div>

        <!-- Controls & Refinement -->
        <div class="c2v-controls-bar">
            <button class="c2v-btn-action c2v-btn-pause">Pause</button>
            <input type="text" class="c2v-refine-input" placeholder="Refine animation (e.g. make text larger, speed up camera)...">
            <button class="c2v-btn-action btn-refine">Refine Turn</button>
        </div>
    `;

    // References to DOM elements
    const statusDot = container.querySelector(".c2v-status-dot");
    const statusText = container.querySelector(".c2v-status-text");
    const sessionBadge = container.querySelector(".c2v-session-id");
    const iframe = container.querySelector(".c2v-preview-iframe");
    const slider = container.querySelector(".c2v-slider");
    const timeDisplay = container.querySelector(".c2v-time-display");
    const btnPlay = container.querySelector(".btn-play");
    const btnStepBack = container.querySelector(".btn-step-back");
    const btnStepFwd = container.querySelector(".btn-step-fwd");
    const codeView = container.querySelector(".c2v-code-view");
    const logsView = container.querySelector(".c2v-logs-view");
    const btnPause = container.querySelector(".c2v-btn-pause");
    const refineInput = container.querySelector(".c2v-refine-input");
    const btnRefine = container.querySelector(".btn-refine");

    // Tab switching logic
    const tabBtns = container.querySelectorAll(".c2v-tab-btn");
    const panels = container.querySelectorAll(".c2v-panel");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const tabName = btn.getAttribute("data-tab");
            tabBtns.forEach(b => b.classList.remove("active"));
            panels.forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            container.querySelector(`[data-panel="${tabName}"]`).classList.add("active");
        });
    });

    // Seek helper: calls window.seek(t) inside iframe
    function seekToTime(t) {
        playbackTime = Math.max(0, Math.min(t, maxDuration));
        slider.value = playbackTime;
        timeDisplay.textContent = `${playbackTime.toFixed(2)}s / ${maxDuration.toFixed(2)}s`;
        try {
            if (iframe.contentWindow && typeof iframe.contentWindow.seek === "function") {
                iframe.contentWindow.seek(playbackTime);
            }
        } catch (e) {
            console.debug("seek error:", e);
        }
    }

    // Playback loop
    function togglePlay() {
        isPlaying = !isPlaying;
        btnPlay.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
        if (isPlaying) {
            let lastTs = performance.now();
            function loop(now) {
                if (!isPlaying) return;
                const dt = (now - lastTs) / 1000;
                lastTs = now;
                let nextT = playbackTime + dt;
                if (nextT > maxDuration) nextT = 0;
                seekToTime(nextT);
                animFrameId = requestAnimationFrame(loop);
            }
            animFrameId = requestAnimationFrame(loop);
        } else {
            if (animFrameId) cancelAnimationFrame(animFrameId);
        }
    }

    btnPlay.addEventListener("click", togglePlay);
    btnStepBack.addEventListener("click", () => seekToTime(playbackTime - 0.1));
    btnStepFwd.addEventListener("click", () => seekToTime(playbackTime + 0.1));
    slider.addEventListener("input", (e) => seekToTime(parseFloat(e.target.value)));

    // Pause button handler
    btnPause.addEventListener("click", async () => {
        if (!currentSessionId) return;
        try {
            const res = await fetch("/code2video/control", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: currentSessionId, action: "pause" })
            });
            const data = await res.json();
            if (data.success) {
                statusDot.className = "c2v-status-dot paused";
                statusText.textContent = "Paused";
            }
        } catch (e) {
            console.error("Could not pause session:", e);
        }
    });

    // Refine button handler
    btnRefine.addEventListener("click", () => {
        const text = refineInput.value.trim();
        if (!text) return;
        if (refineWidget) {
            refineWidget.value = text;
        }
        refineInput.value = "";
        // Queue prompt for the next turn
        app.queuePrompt(0, 1);
    });

    // WebSocket event listeners
    api.addEventListener("code2video_status", ({ detail }) => {
        if (detail.session_id !== currentSessionId && currentSessionId) return;
        currentSessionId = detail.session_id || currentSessionId;
        sessionBadge.textContent = currentSessionId.slice(0, 8);
        if (sessionWidget) sessionWidget.value = currentSessionId;

        statusDot.className = `c2v-status-dot ${detail.status}`;
        statusText.textContent = detail.status.charAt(0).toUpperCase() + detail.status.slice(1);

        const entry = document.createElement("div");
        entry.className = "c2v-log-entry";
        entry.textContent = `[Turn ${detail.turn || 1}] Status: ${detail.status}`;
        logsView.appendChild(entry);
        logsView.scrollTop = logsView.scrollHeight;
    });

    api.addEventListener("code2video_token", ({ detail }) => {
        if (detail.session_id !== currentSessionId && currentSessionId) return;
        const entry = document.createElement("div");
        entry.className = "c2v-log-entry";
        entry.textContent = detail.token;
        logsView.appendChild(entry);
        logsView.scrollTop = logsView.scrollHeight;
    });

    api.addEventListener("code2video_tool_executing", ({ detail }) => {
        if (detail.session_id !== currentSessionId && currentSessionId) return;
        const entry = document.createElement("div");
        entry.className = "c2v-log-entry c2v-log-tool";
        entry.textContent = `⚙ Tool ${detail.function}: ${JSON.stringify(detail.params)}`;
        logsView.appendChild(entry);
        logsView.scrollTop = logsView.scrollHeight;
    });

    api.addEventListener("code2video_preview_ready", ({ detail }) => {
        if (detail.session_id !== currentSessionId && currentSessionId) return;
        currentSessionId = detail.session_id;
        sessionBadge.textContent = currentSessionId.slice(0, 8);
        if (sessionWidget) sessionWidget.value = currentSessionId;

        // Set iframe source
        iframe.src = detail.url;
        statusDot.className = "c2v-status-dot ready";
        statusText.textContent = "Preview Ready";

        // Update code view
        if (detail.code) {
            codeView.textContent = detail.code;
        }

        // Wait for iframe load to detect duration and window.seek
        iframe.onload = () => {
            try {
                if (iframe.contentWindow?.DUR) {
                    maxDuration = iframe.contentWindow.DUR;
                    slider.max = maxDuration;
                    seekToTime(0);
                }
            } catch (e) {
                console.debug("Could not read window.DUR from iframe:", e);
            }
        };
    });

    // Register DOM widget with LiteGraph node
    const widget = node.addDOMWidget("code2video_studio", "studio", container, {
        serialize: false,
        hideOnZoom: false,
    });

    // Resize node to comfortably accommodate preview and controls
    node.setSize([540, 520]);
    return widget;
}

function getConnectedSocketNames(node) {
    const list = ["@image_1", "@image_2", "@video_1", "@audio_1"];
    if (node.inputs) {
        node.inputs.forEach(i => {
            if (i.link && i.name) {
                list.push(`@${i.name}`);
            }
        });
    }
    return Array.from(new Set(list));
}
