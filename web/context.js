/**
 * Session context store and circular turn gauge widget.
 */

const STORAGE_KEY = "comfyui_code2video_sessions";

export const SessionContext = {
    load() {
        try {
            return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
        } catch {
            return {};
        }
    },

    save(data) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
        } catch (e) {
            console.warn("Could not save session context:", e);
        }
    },

    setSession(sessionId, info) {
        const store = this.load();
        store[sessionId] = { ...(store[sessionId] || {}), ...info, lastUpdated: Date.now() };
        this.save(store);
    },

    getSession(sessionId) {
        const store = this.load();
        return store[sessionId] || null;
    }
};

export function createCircularGauge(currentTurns, maxTurns) {
    const size = 28;
    const stroke = 3;
    const radius = (size - stroke) / 2;
    const circumference = 2 * Math.PI * radius;
    const progress = Math.min(1.0, currentTurns / Math.max(1, maxTurns));
    const offset = circumference - progress * circumference;

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", String(size));
    svg.setAttribute("height", String(size));
    svg.style.transform = "rotate(-90deg)";

    svg.innerHTML = `
        <circle cx="${size / 2}" cy="${size / 2}" r="${radius}"
                stroke="#3f3f46" stroke-width="${stroke}" fill="transparent" />
        <circle cx="${size / 2}" cy="${size / 2}" r="${radius}"
                stroke="#6366f1" stroke-width="${stroke}" fill="transparent"
                stroke-dasharray="${circumference}" stroke-dashoffset="${offset}"
                stroke-linecap="round" />
    `;
    return svg;
}
