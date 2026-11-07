/**
 * Application entry point: wires the renderer, camera, controls, HUD, and
 * capture together.
 *
 * The render loop is deliberately lazy — a frame is traced only when something
 * changed or accumulation is still improving the image. That keeps an idle tab
 * at zero GPU cost and lets progressive accumulation settle without a timer.
 */

import { Renderer } from "./gl/renderer.js";
import { createCamera, orbit, zoom, rayDirection, basisMatrix } from "./core/camera.js";
import { defaultParams, derive, hudRows, si, duration } from "./core/state.js";
import { horizonRadius } from "./core/kerr.js";
import { Capture } from "./gl/capture.js";
import { buildPanel } from "./ui/panel.js";

const canvas = document.getElementById("gl");
const hudEl = document.getElementById("hud");
const panelEl = document.getElementById("panel");
const statusEl = document.getElementById("status");

const params = defaultParams();
const camera = createCamera({
  r: params.camR,
  theta: params.camTheta,
  phi: params.camPhi,
  fov: (params.fovDeg * Math.PI) / 180,
});

let renderer;
try {
  renderer = new Renderer(canvas);
} catch (err) {
  statusEl.textContent = String(err.message || err);
  statusEl.classList.add("error");
  throw err;
}

const capture = new Capture(renderer, canvas);

/** Anything that changes the traced image resets the running average. */
let dirty = true;
function invalidate() {
  dirty = true;
  renderer.resetAccumulation();
}

/* ------------------------------------------------------------------ *
 * Canvas sizing
 * ------------------------------------------------------------------ */

function resize() {
  const scale = params.renderScale;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr * scale));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr * scale));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  renderer.resize(w, h);
}

new ResizeObserver(resize).observe(canvas);
resize();

/* ------------------------------------------------------------------ *
 * Camera interaction
 * ------------------------------------------------------------------ */

let dragging = false;
let lastX = 0;
let lastY = 0;

canvas.addEventListener("pointerdown", (e) => {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener("pointerup", (e) => {
  dragging = false;
  try {
    canvas.releasePointerCapture(e.pointerId);
  } catch {
    /* pointer already released */
  }
});

canvas.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;
  orbit(camera, -dy * 0.005, dx * 0.005);
  params.camTheta = camera.theta;
  params.camPhi = camera.phi;
  invalidate();
});

canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    zoom(camera, Math.exp(e.deltaY * 0.0015), params.spin);
    params.camR = camera.r;
    panel.sync();
    invalidate();
  },
  { passive: false },
);

window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement) return;
  const step = e.shiftKey ? 0.05 : 0.01;
  let handled = true;
  switch (e.key) {
    case "ArrowLeft": orbit(camera, 0, -step); break;
    case "ArrowRight": orbit(camera, 0, step); break;
    case "ArrowUp": orbit(camera, -step, 0); break;
    case "ArrowDown": orbit(camera, step, 0); break;
    case "=": case "+": zoom(camera, 0.92, params.spin); break;
    case "-": case "_": zoom(camera, 1.08, params.spin); break;
    case "r": case "R": resetView(); break;
    case "s": case "S": capture.snapshot(); break;
    default: handled = false;
  }
  if (handled) {
    e.preventDefault();
    params.camR = camera.r;
    params.camTheta = camera.theta;
    params.camPhi = camera.phi;
    panel.sync();
    invalidate();
  }
});

function resetView() {
  const d = defaultParams();
  camera.r = d.camR;
  camera.theta = d.camTheta;
  camera.phi = d.camPhi;
  camera.fov = (d.fovDeg * Math.PI) / 180;
  params.camR = camera.r;
  params.camTheta = camera.theta;
  params.camPhi = camera.phi;
  params.fovDeg = d.fovDeg;
  panel.sync();
  invalidate();
}

/* ------------------------------------------------------------------ *
 * Control panel
 * ------------------------------------------------------------------ */

const panel = buildPanel(panelEl, params, {
  onPhysicsChange: () => {
    camera.r = params.camR;
    camera.theta = params.camTheta;
    camera.phi = params.camPhi;
    camera.fov = (params.fovDeg * Math.PI) / 180;
    // Clamping in zoom() can move r; keep the panel honest about it.
    params.camR = camera.r;
    // resize() is a no-op unless the dimensions actually changed, so it is
    // safe to call on every parameter change.
    resize();
    invalidate();
  },
  // Exposure, bloom, and vignette only affect the post pass, so they must not
  // throw away accumulated history.
  onGradeChange: () => {
    dirty = true;
  },
  onReset: resetView,
  onSnapshot: () => capture.snapshot(),
  onRecord: () => capture.toggleRecording(),
});

/* ------------------------------------------------------------------ *
 * HUD
 * ------------------------------------------------------------------ */

let hudTimer = 0;
function updateHud(derived, traceFps, traced) {
  const now = performance.now();
  if (now - hudTimer < 250) return;
  hudTimer = now;

  const rows = hudRows(params, derived);
  hudEl.innerHTML = rows
    .map(([k, v]) => `<div class="row"><span class="k">${k}</span><span class="v">${v}</span></div>`)
    .join("");

  const accum = params.accumulate ? `${Math.min(renderer.accumIndex, params.maxAccumFrames)}` : "off";
  // The trace rate, not the requestAnimationFrame rate: the loop idles at 60 Hz
  // once accumulation converges, and reporting that as fps would imply the
  // integrator is keeping up when it has simply stopped being asked to work.
  const fpsText = traceFps === null ? "idle" : `${traceFps.toFixed(0)} fps`;
  const stepsText = traced ? `${params.maxSteps} steps/ray` : "settled";
  statusEl.innerHTML =
    `<span>${fpsText}</span>` +
    `<span>${stepsText}</span>` +
    `<span>accum ${accum}</span>` +
    `<span>${renderer.width}×${renderer.height}</span>`;
}

/* ------------------------------------------------------------------ *
 * Render loop
 * ------------------------------------------------------------------ */

/** Rate at which frames are actually *traced*, or null when the loop is idle. */
let traceFps = null;
let traceWindowStart = performance.now();
let traceWindowCount = 0;
/** Whether the previous loop iteration traced a frame. Used to detect the
 *  idle -> tracing transition, where the window start is stale and would
 *  otherwise measure the idle gap and report a bogus sub-1 fps. */
let wasTracing = false;

function frame() {
  const now = performance.now();
  const derived = derive(params);
  const needFrame = dirty || (params.accumulate && renderer.accumIndex < params.maxAccumFrames);

  if (needFrame) {
    // Resuming after an idle gap: start a fresh window so the first traced
    // frame is not measured against a stale timestamp.
    if (!wasTracing) {
      traceWindowStart = now;
      traceWindowCount = 0;
    }
    const basis = basisMatrix(camera);
    const scene = {
      spin: params.spin,
      escapeRadius: params.escapeRadius,
      diskInner: derived.diskInner,
      diskOuter: derived.diskOuter,
      diskEnabled: params.diskEnabled,
      diskTemp: params.diskTemperature,
      diskGain: params.diskBrightness,
      step: params.step,
      camPos: [camera.r, camera.theta, camera.phi],
      camBasis: basis,
      tanHalfFov: Math.tan(camera.fov / 2),
      starBrightness: params.starBrightness,
      maxSteps: params.maxSteps,
      diskPrograde: params.diskPrograde,
      accumulate: params.accumulate,
    };
    const grade = {
      exposure: params.exposure,
      bloomStrength: params.bloomStrength,
      bloomThreshold: params.bloomThreshold,
      vignette: params.vignette,
    };
    renderer.render(scene, grade);
    dirty = false;

    // Trace rate over a ~1 s window. The estimate is published from the first
    // frame rather than only once the window closes, so the readout is never
    // blank while the integrator is actually working; averaging over the window
    // keeps it stable when a single frame is expensive.
    traceWindowCount++;
    const span = performance.now() - traceWindowStart;
    if (span > 0) traceFps = (traceWindowCount * 1000) / span;
    if (span >= 1000) {
      traceWindowStart = performance.now();
      traceWindowCount = 0;
    }
    wasTracing = true;
  } else {
    // Idle: accumulation has converged and nothing has changed. Reporting the
    // requestAnimationFrame rate here would imply the integrator is keeping up.
    traceFps = null;
    wasTracing = false;
  }

  updateHud(derived, traceFps, needFrame);
  requestAnimationFrame(frame);
}

panel.sync();
requestAnimationFrame(frame);

// Expose for debugging from the console.
window.kerr = { params, camera, renderer, derive, rayDirection, horizonRadius, si, duration };
