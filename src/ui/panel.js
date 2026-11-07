/**
 * Control panel.
 *
 * Declarative: the layout is a list of field descriptors, and each descriptor
 * knows how to read from and write to `params`. `sync()` pushes state into the
 * DOM, which is the only way the panel can ever disagree with the simulation —
 * the wheel handler and keyboard shortcuts mutate the camera, then call sync().
 */

/**
 * @typedef {Object} Field
 * @property {string} key param key
 * @property {string} label
 * @property {number} [min]
 * @property {number} [max]
 * @property {number} [step]
 * @property {string} [unit]
 * @property {boolean} [isBool]
 * @property {boolean} [physics] changing it invalidates accumulated history
 * @property {(v:number)=>string} [fmt]
 */

/** @type {{group: string, fields: Field[]}[]} */
const LAYOUT = [
  {
    group: "Spacetime",
    fields: [
      { key: "spin", label: "Spin a/M", min: 0, max: 0.998, step: 0.001, physics: true, fmt: (v) => v.toFixed(3) },
      { key: "solarMasses", label: "Mass", min: 1, max: 1e6, step: 1, unit: "M☉", log: true },
    ],
  },
  {
    group: "Accretion disk",
    fields: [
      { key: "diskEnabled", label: "Enabled", isBool: true, physics: true },
      { key: "diskPrograde", label: "Prograde", isBool: true, physics: true, toBool: (v) => v > 0, fromBool: (b) => (b ? 1 : -1) },
      { key: "diskInnerScale", label: "Inner edge ×ISCO", min: 1, max: 6, step: 0.01, physics: true, fmt: (v) => v.toFixed(2) },
      { key: "diskOuter", label: "Outer edge", min: 6, max: 60, step: 0.5, unit: "M", physics: true, fmt: (v) => v.toFixed(1) },
      { key: "diskTemperature", label: "Temperature", min: 0.3, max: 3, step: 0.01, physics: true, fmt: (v) => v.toFixed(2) },
      { key: "diskBrightness", label: "Brightness", min: 0.02, max: 3, step: 0.01, physics: true, fmt: (v) => v.toFixed(2) },
    ],
  },
  {
    group: "Camera",
    fields: [
      { key: "camR", label: "Radius", min: 2.5, max: 120, step: 0.05, unit: "M", physics: true, fmt: (v) => v.toFixed(2) },
      { key: "camTheta", label: "Polar angle", min: 0.05, max: 3.09, step: 0.005, physics: true, fmt: (v) => `${(v * 180 / Math.PI).toFixed(1)}°` },
      { key: "camPhi", label: "Azimuth", min: -Math.PI, max: Math.PI, step: 0.005, physics: true, fmt: (v) => `${(v * 180 / Math.PI).toFixed(1)}°` },
      { key: "fovDeg", label: "Field of view", min: 15, max: 110, step: 0.5, unit: "°", physics: true, fmt: (v) => v.toFixed(0) },
    ],
  },
  {
    group: "Integration",
    fields: [
      { key: "step", label: "Step tolerance", min: 0.002, max: 0.12, step: 0.001, physics: true, fmt: (v) => v.toFixed(3) },
      { key: "maxSteps", label: "Max steps/ray", min: 200, max: 4095, step: 10, physics: true, fmt: (v) => v.toFixed(0) },
      { key: "escapeRadius", label: "Escape radius", min: 30, max: 400, step: 1, unit: "M", physics: true, fmt: (v) => v.toFixed(0) },
    ],
  },
  {
    group: "Image",
    fields: [
      { key: "exposure", label: "Exposure", min: 0.1, max: 6, step: 0.01, fmt: (v) => v.toFixed(2) },
      { key: "bloomStrength", label: "Bloom", min: 0, max: 2, step: 0.01, fmt: (v) => v.toFixed(2) },
      { key: "bloomThreshold", label: "Bloom threshold", min: 0, max: 3, step: 0.01, fmt: (v) => v.toFixed(2) },
      { key: "vignette", label: "Vignette", min: 0, max: 1.5, step: 0.01, fmt: (v) => v.toFixed(2) },
      { key: "starBrightness", label: "Stars", min: 0, max: 3, step: 0.01, physics: true, fmt: (v) => v.toFixed(2) },
    ],
  },
  {
    group: "Performance",
    fields: [
      { key: "renderScale", label: "Render scale", min: 0.25, max: 1, step: 0.05, physics: true, fmt: (v) => `${(v * 100).toFixed(0)}%` },
      { key: "accumulate", label: "Progressive", isBool: true },
      { key: "maxAccumFrames", label: "Max accum", min: 1, max: 1000, step: 1, fmt: (v) => v.toFixed(0) },
    ],
  },
];

/**
 * Build the panel.
 *
 * @param {HTMLElement} root
 * @param {Object} params mutated in place
 * @param {{onPhysicsChange:()=>void, onGradeChange:()=>void, onReset:()=>void, onSnapshot:()=>void, onRecord:()=>void}} handlers
 * @returns {{sync:()=>void}}
 */
export function buildPanel(root, params, handlers) {
  /** @type {Map<string, {update:()=>void}>} */
  const controls = new Map();

  for (const section of LAYOUT) {
    const details = document.createElement("details");
    details.open = true;
    const summary = document.createElement("summary");
    summary.textContent = section.group;
    details.appendChild(summary);

    for (const field of section.fields) {
      const row = document.createElement("div");
      row.className = "field";

      if (field.isBool) {
        const label = document.createElement("label");
        label.className = "check";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = field.toBool ? field.toBool(params[field.key]) : Boolean(params[field.key]);
        label.append(input, document.createTextNode(field.label));
        row.appendChild(label);

        input.addEventListener("change", () => {
          const value = field.fromBool ? field.fromBool(input.checked) : input.checked;
          params[field.key] = value;
          notify(field);
        });
        controls.set(field.key, {
          update: () => {
            input.checked = field.toBool ? field.toBool(params[field.key]) : Boolean(params[field.key]);
          },
        });
      } else {
        const label = document.createElement("label");
        const name = document.createElement("span");
        name.className = "name";
        name.textContent = field.label;
        const value = document.createElement("span");
        value.className = "value";
        label.append(name, value);

        const input = document.createElement("input");
        input.type = "range";
        input.min = String(field.min);
        input.max = String(field.max);
        input.step = String(field.step);
        input.value = String(params[field.key]);

        row.append(label, input);

        const render = () => {
          const v = params[field.key];
          value.textContent = (field.fmt ? field.fmt(v) : v.toFixed(2)) + (field.unit ? ` ${field.unit}` : "");
        };
        render();

        input.addEventListener("input", () => {
          params[field.key] = Number(input.value);
          render();
          notify(field);
        });

        controls.set(field.key, {
          update: () => {
            input.value = String(params[field.key]);
            render();
          },
        });
      }

      details.appendChild(row);
    }
    root.appendChild(details);
  }

  function notify(field) {
    if (field.physics) handlers.onPhysicsChange();
    else handlers.onGradeChange();
  }

  // Actions
  const actions = document.createElement("div");
  actions.className = "actions";
  const mk = (label, fn, cls = "") => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    if (cls) b.className = cls;
    b.addEventListener("click", fn);
    actions.appendChild(b);
    return b;
  };
  const recordBtn = mk("● Record", () => {
    const on = handlers.onRecord();
    recordBtn.classList.toggle("active", on);
    recordBtn.textContent = on ? "■ Stop" : "● Record";
  });
  mk("PNG", () => handlers.onSnapshot());
  mk("Reset view", () => handlers.onReset());
  root.appendChild(actions);

  return {
    sync() {
      for (const c of controls.values()) c.update();
    },
  };
}
