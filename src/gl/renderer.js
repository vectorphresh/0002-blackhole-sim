/**
 * WebGL2 renderer.
 *
 * Two passes:
 *   1. trace — one null geodesic per pixel into an RGBA16F HDR target,
 *      ping-ponged so each frame can blend into the running average
 *   2. post  — bloom + exposure + ACES tonemap + vignette to the canvas
 *
 * The fullscreen triangle is drawn with no vertex buffer at all; the two
 * required vertices come from gl_VertexID.
 */

import { VERTEX, TRACE, POST } from "./shaders.js";

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    // Surface the offending line: GLSL line numbers count from 1 at the top of
    // the concatenated source, which is easy to misread for the trace shader.
    const line = /ERROR:\s*\d+:(\d+)/.exec(log);
    let context = "";
    if (line) {
      const n = Number(line[1]);
      context =
        "\n  near line " +
        n +
        ":\n" +
        source
          .split("\n")
          .slice(Math.max(0, n - 3), n + 2)
          .map((l, i) => `${Math.max(0, n - 2) + i}: ${l}`)
          .join("\n");
    }
    throw new Error(`Shader compile failed:\n${log}${context}`);
  }
  return shader;
}

function link(gl, sources) {
  const [vsSrc, fsSrc] = sources;
  const vs = compile(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`Program link failed: ${gl.getProgramInfoLog(program)}`);
  }
  return program;
}

/** Cache every uniform location on a program once. */
function uniforms(gl, program, names) {
  const out = {};
  for (const name of names) out[name] = gl.getUniformLocation(program, name);
  return out;
}

/**
 * @typedef {Object} RenderTarget
 * @property {WebGLFramebuffer} fbo
 * @property {WebGLTexture} tex
 * @property {number} width
 * @property {number} height
 */

function createTarget(gl, width, height, internalFormat, format, type, filter) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error(`Framebuffer incomplete: 0x${status.toString(16)}`);
  }
  return { fbo, tex, width, height };
}

function destroyTarget(gl, t) {
  if (!t) return;
  gl.deleteFramebuffer(t.fbo);
  gl.deleteTexture(t.tex);
}

export class Renderer {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", {
      antialias: false, // we do our own sampling; MSAA on a ray tracer is wasted
      depth: false,
      stencil: false,
      alpha: false,
      powerPreference: "high-performance",
      preserveDrawingBuffer: true, // needed for direct canvas capture
    });
    if (!gl) throw new Error("WebGL2 is not available in this browser.");
    this.gl = gl;

    // RGBA16F colour attachments are an extension in WebGL2, not core.
    if (!gl.getExtension("EXT_color_buffer_float")) {
      throw new Error("EXT_color_buffer_float is required for HDR accumulation.");
    }

    this.traceProgram = link(gl, [VERTEX, TRACE]);
    this.postProgram = link(gl, [VERTEX, POST]);

    this.traceU = uniforms(gl, this.traceProgram, [
      "uSpin", "uEscapeRadius", "uDiskInner", "uDiskOuter", "uDiskEnabled",
      "uDiskTemp", "uDiskGain", "uStep",
      "uCamPos", "uCamBasis", "uResolution", "uTanHalfFov", "uAccumIndex",
      "uStarBrightness", "uMaxSteps", "uDiskPrograde", "uPrevAccum",
    ]);
    this.postU = uniforms(gl, this.postProgram, [
      "uScene", "uExposure", "uBloomStrength", "uBloomThreshold", "uVignette",
    ]);

    this.vao = gl.createVertexArray();
    this.width = 0;
    this.height = 0;
    this.accumIndex = 0;
    this._ping = 0;
  }

  /**
   * Resize the HDR accumulation targets. Any change in size invalidates the
   * running average, so accumulation restarts.
   */
  resize(width, height) {
    const gl = this.gl;
    if (width === this.width && height === this.height) return false;
    destroyTarget(gl, this.accumA);
    destroyTarget(gl, this.accumB);
    this.accumA = createTarget(gl, width, height, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    this.accumB = createTarget(gl, width, height, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    for (const t of [this.accumA, this.accumB]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.width = width;
    this.height = height;
    this.resetAccumulation();
    return true;
  }

  /** Discard the running average; call whenever a parameter changes. */
  resetAccumulation() {
    this.accumIndex = 0;
    this._ping = 0;
    const gl = this.gl;
    for (const t of [this.accumA, this.accumB]) {
      if (!t) continue;
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * @param {Object} scene trace-pass parameters
   * @param {Object} grade post-pass parameters
   */
  render(scene, grade) {
    const gl = this.gl;
    if (!this.accumA) throw new Error("Renderer.resize must be called before render.");

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(this.vao);

    // --- trace pass -----------------------------------------------------
    const src = this._ping === 0 ? this.accumA : this.accumB;
    const dst = this._ping === 0 ? this.accumB : this.accumA;

    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
    gl.viewport(0, 0, this.width, this.height);
    gl.useProgram(this.traceProgram);

    const u = this.traceU;
    gl.uniform1f(u.uSpin, scene.spin);
    gl.uniform1f(u.uEscapeRadius, scene.escapeRadius);
    gl.uniform1f(u.uDiskInner, scene.diskInner);
    gl.uniform1f(u.uDiskOuter, scene.diskOuter);
    gl.uniform1f(u.uDiskEnabled, scene.diskEnabled ? 1 : 0);
    gl.uniform1f(u.uDiskTemp, scene.diskTemp);
    gl.uniform1f(u.uDiskGain, scene.diskGain);
    gl.uniform1f(u.uStep, scene.step);
    gl.uniform3f(u.uCamPos, scene.camPos[0], scene.camPos[1], scene.camPos[2]);
    gl.uniformMatrix3fv(u.uCamBasis, false, scene.camBasis);
    gl.uniform2f(u.uResolution, this.width, this.height);
    gl.uniform1f(u.uTanHalfFov, scene.tanHalfFov);
    // With accumulation disabled the frame stands alone, so the shader must
    // not blend against history.
    gl.uniform1f(u.uAccumIndex, scene.accumulate ? this.accumIndex : 0);
    gl.uniform1f(u.uStarBrightness, scene.starBrightness);
    gl.uniform1i(u.uMaxSteps, scene.maxSteps);
    gl.uniform1i(u.uDiskPrograde, scene.diskPrograde);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, src.tex);
    gl.uniform1i(u.uPrevAccum, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // --- post pass ------------------------------------------------------
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.postProgram);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, dst.tex);
    gl.uniform1i(this.postU.uScene, 0);
    gl.uniform1f(this.postU.uExposure, grade.exposure);
    gl.uniform1f(this.postU.uBloomStrength, grade.bloomStrength);
    gl.uniform1f(this.postU.uBloomThreshold, grade.bloomThreshold);
    gl.uniform1f(this.postU.uVignette, grade.vignette);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // The frame just written is the next frame's history.
    this._ping ^= 1;
    this.accumIndex += 1;
    return dst;
  }

  /** The most recently traced HDR target, for direct capture. */
  get lastTarget() {
    return this._ping === 0 ? this.accumB : this.accumA;
  }

  dispose() {
    const gl = this.gl;
    destroyTarget(gl, this.accumA);
    destroyTarget(gl, this.accumB);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.traceProgram);
    gl.deleteProgram(this.postProgram);
  }
}
