/**
 * Frame capture: PNG snapshots and WebM video.
 *
 * Both read the canvas directly. The renderer requests
 * `preserveDrawingBuffer: true` so the colour buffer is still readable after
 * the compositor has presented it — without that, `toDataURL` and
 * `drawImage` would see a cleared buffer.
 */

export class Capture {
  /**
   * @param {import("./renderer.js").Renderer} renderer
   * @param {HTMLCanvasElement} canvas
   */
  constructor(renderer, canvas) {
    this.renderer = renderer;
    this.canvas = canvas;
    this.recorder = null;
    this.chunks = [];
    this.recording = false;
    /** Scratch canvas used to composite the scaled frame for recording. */
    this._scratch = document.createElement("canvas");
  }

  /** Timestamped filename stem, e.g. `kerr-2025-01-04T12-33-07`. */
  static stem() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return (
      `kerr-${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
      `T${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`
    );
  }

  static download(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    // Revoke on the next tick so the download has started.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /** Save the current frame as PNG. */
  snapshot() {
    this.canvas.toBlob((blob) => {
      if (!blob) return;
      Capture.download(blob, `${Capture.stem()}.png`);
    }, "image/png");
  }

  /**
   * Start or stop a WebM recording.
   *
   * `captureStream` taps the canvas at the frame rate the browser is already
   * presenting, so recording adds no extra GPU work beyond the encode.
   *
   * @returns {boolean} the new recording state
   */
  toggleRecording() {
    if (this.recording) {
      this.recorder.stop();
      return (this.recording = false);
    }

    if (!this.canvas.captureStream) {
      alert("Canvas recording is not supported in this browser.");
      return false;
    }

    const codecs = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
    ];
    const mimeType = codecs.find((c) => MediaRecorder.isTypeSupported(c));
    if (!mimeType) {
      alert("No supported WebM codec for MediaRecorder.");
      return false;
    }

    this.chunks = [];
    const stream = this.canvas.captureStream(60);
    this.recorder = new MediaRecorder(stream, {
      mimeType,
      videoBitsPerSecond: 12_000_000,
    });
    this.recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) this.chunks.push(e.data);
    };
    this.recorder.onstop = () => {
      const blob = new Blob(this.chunks, { type: mimeType });
      Capture.download(blob, `${Capture.stem()}.webm`);
      this.chunks = [];
    };
    this.recorder.start();
    this.recording = true;
    return true;
  }
}
