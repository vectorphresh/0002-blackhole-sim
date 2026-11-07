/**
 * Minimal static file server. No dependencies, no build step.
 *
 * Only serves files from the project root, and refuses anything that resolves
 * outside it — a path-traversal check is the one security property a dev
 * server actually needs.
 */

import http from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL(".", import.meta.url)));
const PORT = Number(process.env.PORT ?? 8080);
const HOST = process.env.HOST ?? "127.0.0.1";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".md": "text/markdown; charset=utf-8",
  ".webm": "video/webm",
};

/**
 * Map a request URL to an absolute path inside ROOT, or null if the request
 * escapes the project.
 */
function safePath(urlPath) {
  const decoded = decodeURIComponent(urlPath.split("?")[0]);
  const rel = normalize(decoded === "/" || decoded === "" ? "index.html" : decoded);
  const abs = resolve(join(ROOT, rel));
  if (abs !== ROOT && !abs.startsWith(ROOT + "/")) return null;
  return abs;
}

const server = http.createServer(async (req, res) => {
  const abs = safePath(req.url ?? "/");
  if (!abs) {
    res.writeHead(403).end("Forbidden");
    return;
  }

  try {
    const info = await stat(abs);
    if (info.isDirectory()) {
      res.writeHead(403).end("Forbidden");
      return;
    }
    const body = await readFile(abs);
    res.writeHead(200, {
      "Content-Type": MIME[extname(abs).toLowerCase()] ?? "application/octet-stream",
      "Content-Length": body.length,
      // Never cache: a stale shader is a confusing debugging session.
      "Cache-Control": "no-store",
    });
    res.end(body);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain" }).end("404 Not Found");
  }
});

server.listen(PORT, HOST, () => {
  console.log(`Kerr black hole simulation → http://${HOST}:${PORT}/`);
  console.log(`Serving ${ROOT}`);
});
