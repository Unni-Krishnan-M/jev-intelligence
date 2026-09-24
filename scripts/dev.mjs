// One command for local development: starts the FastAPI backend and the Next.js frontend together,
// prefixes their output, and stops both when either exits or on Ctrl+C.
// Usage: `npm run dev` from the repository root. Needs uv and pnpm on PATH; no npm packages.

import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { connect } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const API_PORT = process.env.JEV_API_PORT ?? "8000";
const WEB_PORT = process.env.JEV_WEB_PORT ?? "3000";
const isWin = process.platform === "win32";

function need(cmd, hint) {
  const probe = spawnSync(cmd, ["--version"], { shell: isWin, stdio: "ignore" });
  if (probe.error || probe.status !== 0) {
    console.error(`[dev] '${cmd}' not found on PATH. ${hint}`);
    process.exit(1);
  }
}
need("uv", "Install it from https://docs.astral.sh/uv/");
need("pnpm", "Install it with `npm install -g pnpm` (or `corepack enable`).");

// fail fast with a clear message instead of a half-started stack
function portInUse(port) {
  return new Promise((resolve) => {
    const sock = connect({ host: "127.0.0.1", port: Number(port) });
    sock.once("connect", () => (sock.destroy(), resolve(true)));
    sock.once("error", () => resolve(false));
  });
}
for (const [name, port, env] of [["API", API_PORT, "JEV_API_PORT"], ["web", WEB_PORT, "JEV_WEB_PORT"]]) {
  if (await portInUse(port)) {
    console.error(
      `[dev] port ${port} (${name}) is already in use — is JEV already running in another terminal?\n` +
        `      Stop it, or pick another port: ${env}=<port> npm run dev` +
        (isWin ? "" : `\n      Find the process with: lsof -i :${port}`),
    );
    process.exit(1);
  }
}

if (!existsSync(join(root, "frontend", "node_modules"))) {
  console.log("[dev] installing frontend dependencies (first run)…");
  const r = spawnSync("pnpm", ["install"], { cwd: join(root, "frontend"), stdio: "inherit", shell: isWin });
  if (r.status !== 0) process.exit(r.status ?? 1);
}

const colours = { api: "\x1b[36m", web: "\x1b[33m" };
const reset = "\x1b[0m";
const children = [];
let stopping = false;

function run(name, cmd, args, cwd, env = {}) {
  const child = spawn(cmd, args, {
    cwd,
    env: { ...process.env, FORCE_COLOR: "1", ...env },
    shell: isWin,
    // own process group, so stopping it also stops its children (uvicorn reloader, next workers)
    detached: !isWin,
  });
  const tag = `${colours[name]}[${name}]${reset} `;
  for (const stream of [child.stdout, child.stderr]) {
    let buf = "";
    stream.on("data", (chunk) => {
      buf += chunk.toString();
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) process.stdout.write(tag + line + "\n");
    });
  }
  child.on("exit", (code) => {
    if (!stopping) {
      console.log(`[dev] ${name} exited (code ${code ?? "signal"}); stopping the rest`);
      stop(code ?? 1);
    }
  });
  children.push(child);
}

function stop(code = 0) {
  stopping = true;
  for (const child of children) {
    if (child.exitCode !== null) continue;
    try {
      if (isWin) spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
      else process.kill(-child.pid, "SIGTERM");
    } catch {
      // already gone
    }
  }
  setTimeout(() => process.exit(code), 500);
}

process.on("SIGINT", () => stop(0));
process.on("SIGTERM", () => stop(0));

run("api", "uv", ["run", "uvicorn", "jev_api.main:app", "--reload", "--reload-dir", "backend", "--reload-dir", "ml", "--port", API_PORT], root);
run("web", "pnpm", ["dev", "-p", WEB_PORT], join(root, "frontend"), { JEV_API_URL: `http://127.0.0.1:${API_PORT}` });

console.log(`[dev] web → http://localhost:${WEB_PORT}   api → http://localhost:${API_PORT}   (Ctrl+C stops both)`);
