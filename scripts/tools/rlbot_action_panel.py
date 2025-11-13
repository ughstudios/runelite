#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in cur.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


class ActionIPC:
    def __init__(self, ipc_dir: Path):
        self.dir = ipc_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.space_file = self.dir / "action_space.json"
        self.action_file = self.dir / "action.json"
        self._seq = self._init_seq()

    def _init_seq(self) -> int:
        if self.action_file.exists():
            try:
                data = json.loads(self.action_file.read_text("utf-8"))
                return int(data.get("seq", 0))
            except Exception:
                pass
        return 0

    def load_actions(self) -> list[str]:
        if not self.space_file.exists():
            return []
        try:
            data = json.loads(self.space_file.read_text("utf-8"))
        except Exception:
            return []
        actions = data.get("actions") or []
        return [str(a) for a in actions]

    def send_action(self, index: int) -> int:
        self._seq += 1
        payload = {"seq": self._seq, "action": int(index)}
        tmp = self.action_file.with_suffix(self.action_file.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.action_file)
        return self._seq

    def find_index_by_name(self, name: str) -> int | None:
        """Find action index by case-insensitive name or substring match."""
        name = (name or "").strip().lower()
        actions = self.load_actions()
        if not actions or not name:
            return None
        # Prefer exact match, then startswith, then substring
        for i, a in enumerate(actions):
            if a.lower() == name:
                return i
        for i, a in enumerate(actions):
            if a.lower().startswith(name):
                return i
        for i, a in enumerate(actions):
            if name in a.lower():
                return i
        return None


class PanelHandler(BaseHTTPRequestHandler):
    def __init__(self, ipc: ActionIPC, *args, **kwargs):
        self.ipc = ipc
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_panel()
            return
        if parsed.path == "/action":
            params = urllib.parse.parse_qs(parsed.query)
            idx = params.get("idx") or []
            try:
                index = int(idx[0])
            except Exception:
                self.send_json({"status": "error", "message": "invalid index"})
                return
            seq = self.ipc.send_action(index)
            self.send_json({"status": "ok", "seq": seq, "action": index})
            return
        if parsed.path == "/action_by_name":
            params = urllib.parse.parse_qs(parsed.query)
            name = (params.get("name") or [""])[0]
            idx = self.ipc.find_index_by_name(name)
            if idx is None:
                self.send_json({"status": "error", "message": f"action not found: {name}"})
                return
            seq = self.ipc.send_action(int(idx))
            self.send_json({"status": "ok", "seq": seq, "action": int(idx)})
            return
        if parsed.path == "/click":
            params = urllib.parse.parse_qs(parsed.query)
            try:
                x = int(params.get("x", [None])[0])
                y = int(params.get("y", [None])[0])
            except Exception:
                self.send_json({"status": "error", "message": "invalid coordinates"})
                return
            click_file = self.ipc.dir / "click_request.json"
            click_file.write_text(json.dumps({"x": x, "y": y, "ts": int(time.time())}))
            self.send_json({"status": "ok", "coords": [x, y]})
            return
        self.send_response(404)
        self.end_headers()

    def send_panel(self):
        actions = self.ipc.load_actions()
        body = self.render_html(actions)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def send_json(self, data: dict):
        payload = json.dumps(data)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))

    def render_html(self, actions: list[str]) -> str:
        if not actions:
            rows = "<p>No action_space.json found yet. Run RuneLite with RLBot enabled.</p>"
        else:
            rows = "\n".join(
                f'<button data-idx="{idx}" class="action-btn">[{idx}] {name}</button>'
                for idx, name in enumerate(actions)
            )
        quick = """<div id=quick>
  <h3>Quick Actions</h3>
  <div class=quick-row>
    <button class=\"q\" data-name=\"BankWhenFullTask\">Bank Now</button>
    <button class=\"q\" data-name=\"ChopNearestTreeTask\">Chop Tree</button>
    <button class=\"q\" data-name=\"NavigateToTreeHotspotTask\">Go To Trees</button>
    <button class=\"q\" data-name=\"IdleTask\">Idle</button>
  </div>
  <p class=hint>Buttons resolve by name; they work even if indices change.</p>
</div>"""
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>RLBot Action Panel</title>
  <style>
    body {{ font-family: sans-serif; background: #1b1e23; color: #f0f0f0; margin: 1rem; }}
    .action-btn {{ margin: .25rem; padding: .5rem 1rem; border-radius: 4px; background: #2b6cb0; border: none; color: white; cursor: pointer; }}
    .action-btn:hover {{ background: #3182ce; }}
    .q {{ margin: .25rem; padding: .5rem 1rem; border-radius: 4px; background: #805ad5; border: none; color: white; cursor: pointer; }}
    .q:hover {{ background: #6b46c1; }}
    #status {{ margin-top: 1rem; color: #9ae6b4; }}
    .hint {{ color: #a0aec0; font-size: .9rem; }}
  </style>
</head>
<body>
  <h1>RLBot Action Panel</h1>
  <p>IPC dir: {self.ipc.dir}</p>
  {quick}
  <div id="buttons">{rows}</div>
  <h3>Manual click</h3>
  <label>X: <input id="click-x" type="number" value="500" /></label>
  <label>Y: <input id="click-y" type="number" value="360" /></label>
  <button id="click-btn">Click at coords</button>
  <p id="status">Ready</p>
  <script>
    document.querySelectorAll(".action-btn").forEach(btn => {{
      btn.addEventListener("click", async () => {{
        const idx = btn.getAttribute("data-idx");
        const resp = await fetch("/action?idx=" + idx);
        const data = await resp.json();
        if (data.status === "ok") {{
          document.getElementById("status").textContent = "sent seq=" + data.seq;
        }} else {{
          document.getElementById("status").textContent = "error: " + data.message;
        }}
      }});
    }});
    document.querySelectorAll(".q").forEach(btn => {{
      btn.addEventListener("click", async () => {{
        const name = btn.getAttribute("data-name");
        const resp = await fetch("/action_by_name?name=" + encodeURIComponent(name));
        const data = await resp.json();
        if (data.status === "ok") {{
          document.getElementById("status").textContent = name + " -> seq=" + data.seq + " (idx=" + data.action + ")";
        }} else {{
          document.getElementById("status").textContent = "error: " + data.message;
        }}
      }});
    }});
    document.getElementById("click-btn").addEventListener("click", async () => {{
      const x = document.getElementById("click-x").value;
      const y = document.getElementById("click-y").value;
      const resp = await fetch(`/click?x=${{encodeURIComponent(x)}}&y=${{encodeURIComponent(y)}}`);
      const data = await resp.json();
      if (data.status === "ok") {{
        document.getElementById("status").textContent = "click request saved (" + data.coords + ")";
      }} else {{
        document.getElementById("status").textContent = "error: " + data.message;
      }}
    }});
  </script>
</body>
</html>"""
        return html


def run_server(ipc_dir: Path, port: int):
    ipc = ActionIPC(ipc_dir)

    def handler(*args, **kwargs):
        PanelHandler(ipc, *args, **kwargs)

    server = HTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Opening action panel in browser: {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="RLBot Action Web Panel")
    parser.add_argument(
        "--ipc-dir",
        default=str(repo_root() / "rlbot-ipc"),
        help="IPC directory (defaults to repo rlbot-ipc)",
    )
    parser.add_argument("--port", type=int, default=0, help="HTTP port (0 = random)")
    return parser.parse_args()


def main():
    args = parse_args()
    run_server(Path(args.ipc_dir).expanduser().resolve(), args.port)


if __name__ == "__main__":
    main()
