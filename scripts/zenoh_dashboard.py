#!/usr/bin/env python3
"""Small web dashboard for rk_studio Zenoh registry and hand gesture messages."""

from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

try:
    import zenoh
except ImportError as exc:
    raise SystemExit(
        "Missing Python package 'zenoh' for this interpreter:\n"
        f"  {sys.executable}\n"
        "Install it with:\n"
        f"  {sys.executable} -m pip install eclipse-zenoh"
    ) from exc


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>rk_studio Zenoh Dashboard</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #18202a;
      --muted: #6b7280;
      --line: #d8dde6;
      --good: #108a42;
      --bad: #b42318;
      --chip: #eef2f7;
      --accent: #195fcc;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111418;
        --panel: #171c22;
        --text: #eef2f7;
        --muted: #9aa4b2;
        --line: #2d3642;
        --chip: #222a34;
        --accent: #8ab4ff;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--bad);
    }
    .status.online .dot { background: var(--good); }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-height: 260px;
    }
    h2 {
      margin: 0 0 14px;
      font-size: 16px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .kv {
      display: grid;
      grid-template-columns: 120px minmax(0, 1fr);
      gap: 9px 12px;
      align-items: baseline;
    }
    .key { color: var(--muted); }
    .value {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      height: 26px;
      padding: 0 10px;
      border-radius: 999px;
      background: var(--chip);
      font-weight: 600;
    }
    .badge.good { color: var(--good); }
    .badge.bad { color: var(--bad); }
    .hands {
      display: grid;
      gap: 10px;
      margin-top: 6px;
    }
    .hand {
      display: grid;
      grid-template-columns: 56px minmax(0, 1fr);
      gap: 8px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: color-mix(in srgb, var(--panel) 82%, var(--chip));
    }
    .empty {
      color: var(--muted);
      padding: 18px 0;
    }
    .log {
      grid-column: 1 / -1;
      min-height: 170px;
    }
    pre {
      max-height: 230px;
      overflow: auto;
      margin: 0;
      padding: 12px;
      border-radius: 8px;
      background: var(--chip);
      color: var(--text);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    @media (max-width: 760px) {
      main { padding: 16px; }
      header { align-items: flex-start; flex-direction: column; }
      .grid { grid-template-columns: 1fr; }
      .log { grid-column: auto; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>rk_studio Zenoh Dashboard</h1>
      <div id="connection" class="status"><span class="dot"></span><span>connecting</span></div>
    </header>
    <div class="grid">
      <section>
        <h2>注册状态</h2>
        <div class="kv">
          <div class="key">状态</div><div class="value"><span id="reg-badge" class="badge bad">未注册</span></div>
          <div class="key">Entity ID</div><div id="entity-id" class="value">-</div>
          <div class="key">名称</div><div id="display-name" class="value">-</div>
          <div class="key">Owner</div><div id="owner" class="value">-</div>
          <div class="key">设备类型</div><div id="device-type" class="value">-</div>
          <div class="key">视频流</div><div id="video-url" class="value">-</div>
          <div class="key">更新时间</div><div id="registry-time" class="value">-</div>
        </div>
      </section>
      <section>
        <h2>手势识别</h2>
        <div class="kv">
          <div class="key">Camera</div><div id="camera-id" class="value">-</div>
          <div class="key">PTS(ns)</div><div id="pts-ns" class="value">-</div>
          <div class="key">更新时间</div><div id="hands-time" class="value">-</div>
        </div>
        <div id="hands" class="hands"><div class="empty">等待手势数据</div></div>
      </section>
      <section class="log">
        <h2>最近消息</h2>
        <pre id="last-message">等待 Zenoh 消息...</pre>
      </section>
    </div>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const connection = $("connection");

    function text(value) {
      return value === undefined || value === null || value === "" ? "-" : String(value);
    }

    function escapeHtml(value) {
      return text(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function render(state) {
      const reg = state.registry || {};
      const metadata = reg.metadata || {};
      const registered = Boolean(reg.registered);
      const badge = $("reg-badge");
      badge.textContent = registered ? "已注册" : "未注册";
      badge.className = "badge " + (registered ? "good" : "bad");
      $("entity-id").textContent = text(reg.entity_id);
      $("display-name").textContent = text(reg.display_name);
      $("owner").textContent = text(metadata.owner);
      $("device-type").textContent = text(metadata.device_type);
      $("video-url").textContent = text(metadata.video_stream_url);
      $("registry-time").textContent = text(reg.updated_at);

      const handsState = state.mediapipe || {};
      $("camera-id").textContent = text(handsState.camera_id);
      $("pts-ns").textContent = text(handsState.pts_ns);
      $("hands-time").textContent = text(handsState.updated_at);

      const hands = Array.isArray(handsState.hands) ? handsState.hands : [];
      if (hands.length === 0) {
        $("hands").innerHTML = '<div class="empty">暂无手势</div>';
      } else {
        $("hands").innerHTML = hands.map((hand) => {
          const gesture = text(hand.gesture || "unknown");
          return `<div class="hand"><strong>#${escapeHtml(hand.id)}</strong><div>${escapeHtml(gesture)}</div></div>`;
        }).join("");
      }

      if (state.last_message) {
        $("last-message").textContent = JSON.stringify(state.last_message, null, 2);
      }
    }

    async function loadInitialState() {
      const response = await fetch("/state");
      render(await response.json());
    }

    function connectEvents() {
      const events = new EventSource("/events");
      events.onopen = () => {
        connection.classList.add("online");
        connection.lastElementChild.textContent = "connected";
      };
      events.onmessage = (event) => {
        render(JSON.parse(event.data));
      };
      events.onerror = () => {
        connection.classList.remove("online");
        connection.lastElementChild.textContent = "reconnecting";
      };
    }

    loadInitialState().catch(console.error);
    connectEvents();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host, default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="HTTP bind port, default: 8765")
    parser.add_argument(
        "--registry-key",
        default="zho/entity/registry",
        help="Zenoh registry key expression, default: zho/entity/registry",
    )
    parser.add_argument(
        "--mediapipe-key",
        default="halmet/mediapipe",
        help="Zenoh Mediapipe key expression, default: halmet/mediapipe",
    )
    parser.add_argument("--mode", default="peer", choices=("peer", "client", "router"), help="Zenoh mode")
    parser.add_argument("--connect", action="append", default=[], help="Zenoh endpoint to connect to")
    parser.add_argument("--listen", action="append", default=[], help="Zenoh endpoint to listen on")
    return parser.parse_args()


def make_config(args: argparse.Namespace):
    config = zenoh.Config()
    config.insert_json5("mode", json.dumps(args.mode))
    if args.connect:
        config.insert_json5("connect/endpoints", json.dumps(args.connect))
    if args.listen:
        config.insert_json5("listen/endpoints", json.dumps(args.listen))
    return config


def payload_to_text(payload) -> str:
    if hasattr(payload, "to_bytes"):
        return payload.to_bytes().decode("utf-8", errors="replace")
    return bytes(payload).decode("utf-8", errors="replace")


class DashboardState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[queue.Queue[str]] = []
        self._state: dict[str, Any] = {
            "registry": {"registered": False},
            "mediapipe": {"hands": []},
            "last_message": None,
        }

    def snapshot_json(self) -> str:
        with self._lock:
            return json.dumps(self._state, ensure_ascii=False)

    def add_client(self) -> queue.Queue[str]:
        client: queue.Queue[str] = queue.Queue(maxsize=16)
        with self._lock:
            self._clients.append(client)
            client.put_nowait(json.dumps(self._state, ensure_ascii=False))
        return client

    def remove_client(self, client: queue.Queue[str]) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    def apply_registry(self, key: str, payload: dict[str, Any]) -> None:
        action = payload.get("action")
        entity_id = payload.get("entity_id")
        registered = action == "REG_REGISTER"
        registry = {
            "registered": registered,
            "action": action,
            "entity_id": entity_id,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if registered:
            registry.update(
                {
                    "display_name": payload.get("display_name"),
                    "metadata": payload.get("metadata") or {},
                }
            )
        else:
            registry["metadata"] = {}
        self._update("registry", registry, key, payload)

    def apply_mediapipe(self, key: str, payload: dict[str, Any]) -> None:
        mediapipe = {
            "camera_id": payload.get("camera_id"),
            "pts_ns": payload.get("pts_ns"),
            "hands": payload.get("hands") if isinstance(payload.get("hands"), list) else [],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._update("mediapipe", mediapipe, key, payload)

    def _update(self, slot: str, value: dict[str, Any], key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._state[slot] = value
            self._state["last_message"] = {"key": key, "payload": payload}
            message = json.dumps(self._state, ensure_ascii=False)
            clients = list(self._clients)
        for client in clients:
            try:
                client.put_nowait(message)
            except queue.Full:
                pass


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/state":
                body = state.snapshot_json().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/events":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                client = state.add_client()
                try:
                    while True:
                        try:
                            message = client.get(timeout=15)
                            self.wfile.write(f"data: {message}\n\n".encode("utf-8"))
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    state.remove_client(client)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


class DashboardHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> int:
    args = parse_args()
    stopped = threading.Event()
    state = DashboardState()

    def handle_signal(signum, frame) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    session = zenoh.open(make_config(args))

    def on_registry(sample) -> None:
        key = str(sample.key_expr)
        text = payload_to_text(sample.payload)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"raw": text}
        state.apply_registry(key, payload)

    def on_mediapipe(sample) -> None:
        key = str(sample.key_expr)
        text = payload_to_text(sample.payload)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"raw": text, "hands": []}
        state.apply_mediapipe(key, payload)

    registry_sub = session.declare_subscriber(args.registry_key, on_registry)
    mediapipe_sub = session.declare_subscriber(args.mediapipe_key, on_mediapipe)

    httpd = DashboardHttpServer((args.host, args.port), make_handler(state))
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()

    url = f"http://{args.host}:{args.port}/"
    print(f"dashboard: {url}")
    print(f"registry:  {args.registry_key}")
    print(f"mediapipe: {args.mediapipe_key}")
    if args.connect:
        print(f"connect:   {args.connect}")
    if args.listen:
        print(f"listen:    {args.listen}")

    try:
        while not stopped.is_set():
            time.sleep(0.2)
    finally:
        httpd.shutdown()
        registry_sub.undeclare()
        mediapipe_sub.undeclare()
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
