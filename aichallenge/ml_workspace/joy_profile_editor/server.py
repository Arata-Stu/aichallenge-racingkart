#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import fcntl
import json
import os
import select
import struct
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JSIOCGAXES = 0x80016A11
JSIOCGBUTTONS = 0x80016A12
JSIOCGNAME = lambda length: 0x80006A13 + (length << 16)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_output_path() -> Path:
    return repo_root() / "aichallenge/workspace/src/aichallenge_tools/teleop_manager/config/teleop.param.yaml"


def read_js_device_snapshot(path_text: str = "/dev/input/js0", duration_s: float = 0.05) -> dict:
    path = Path(path_text)
    if not path.exists():
        return {"ok": False, "error": f"{path} does not exist", "path": str(path)}

    fd = os.open(str(path), os.O_RDONLY | os.O_NONBLOCK)
    try:
        axes_count = array.array("B", [0])
        buttons_count = array.array("B", [0])
        name_buffer = array.array("B", [0] * 128)
        fcntl.ioctl(fd, JSIOCGAXES, axes_count, True)
        fcntl.ioctl(fd, JSIOCGBUTTONS, buttons_count, True)
        try:
            fcntl.ioctl(fd, JSIOCGNAME(len(name_buffer)), name_buffer, True)
            name = name_buffer.tobytes().split(b"\0", 1)[0].decode(errors="replace")
        except OSError:
            name = path.name

        axes = [0 for _ in range(int(axes_count[0]))]
        buttons = [0 for _ in range(int(buttons_count[0]))]
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], max(0.0, deadline - time.monotonic()))
            if not ready:
                break
            while True:
                try:
                    data = os.read(fd, 8)
                except BlockingIOError:
                    break
                if len(data) != 8:
                    break
                _, value, event_type, number = struct.unpack("IhBB", data)
                kind = event_type & ~JS_EVENT_INIT
                if kind == JS_EVENT_AXIS and number < len(axes):
                    axes[number] = int(value)
                elif kind == JS_EVENT_BUTTON and number < len(buttons):
                    buttons[number] = int(value)

        return {
            "ok": True,
            "path": str(path),
            "name": name,
            "axes": axes,
            "axes_normalized": [max(-1.0, min(1.0, value / 32767.0)) for value in axes],
            "buttons": buttons,
        }
    except OSError as exc:
        return {"ok": False, "error": str(exc), "path": str(path)}
    finally:
        os.close(fd)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


class JoyEditorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], output_path: Path):
        super().__init__(address, Handler)
        self.output_path = output_path.resolve()
        self.allowed_root = repo_root().resolve()


class Handler(BaseHTTPRequestHandler):
    server: JoyEditorServer

    def log_message(self, fmt: str, *args) -> None:
        if self.path.startswith("/api/joy"):
            return
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._html()
            return
        if parsed.path == "/api/joy/js0":
            query = parse_qs(parsed.query)
            device = query.get("path", ["/dev/input/js0"])[0] or "/dev/input/js0"
            self._json(read_js_device_snapshot(device))
            return
        if parsed.path == "/api/config":
            self._json({"output_path": str(self.server.output_path)})
            return
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            yaml_text = str(body.get("yaml", ""))
            if not yaml_text.strip():
                raise ValueError("yaml is empty")
            output_path = self.server.output_path.resolve()
            if self.server.allowed_root not in output_path.parents:
                raise ValueError("output path is outside repository")
            atomic_write(output_path, yaml_text)
            self._json({"ok": True, "saved": str(output_path)})
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _html(self) -> None:
        path = Path(__file__).resolve().parent / "web/index.html"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="AIC teleop joy profile editor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--output", type=Path, default=default_output_path())
    args = parser.parse_args()

    server = JoyEditorServer((args.host, args.port), args.output)
    print(f"Joy editor: http://{args.host}:{args.port}/")
    print(f"Output YAML: {server.output_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
