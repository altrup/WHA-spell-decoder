"""Local browser UI to review and edit examples.jsonl / records.jsonl next to their images."""

import json
import webbrowser
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from .fetch import RAW

PROCESSED = RAW.parent / "processed"
FILES = ("examples", "records")

PAGE = Path(__file__).with_name("review.html")


def path_for(query: str):
    name = parse_qs(query).get("file", ["records"])[0]
    if name not in FILES:
        raise ValueError(name)
    return PROCESSED / f"{name}.jsonl"


class Handler(BaseHTTPRequestHandler):
    def reply(self, body: bytes, ctype: str = "application/json", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/":
            self.reply(PAGE.read_bytes(), "text/html")
        elif url.path == "/records":
            p = path_for(url.query)
            self.reply(p.read_bytes().replace(b"\n", b",").rstrip(b",").join((b"[", b"]")) if p.exists() else b"[]")
        elif url.path.startswith("/raw/"):
            f = (RAW / url.path[5:]).resolve()
            if RAW in f.parents and f.is_file():
                self.reply(f.read_bytes(), "image/*")
            else:
                self.reply(b"not found", "text/plain", 404)
        else:
            self.reply(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path != "/save":
            return self.reply(b"not found", "text/plain", 404)
        p = path_for(url.query)
        new = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        rows = [new if r["id"] == new["id"] else r for r in rows]
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.reply(b"{}")


def main() -> None:
    server = HTTPServer(("127.0.0.1", 8765), Handler)
    print("http://127.0.0.1:8765  (Ctrl-C to stop)")
    webbrowser.open("http://127.0.0.1:8765")
    server.serve_forever()
