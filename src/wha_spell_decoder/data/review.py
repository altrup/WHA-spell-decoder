"""Local browser UI to review and edit examples.jsonl / records.jsonl next to their images, and to write new records."""

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .annotate import CROPS, EXAMPLES, NOT_SPELLS, RECORDS, SIGILS, crop_list, cut, load, sign_sections, targets
from .fetch import RAW

FILES = {"examples": EXAMPLES, "records": RECORDS, "todo": RECORDS}  # todo = unannotated targets, saved into records
HOST, PORT = "127.0.0.1", 8765
ORIGIN = f"http://{HOST}:{PORT}"

PAGE = Path(__file__).with_name("review.html")
CROP_PAGE = PAGE.with_name("crop.html")


def file_name(query: str) -> str:
    name = parse_qs(query).get("file", ["records"])[0]
    if name not in FILES:
        raise ValueError(name)
    return name


def listing(name: str) -> list[dict]:
    text = {t["id"]: t["text"] for t in targets()}
    if name == "todo":
        done = load(EXAMPLES).keys() | load(RECORDS).keys()
        return [t for t in targets() if t["id"] not in done]
    return [{**r, "text": text.get(r["id"], "")} for r in load(FILES[name]).values()]


def crop_state() -> dict:
    index = json.loads((RAW / "telepedia" / "index.json").read_text())
    return {"crops": crop_list(), "pages": sorted(set(index) - NOT_SPELLS),
            "images": sorted(f"telepedia/images/{p.name}" for p in (RAW / "telepedia" / "images").iterdir())}


def save_crops(crops: list[dict]) -> None:
    """Rewrite crops.json and recut every listed seal, dropping files for boxes that changed or went away."""
    for c in crop_list():
        (RAW / "telepedia" / "images" / c["out"]).unlink(missing_ok=True)
    CROPS.write_text(json.dumps(crops, indent=1) + "\n")
    for c in crops:
        cut(c)


def signs() -> list[dict[str, str]]:
    out = [{"name": f"{n} sigil", "type": "sigil", "image": f"fandom/images/{img}"} for n, img in SIGILS.items()]
    return out + [{"name": n, "type": s["type"], "image": f"telepedia/images/{s['image']}"} for n, s in sign_sections().items()]


class Handler(BaseHTTPRequestHandler):
    def reply(self, body: bytes, ctype: str = "application/json", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def local(self) -> bool:
        """Reject DNS-rebinding and cross-site requests: only the page we serve may talk to us."""
        host_ok = self.headers.get("Host") in (f"{HOST}:{PORT}", f"localhost:{PORT}")
        origin_ok = self.command == "GET" or self.headers.get("Origin") == ORIGIN
        if not (host_ok and origin_ok):
            self.reply(b"forbidden", "text/plain", 403)
        return host_ok and origin_ok

    def do_GET(self) -> None:
        if not self.local():
            return
        url = urlparse(self.path)
        if url.path == "/":
            self.reply(PAGE.read_bytes(), "text/html")
        elif url.path == "/records":
            self.reply(json.dumps(listing(file_name(url.query))).encode())
        elif url.path == "/signs":
            self.reply(json.dumps(signs()).encode())
        elif url.path == "/crop":
            self.reply(CROP_PAGE.read_bytes(), "text/html")
        elif url.path == "/crops":
            self.reply(json.dumps(crop_state()).encode())
        elif url.path.startswith("/raw/"):
            f = (RAW / url.path[5:]).resolve()
            if RAW in f.parents and f.is_file():
                self.reply(f.read_bytes(), "image/*")
            else:
                self.reply(b"not found", "text/plain", 404)
        else:
            self.reply(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        if not self.local():
            return
        url = urlparse(self.path)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if url.path == "/crops":
            save_crops(body)
            return self.reply(b"{}")
        if url.path != "/save":
            return self.reply(b"not found", "text/plain", 404)
        p = FILES[file_name(url.query)]
        new = body
        new.pop("text", None)
        rows = list(load(p).values())
        if any(r["id"] == new["id"] for r in rows):
            rows = [new if r["id"] == new["id"] else r for r in rows]
        else:
            rows.append(new)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.reply(b"{}")


def main() -> None:
    server = HTTPServer((HOST, PORT), Handler)
    print(f"{ORIGIN}  (Ctrl-C to stop)")
    webbrowser.open(ORIGIN)
    server.serve_forever()
