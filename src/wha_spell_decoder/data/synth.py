"""Compose synthetic seals from sign and sigil atoms. The exact layout is saved so the annotator reads a true description."""

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .annotate import SIGILS, sign_sections
from .fetch import RAW

SYNTH = RAW.parent / "synth"
ATOMS = SYNTH / "atoms"  # user-drawn glyphs; atoms.json gives kind and note, else kind is "unknown"
LAYOUTS = SYNTH / "layouts.jsonl"
POINTS = ["north", "north-northeast", "northeast", "east-northeast", "east", "east-southeast", "southeast", "south-southeast",
          "south", "south-southwest", "southwest", "west-southwest", "west", "west-northwest", "northwest", "north-northwest"]


def mask(path: Path) -> Image.Image:
    """Black strokes become opaque on an L mask, trimmed to the glyph."""
    m = Image.open(path).convert("L").point(lambda v: 255 if v < 128 else 0)
    return m.crop(m.getbbox() or (0, 0, 1, 1))


def atoms() -> dict[str, dict]:
    out = {n: {"kind": "sign", "type": s["type"], "note": "", "mask": mask(RAW / "telepedia" / "images" / s["image"])}
           for n, s in sign_sections().items()}
    out |= {n: {"kind": "sigil", "type": "", "note": "", "mask": mask(RAW / "fandom" / "images" / f)} for n, f in SIGILS.items()}
    meta = json.loads((ATOMS / "atoms.json").read_text()) if (ATOMS / "atoms.json").exists() else {}
    for p in sorted(ATOMS.glob("*.png")) if ATOMS.exists() else []:
        out[p.stem] = {"kind": "unknown", "type": "", "note": "", **meta.get(p.name, {}), "mask": mask(p)}
    return out


def compass(angle: float) -> str:
    return POINTS[round(angle / 22.5) % 16]


def seal(rng: random.Random, a: dict[str, dict], depth: int = 0) -> dict:
    """Random layout. Angles are degrees clockwise from north; radii and scales are fractions of the ring radius."""
    names = {k: [n for n, v in a.items() if v["kind"] == k] for k in ("sign", "sigil", "unknown")}
    centre = lambda: rng.choice(names["sigil"]) if rng.random() < 0.7 else rng.choice(names["sign"] + names["unknown"])
    layout = {"ring": "open" if rng.random() < 0.15 else "closed", "double": depth == 0 and rng.random() < 0.2,
              "sigils": [{"name": centre(), "scale": round(rng.uniform(0.2, 0.55), 2)} for _ in range(rng.choices([0, 1, 2], [5, 80, 15])[0])],
              "signs": [], "subseals": []}
    pool = names["sign"] + names["unknown"]
    while len(layout["signs"]) < rng.randint(2, 8):
        name = rng.choice(pool)
        kind = a[name]["type"]
        facing = rng.choice(["center", "ring"]) if kind in ("Directional", "Semi") or rng.random() < 0.85 else "any"
        inverted = kind == "Semi" and rng.random() < 0.3
        count = rng.choice([2, 3, 4, 6]) if rng.random() < 0.5 else 1
        start, radius, scale = rng.uniform(0, 360), rng.uniform(0.55, 0.85 if not layout["double"] else 1.05), rng.uniform(0.15, 0.3)
        for i in range(count):
            angle = (start + 360 * i / count + rng.uniform(-4, 4)) % 360
            if any(abs((angle - o["angle"] + 180) % 360 - 180) < 28 and abs(radius - o["radius"]) < 0.2 for o in layout["signs"]):
                continue  # too close to a sign already placed
            layout["signs"].append({"name": name, "angle": round(angle, 1), "radius": round(radius + rng.uniform(-0.03, 0.03), 2),
                                    "facing": facing, "inverted": inverted, "scale": round(scale, 2),
                                    "rotation": round(rng.uniform(0, 360), 1) if facing == "any" else 0})
    if depth == 0 and rng.random() < 0.2:
        count, radius = rng.choice([1, 2, 3, 4]), rng.choice([0.6, 1.35])
        start = rng.uniform(0, 360)
        if radius < 1:
            for s in layout["sigils"]:
                s["scale"] = min(s["scale"], 0.3)
        for i in range(count):
            layout["subseals"].append({"angle": round((start + 360 * i / count) % 360, 1), "radius": radius,
                                       "scale": 0.22 if radius < 1 else 0.3, "seal": seal(rng, a, 1)})
    return layout


def place(canvas: Image.Image, m: Image.Image, cx: float, cy: float, height: float, rotation: float = 0, flip: bool = False) -> None:
    h = max(int(height), 2)
    m = m.resize((max(int(m.width * h / m.height), 1), h))
    if flip:
        m = m.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    m = m.rotate(rotation, expand=True, resample=Image.Resampling.BILINEAR)
    canvas.paste(0, (int(cx - m.width / 2), int(cy - m.height / 2)), m)


def draw(layout: dict, a: dict[str, dict], size: int = 512, cx: float | None = None, cy: float | None = None,
         radius: float | None = None, canvas: Image.Image | None = None, rng: random.Random | None = None) -> Image.Image:
    rng = rng or random.Random(0)
    canvas = canvas or Image.new("L", (size, size), 255)
    reach = max([1.0] + [sub["radius"] + sub["scale"] for sub in layout["subseals"]])  # satellites must fit on the canvas
    cx, cy, radius = cx or size / 2, cy or size / 2, radius or size * 0.46 / reach
    d = ImageDraw.Draw(canvas)
    width = max(int(radius * 0.025), 1)
    rings = [radius, radius * 0.82] if layout["double"] else [radius]
    for r in rings:
        box = (cx - r, cy - r, cx + r, cy + r)
        if layout["ring"] == "open":
            gap = rng.uniform(20, 70)
            start = rng.uniform(0, 360)
            d.arc(box, start + gap, start + 360, fill=0, width=width)
        else:
            d.ellipse(box, outline=0, width=width)
    for i, s in enumerate(layout["sigils"]):
        off = (i - (len(layout["sigils"]) - 1) / 2) * radius * 0.5
        place(canvas, a[s["name"]]["mask"], cx + off, cy, s["scale"] * radius)
    for s in layout["signs"]:
        t = math.radians(s["angle"])
        x, y = cx + s["radius"] * radius * math.sin(t), cy - s["radius"] * radius * math.cos(t)
        rot = {"center": 180 - s["angle"], "ring": -s["angle"], "any": s.get("rotation", 0)}[s["facing"]]
        place(canvas, a[s["name"]]["mask"], x, y, s["scale"] * radius, rot, s["inverted"])
    for sub in layout["subseals"]:
        t = math.radians(sub["angle"])
        x, y = cx + sub["radius"] * radius * math.sin(t), cy - sub["radius"] * radius * math.cos(t)
        r = sub["scale"] * radius
        if sub["radius"] > 1:
            d.line((cx + radius * math.sin(t), cy - radius * math.cos(t), x - r * math.sin(t), y + r * math.cos(t)), fill=0, width=width)
        draw(sub["seal"], a, size, x, y, r, canvas, rng)
    return canvas


def finish(img: Image.Image, rng: random.Random) -> Image.Image:
    """Vary stroke weight and sharpness so no two renders are pixel-identical."""
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.MinFilter(3))
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.8)))


def describe(layout: dict, a: dict[str, dict] | None = None, indent: str = "") -> str:
    a = a or {}
    label = lambda n: f'unnamed glyph "{n}"' + (f" (note: {a[n]['note']})" if a.get(n, {}).get("note") else "") if a.get(n, {}).get("kind") == "unknown" else n
    lines = [f"{indent}The ring {layout['ring']}" + (", with a second ring just inside it" if layout["double"] else "") + "."]
    for s in layout["sigils"]:
        lines.append(f"{indent}Center: {label(s['name'])}, {int(s['scale'] * 100)}% of the ring radius.")
    if not layout["sigils"]:
        lines.append(f"{indent}Center: empty.")
    for s in layout["signs"]:
        where = f"{compass(s['angle'])} ({s['angle']:.0f} degrees), {int(s['radius'] * 100)}% out from the center"
        face = {"center": "top of the glyph toward the center", "ring": "top of the glyph toward the ring", "any": f"rotated {s.get('rotation', 0):.0f} degrees"}[s["facing"]]
        lines.append(f"{indent}Sign: {label(s['name'])} at {where}, {face}" + (", inverted" if s["inverted"] else "") + ".")
    for sub in layout["subseals"]:
        side = "outside the ring, joined to it by a line" if sub["radius"] > 1 else "inside the ring"
        lines.append(f"{indent}Sub-seal at {compass(sub['angle'])}, {side}, {int(sub['scale'] * 100)}% of the ring radius:")
        lines.append(describe(sub["seal"], a, indent + "  "))
    return "\n".join(lines)


def synth_targets() -> list[dict[str, str]]:
    a = atoms()
    head = "This seal has no article. The layout below is exact; derive the effect from the sign rules.\n"
    return [{"id": r["id"], "kind": "synth", "name": r["id"].split("/")[-1], "source": "synth/layouts.jsonl",
             "text": head + describe(r["layout"], a)}
            for r in map(json.loads, LAYOUTS.read_text().splitlines())] if LAYOUTS.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic seals into data/synth/")
    ap.add_argument("count", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    a = atoms()
    (SYNTH / "images").mkdir(parents=True, exist_ok=True)
    with LAYOUTS.open("a") as f:
        for i in range(args.count):
            rng = random.Random(f"{args.seed}-{i}")
            name = f"s{args.seed}_{i:05d}.png"
            if (SYNTH / "images" / name).exists():
                continue
            layout = seal(rng, a)
            finish(draw(layout, a, args.size, rng=rng), rng).save(SYNTH / "images" / name)
            f.write(json.dumps({"id": f"synth/images/{name}", "layout": layout}) + "\n")
            print(name)
