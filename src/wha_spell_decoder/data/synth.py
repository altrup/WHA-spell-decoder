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
    im = Image.open(path).convert("RGBA")
    flat = Image.new("RGBA", im.size, "white")
    flat.alpha_composite(im)  # transparent pixels are background, not strokes
    m = flat.convert("L").point(lambda v: 255 if v < 128 else 0)
    m = m.crop(m.getbbox() or (0, 0, 1, 1))
    if max(m.size) < 240:  # wiki atoms are 70 px; upscale and re-threshold so large placements are not blocky
        k = 240 / max(m.size)
        m = m.resize((int(m.width * k) or 1, int(m.height * k) or 1), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(1.5))
        m = m.point(lambda v: 255 if v > 110 else 0)
    return m


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


OPS = ("dot", "stroke", "erase", "merge", "stretch", "mirror")
SIDES = ("top", "bottom", "left", "right")


def mutant(base: str, a: dict[str, dict], rng: random.Random) -> dict:
    """A novel glyph made from a known one. The spec is enough to redraw it and to say what was changed."""
    op = rng.choice(OPS)
    spec = {"name": f"{base}+{op}", "base": base, "op": op, "side": rng.choice(SIDES), "pos": round(rng.uniform(0.2, 0.8), 2)}
    if op == "merge":
        spec["other"] = rng.choice([n for n, v in a.items() if v["kind"] != "unknown" and n != base])
    return spec


def glyph(g: dict, a: dict[str, dict]) -> Image.Image:
    """The mask for a layout entry: a plain atom, or a mutant rebuilt from its spec."""
    if "op" not in g:
        return a[g["name"]]["mask"]
    m = a[g["base"]]["mask"].copy()
    w, h = m.size
    side, pos = g["side"], g["pos"]
    edge = {"top": (int(w * pos), 0), "bottom": (int(w * pos), h), "left": (0, int(h * pos)), "right": (w, int(h * pos))}[side]
    pad = int(max(w, h) * 0.3)
    out = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    out.paste(m, (pad, pad))
    d = ImageDraw.Draw(out)
    x, y = edge[0] + pad, edge[1] + pad
    t = max(int(max(w, h) * 0.06), 2)
    if g["op"] == "dot":
        r = max(w, h) * 0.08
        d.ellipse((x - r, y - r, x + r, y + r), fill=255)
    elif g["op"] == "stroke":
        dx, dy = {"top": (0, -1), "bottom": (0, 1), "left": (-1, 0), "right": (1, 0)}[side]
        d.line((x, y, x + dx * pad * 0.8, y + dy * pad * 0.8), fill=255, width=t)
    elif g["op"] == "erase":
        f = 0.35
        box = {"top": (0, 0, out.width, pad + h * f), "bottom": (0, pad + h * (1 - f), out.width, out.height),
               "left": (0, 0, pad + w * f, out.height), "right": (pad + w * (1 - f), 0, out.width, out.height)}[side]
        d.rectangle(box, fill=0)
    elif g["op"] == "merge":
        o = a[g["other"]]["mask"]
        k = max(w, h) * 0.5 / max(o.size)
        o = o.resize((max(int(o.width * k), 1), max(int(o.height * k), 1)))
        out.paste(255, (int(x - o.width / 2), int(y - o.height / 2)), o)
    elif g["op"] == "stretch":
        out = out.resize((int(out.width * (1.6 if side in ("left", "right") else 1)), int(out.height * (1.6 if side in ("top", "bottom") else 1))))
    elif g["op"] == "mirror":
        if side in ("left", "right"):
            half = out.crop((0, 0, out.width // 2, out.height)) if side == "left" else out.crop((out.width // 2, 0, out.width, out.height))
            flipped = half.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            out.paste(half if side == "left" else flipped, (0, 0))
            out.paste(flipped if side == "left" else half, (out.width // 2, 0))
        else:
            half = out.crop((0, 0, out.width, out.height // 2)) if side == "top" else out.crop((0, out.height // 2, out.width, out.height))
            flipped = half.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            out.paste(half if side == "top" else flipped, (0, 0))
            out.paste(flipped if side == "top" else half, (0, out.height // 2))
    return out.crop(out.getbbox() or (0, 0, 1, 1))


def note(g: dict, a: dict[str, dict]) -> str:
    if "op" not in g:
        return a[g["name"]]["note"]
    base, side = g["base"], g["side"]
    return {"dot": f"{base} with an extra dot on its {side} side", "stroke": f"{base} with an extra stroke from its {side} edge",
            "erase": f"{base} with its {side} part missing", "merge": f"{base} with a small {g.get('other')} attached at the {side}",
            "stretch": f"{base} stretched {'wide' if side in ('left', 'right') else 'tall'}",
            "mirror": f"{base} with its {side} half mirrored onto the other"}[g["op"]]


def seal(rng: random.Random, a: dict[str, dict], depth: int = 0) -> dict:
    """Random layout. Angles are degrees clockwise from north; radii and scales are fractions of the ring radius."""
    names = {k: [n for n, v in a.items() if v["kind"] == k] for k in ("sign", "sigil", "unknown")}
    novel = rng.random() < 0.2  # one seal in five may carry glyphs that are not in the books

    def pick(known: list[str]) -> dict:
        if novel and rng.random() < 0.5:
            return mutant(rng.choice(known), a, rng) if not names["unknown"] or rng.random() < 0.5 else {"name": rng.choice(names["unknown"])}
        return {"name": rng.choice(known)}

    centre = lambda: pick(names["sigil"]) if rng.random() < 0.7 else pick(names["sign"])
    base = rng.choice([0.0, 0.0, 0.2, 0.5, 0.8])  # seal-wide neatness; each glyph strays from it
    messy = lambda: round(min(1, max(0, base + rng.gauss(0, 0.2))), 2)
    layout = {"ring": "open" if rng.random() < 0.15 else "closed", "ring_messy": messy(), "double": depth == 0 and rng.random() < 0.2,
              "sigils": [], "signs": [], "subseals": []}
    n_sigils = rng.choices([0, 1, 2], [5, 80, 15])[0] if depth == 0 else 1
    top = {0: 0.55, 1: 0.55, 2: 0.32}[n_sigils] if depth == 0 else 0.6  # two centre glyphs must not overlap
    layout["sigils"] = [{**centre(), "scale": round(rng.uniform(0.2, top), 2), "messy": messy()} for _ in range(n_sigils)]
    if depth == 0 and rng.random() < 0.2:
        count, radius = rng.choice([1, 2, 3, 4]), rng.choice([0.6, 1.35])
        start = rng.uniform(0, 360)
        if radius < 1:
            for s in layout["sigils"]:
                s["scale"] = min(s["scale"], 0.3)
        for i in range(count):
            layout["subseals"].append({"angle": round((start + 360 * i / count) % 360, 1), "radius": radius,
                                       "scale": 0.26 if radius < 1 else 0.36, "seal": seal(rng, a, 1)})
    want = rng.randint(2, 8) if depth == 0 else 0  # sub-seals hold one centre glyph only
    for _ in range(40):  # placement attempts; a crowded ring simply ends with fewer signs
        if len(layout["signs"]) >= want:
            break
        g = pick(names["sign"])
        kind = a[g.get("base", g["name"])]["type"]
        facing = rng.choice(["center", "ring"]) if kind in ("Directional", "Semi") or rng.random() < 0.85 else "any"
        inverted = kind == "Semi" and rng.random() < 0.3
        count = rng.choice([2, 3, 4, 6]) if rng.random() < 0.5 and depth == 0 else 1
        start, scale = rng.uniform(0, 360), rng.uniform(0.15, 0.3) if depth == 0 else 0.3
        radius = 0.91 if layout["double"] and rng.random() < 0.5 else rng.uniform(0.55, 0.72 if layout["double"] else 0.85) if depth == 0 else 0.72
        if radius == 0.91:
            scale = min(scale, 0.13)  # the band between the rings is narrow
        elif count >= 4:
            scale = min(scale, 0.22)
        radius = min(radius, 0.97 - scale / 2)  # stay inside the ring
        for i in range(count):
            angle = (start + 360 * i / count + rng.uniform(-4, 4)) % 360
            taken = layout["signs"] + [o for o in layout["subseals"] if o["radius"] < 1]
            if any(abs((angle - o["angle"] + 180) % 360 - 180) < (36 if "name" in o else 45) and abs(radius - o["radius"]) < 0.25 for o in taken):
                continue  # too close to a sign or inner sub-seal already placed
            layout["signs"].append({**g, "angle": round(angle, 1), "radius": round(radius + rng.uniform(-0.03, 0.03), 2),
                                    "facing": facing, "inverted": inverted, "scale": round(scale, 2), "messy": messy(),
                                    "rotation": round(rng.uniform(0, 360), 1) if facing == "any" else 0})
    return layout


def rough(m: Image.Image, messy: float, rng: random.Random) -> Image.Image:
    """Skew the glyph's corners and vary its stroke weight in proportion to messiness."""
    if messy < 0.05:
        return m
    pad = int(max(m.size) * 0.12 * messy) + 1
    padded = Image.new("L", (m.width + 2 * pad, m.height + 2 * pad), 0)
    padded.paste(m, (pad, pad))  # room for the skew, so strokes are not clipped
    m, (w, h) = padded, padded.size
    j = lambda v: v + rng.uniform(-messy, messy) * 0.12 * max(w, h)
    return m.transform((w, h), Image.Transform.QUAD, (j(0), j(0), j(0), j(h), j(w), j(h), j(w), j(0)), Image.Resampling.BILINEAR)


def place(canvas: Image.Image, m: Image.Image, cx: float, cy: float, height: float, rotation: float = 0, flip: bool = False,
          messy: float = 0, rng: random.Random | None = None) -> None:
    h = max(int(height * min(1, 1.4 * m.height / m.width)), 2)  # wide glyphs are sized by width, not height
    m = m.resize((max(int(m.width * h / m.height), 1), h))
    if flip:
        m = m.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    rng = rng or random.Random(0)
    m = rough(m, messy, rng)
    m = m.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    if rng.random() < messy * 0.5:
        m = m.filter(ImageFilter.MaxFilter(3))
    canvas.paste(0, (int(cx - m.width / 2), int(cy - m.height / 2)), m)


def ring_points(cx: float, cy: float, r: float, messy: float, rng: random.Random, start: float = 0, sweep: float = 360) -> list[tuple[float, float]]:
    """A circle whose radius wobbles smoothly: a few random sinusoids scaled by messiness."""
    waves = [(rng.randint(2, 6), rng.uniform(0, 2 * math.pi), rng.uniform(0.3, 1)) for _ in range(3)]
    pts = []
    for i in range(int(sweep) + 1):
        t = math.radians(start + i)
        rr = r * (1 + messy * 0.12 * sum(amp * math.sin(k * t + ph) for k, ph, amp in waves) / 3)
        pts.append((cx + rr * math.cos(t), cy + rr * math.sin(t)))
    return pts


def draw(layout: dict, a: dict[str, dict], size: int = 512, cx: float | None = None, cy: float | None = None,
         radius: float | None = None, canvas: Image.Image | None = None, rng: random.Random | None = None) -> Image.Image:
    rng = rng or random.Random(0)
    canvas = canvas or Image.new("L", (size, size), 255)
    reach = max([1.0] + [sub["radius"] + sub["scale"] for sub in layout["subseals"]])  # satellites must fit on the canvas
    cx, cy, radius = cx or size / 2, cy or size / 2, radius or size * 0.46 / reach
    d = ImageDraw.Draw(canvas)
    width = max(int(radius * 0.025), 1)
    rings = [radius, radius * 0.82] if layout["double"] else [radius]
    ring_messy = layout.get("ring_messy", 0)
    for r in rings:
        gap, start = (rng.uniform(20, 70), rng.uniform(0, 360)) if layout["ring"] == "open" else (0, 0)
        d.line(ring_points(cx, cy, r, ring_messy, rng, start + gap, 360 - gap), fill=0, width=width, joint="curve")
    for i, s in enumerate(layout["sigils"]):
        off = (i - (len(layout["sigils"]) - 1) / 2) * radius * 0.5
        place(canvas, glyph(s, a), cx + off, cy, s["scale"] * radius, messy=s.get("messy", 0), rng=rng)
    for s in layout["signs"]:
        t = math.radians(s["angle"])
        x, y = cx + s["radius"] * radius * math.sin(t), cy - s["radius"] * radius * math.cos(t)
        rot = {"center": 180 - s["angle"], "ring": -s["angle"], "any": s.get("rotation", 0)}[s["facing"]]
        place(canvas, glyph(s, a), x, y, s["scale"] * radius, rot, s["inverted"], s.get("messy", 0), rng)
    for sub in layout["subseals"]:
        t = math.radians(sub["angle"])
        x, y = cx + sub["radius"] * radius * math.sin(t), cy - sub["radius"] * radius * math.cos(t)
        r = sub["scale"] * radius
        if sub["radius"] > 1:
            d.line((cx + radius * math.sin(t), cy - radius * math.cos(t), x - r * math.sin(t), y + r * math.cos(t)), fill=0, width=width)
        draw(sub["seal"], a, size, x, y, r, canvas, rng)
    return canvas


def finish(img: Image.Image, rng: random.Random) -> Image.Image:
    """Vary sharpness so no two renders are pixel-identical."""
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.8)))


def describe(layout: dict, a: dict[str, dict] | None = None, indent: str = "") -> str:
    a = a or {}
    def label(g: dict) -> str:
        if "op" not in g and a.get(g["name"], {}).get("kind") != "unknown":
            return g["name"]
        n = note(g, a) if a else g["name"]
        return "unnamed glyph" + (f" (note: {n})" if n else f' "{g["name"]}"')
    neat = lambda v: "drawn neatly" if v < 0.3 else "a little rough" if v < 0.6 else "drawn roughly"
    lines = [f"{indent}The ring {layout['ring']}, {neat(layout.get('ring_messy', 0))}" + (", with a second ring just inside it" if layout["double"] else "") + "."]
    for s in layout["sigils"]:
        lines.append(f"{indent}Center: {label(s)}, {int(s['scale'] * 100)}% of the ring radius, {neat(s.get('messy', 0))}.")
    if not layout["sigils"]:
        lines.append(f"{indent}Center: empty.")
    for s in layout["signs"]:
        where = f"{compass(s['angle'])} ({s['angle']:.0f} degrees), {int(s['radius'] * 100)}% out from the center"
        face = {"center": "top of the glyph toward the center", "ring": "top of the glyph toward the ring", "any": f"rotated {s.get('rotation', 0):.0f} degrees"}[s["facing"]]
        lines.append(f"{indent}Sign: {label(s)} at {where}, {face}" + (", inverted" if s["inverted"] else "") + f", {neat(s.get('messy', 0))}.")
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
    ap.add_argument("--seed", type=int, default=42)
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
