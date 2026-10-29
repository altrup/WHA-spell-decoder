"""Turn raw wiki images + wikitext into training records with Claude. Reruns fill only missing ids."""

import argparse
import base64
import itertools
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel

from .fetch import RAW, inside

RECORDS = RAW.parent / "processed" / "records.jsonl"
EXAMPLES = RECORDS.with_name("examples.jsonl")
CROPS = RECORDS.with_name("crops.json")  # seals cut out of multi-seal pages
SIGNS_PAGE = RAW / "telepedia" / "Signs Explained.wikitext"
SIGILS = {"Fire": "Fire_sigil.png", "Water": "Water.png", "Earth": "Earth.png", "Wind": "Wind_(redirect).png", "Light": "Light.png"}
NOT_SPELLS = {"Spells", "Magic", "Forbidden Magic", "Signs Explained", "Sigils Explained"}
REFERENCE_PAGES = ["telepedia/Signs Explained.wikitext", "telepedia/Sigils Explained.wikitext", "fandom/Signs Explained.wikitext"]

RULES = """You are rewording wiki articles for the manga Witch Hat Atelier, for a new article on interpreting spells.
A seal is a ring containing a sigil plus signs. The sigil is the element glyph at the center (fire, water, earth, wind, light, or a variant) and sets what kind of magic the spell is; its size relative to the ring sets intensity. Signs are modifiers around it that set how the magic behaves. A few signs, such as Repetition and Billow, can sit in the center in place of a sigil; they are still signs. A seal may have several sigils or none.
A spell activates only when its ring is closed, so say "ring open" or "ring closed" for that, and "fully visible" or "partly visible" for framing. Never say "complete". Messy or small seals are weaker and shorter lived; a ring that is not round enough gives a fleeting effect or may fail to activate.

For each target image you get the image and the wiki text about it. Produce:
- usable: clean (complete, legible glyph), partial (glyph cut off or only partly visible), occluded (glyph present but obscured by art, ink, hands), no_glyph (no seal or sign visible).
- description: ONLY what is visible in the image. Name signs using the reference names when you can match them; say where they sit (center, north, east, inverted, ...) and how many. Say what is cut off or hidden. Never describe things the wiki text mentions that you cannot see.
- justification: how the visible signs and sigil combine, citing the sign rules. If a glyph matches no known sign, name the closest sign, say it is a guess, and still derive an effect from that reading. Say nothing can be decoded only when the glyph is hidden or cut off. If signs directly contradict each other, or a seal stacks more than three unrelated kinds of modifier, say which parts dominate and that the result is likely unstable; otherwise do not call a seal unstable.
- effect: what the spell does, as a wiki would state it, in one to three sentences. Outcome only: never name signs, the ring, or reasons, those belong in justification. For a lone sign, state it is a component and give its modifier effect. If the glyph is not visible or not readable, say the effect cannot be determined from the image. Never infer the effect from surrounding panel art. Describe what the spell does, never who used it, where, or in which chapter. If the visible part suggests a symmetric seal, you may assume the hidden part mirrors it, but say that you assume it. If the ring is open, say it does not activate and then what it would do if closed. If the seal is weak, unstable, or unpredictable, show it in the behavior: say what wavers, misfires, collapses early, or happens unevenly, as part of describing the effect. Never append it as a bare label.
Never mention the wiki, the reference material, or any source in the output; state things as known facts. Never comment on the image's origin or status, such as it being a redraw, a reconstruction, a fan illustration, or a glyph never shown in the manga. Treat every image as a seal to read.
Write plain prose. Length matches the image: a full seal gets a few sentences per field, a lone sign or an image without a glyph gets one or two."""


class Record(BaseModel):
    usable: Literal["clean", "partial", "occluded", "no_glyph"]
    description: str
    justification: str
    effect: str


def media_type(data: bytes) -> str:
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    raise ValueError("unknown image format")


def site_dir(site: str) -> Path:
    return RAW.parent / site if site == "synth" else RAW / site


def image_block(rel: str) -> dict:
    site, _, name = rel.split("/")
    data = inside(site_dir(site) / "images", name).read_bytes()
    return {"type": "image", "source": {"type": "base64", "media_type": media_type(data), "data": base64.b64encode(data).decode()}}


def sign_sections() -> dict[str, dict[str, str]]:
    parts = re.split(r"^===(.+?)===\s*$", SIGNS_PAGE.read_text(), flags=re.M)
    signs = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        img = re.search(r"\|magicimage=\[\[File:(.+?)[|\]]", body)
        if img:
            kind = re.search(r"\|magictype=(\w*)", body)
            signs[name] = {"image": img[1].replace(" ", "_"), "type": kind[1] if kind else "", "text": body}
    return signs


def spell_text(wikitext: str) -> str:
    """Lead and Description only: usage history and navigation invite story context into the effect."""
    return re.split(r"^==\s*(Usage|History|Gallery|Trivia|References|Site Navigation)\s*==", wikitext, flags=re.M)[0].strip()


def crop_list() -> list[dict]:
    return json.loads(CROPS.read_text()) if CROPS.exists() else []


def cut(c: dict) -> None:
    dest = inside(RAW / "telepedia" / "images", c["out"])
    if not dest.exists():
        Image.open(inside(RAW / "telepedia" / "images", c["src"].split("/")[-1])).crop(c["box"]).save(dest)


def crops() -> list[dict[str, str]]:
    """Cut each listed seal out of its page image (once) and ground it on its own spell page."""
    index = json.loads((RAW / "telepedia" / "index.json").read_text())
    out = []
    for c in crop_list():
        cut(c)
        page = index[c["page"]]["wikitext"]
        out.append({"id": f"telepedia/images/{c['out']}", "kind": "spell", "name": c["page"], "source": f"telepedia/{page}",
                    "text": spell_text((RAW / "telepedia" / page).read_text())})
    return out


def targets() -> list[dict[str, str]]:
    out = [{"id": f"telepedia/images/{s['image']}", "kind": "sign", "name": name,
            "source": f"telepedia/Signs Explained.wikitext#{name}", "text": s["text"]}
           for name, s in sign_sections().items()]
    cut = {c["src"] for c in crop_list()}
    for site in ("telepedia", "fandom"):
        index = json.loads((RAW / site / "index.json").read_text())
        for title, entry in index.items():
            if title in NOT_SPELLS:
                continue
            text = spell_text((RAW / site / entry["wikitext"]).read_text())
            out += [{"id": f"{site}/images/{img}", "kind": "spell", "name": title, "source": f"{site}/{entry['wikitext']}", "text": text}
                    for img in entry["images"] if f"{site}/images/{img}" not in cut]
    out += crops()
    seen: set[str] = set()  # an image shared by several pages is grounded on the first page that lists it
    return [t for t in out if not (t["id"] in seen or seen.add(t["id"]))]


def exemplars(n: int = 20) -> dict[str, dict]:
    """Real clean seals shown to the annotator before synthetic ones, so effects stay in the manga's register."""
    rows = [r for r in load(RECORDS).values() if r["usable"] == "clean" and r["kind"] == "spell"]
    rows = list({r["name"]: r for r in rows}.values())
    return {r["id"]: r for r in random.Random(0).sample(rows, min(n, len(rows)))}


def load(path) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}


def reference_blocks(examples: dict[str, dict]) -> list[dict]:
    blocks = [{"type": "text", "text": "Reference element sigils:"}]
    for name, img in SIGILS.items():
        blocks += [image_block(f"fandom/images/{img}"), {"type": "text", "text": f"{name} sigil"}]
    blocks.append({"type": "text", "text": "Reference signs (name, type):"})
    for name, s in sign_sections().items():
        blocks += [image_block(f"telepedia/images/{s['image']}"), {"type": "text", "text": f"{name} ({s['type'] or 'unknown type'})"}]
    for page in REFERENCE_PAGES:
        blocks.append({"type": "text", "text": f"Reference page {page}:\n" + (RAW / page).read_text()})
    for ex, r in examples.items():
        blocks += [{"type": "text", "text": f"Example target ({r['kind']}: {r['name']}):"}, image_block(ex),
                   {"type": "text", "text": "Example output:\n" + json.dumps({k: r[k] for k in Record.model_fields})}]
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def annotate(client: anthropic.Anthropic, prefix: list[dict], t: dict[str, str]) -> tuple[Record, str, str]:
    content = prefix + [{"type": "text", "text": f"Target ({t['kind']}: {t['name']}). Wiki text:\n{t['text']}"}, image_block(t["id"])]
    resp = client.beta.messages.parse(model="claude-opus-5-5", max_tokens=4000, system=RULES,
                                      output_config={"effort": "high"}, messages=[{"role": "user", "content": content}],
                                      output_format=Record, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refused: {resp.stop_details.category}")
    u = resp.usage
    usage = f"in={u.input_tokens} cache_write={u.cache_creation_input_tokens} cache_read={u.cache_read_input_tokens} out={u.output_tokens}"
    return resp.parsed_output, resp.model, usage


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate raw wiki images into data/processed/records.jsonl")
    ap.add_argument("limit", nargs="?", type=int, default=None, help="stop after this many new records")
    ap.add_argument("--only", help="annotate only targets whose id or name contains this text")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--synth", action="store_true", help="annotate data/synth images into data/synth/records.jsonl")
    args = ap.parse_args()
    load_dotenv(RAW.parents[1] / ".env")
    examples = load(EXAMPLES)
    if args.synth:
        from .synth import SYNTH, synth_targets
        # synthetic examples go last, nearest the target, after real exemplars set the register
        records, pool, examples = SYNTH / "records.jsonl", synth_targets(), {**examples, **exemplars(), **load(SYNTH / "examples.jsonl")}
    else:
        records, pool = RECORDS, targets()
    done = load(records).keys() | examples.keys()
    client = anthropic.Anthropic()
    prefix = reference_blocks(examples)
    todo = [t for t in pool if t["id"] not in done and not (args.only and args.only not in t["id"] + t["name"])]
    todo = todo[:args.limit]

    def run(t: dict[str, str]) -> tuple[dict[str, str], dict | None, str]:
        try:
            rec, model, usage = annotate(client, prefix, t)
        except RuntimeError as e:  # refusal: skipped, picked up again by a later run
            return t, None, str(e)
        return t, {"model": model, **rec.model_dump()}, f"{rec.usable} ({model}; {usage})"

    with records.open("a") as f, ThreadPoolExecutor(args.workers) as pool:
        # first call alone so the cached prefix is written once, then the rest in parallel
        for t, rec, note in itertools.chain([run(todo[0])], pool.map(run, todo[1:])) if todo else []:
            print(f"{t['id']}: {note}")
            if rec:
                f.write(json.dumps({"id": t["id"], "kind": t["kind"], "name": t["name"], "source": t["source"], **rec}) + "\n")
                f.flush()
