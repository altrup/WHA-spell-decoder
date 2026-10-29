import json
import random

from PIL import Image

from wha_spell_decoder.data import synth


def test_atoms_load_wiki_signs_sigils_and_custom(tmp_path, monkeypatch):
    Image.new("RGB", (40, 40), "white").save(tmp_path / "odd.png")
    Image.new("RGB", (40, 40), "white").save(tmp_path / "fire2.png")
    (tmp_path / "atoms.json").write_text(json.dumps({"fire2.png": {"kind": "sigil", "note": "fire with curled tips"}}))
    monkeypatch.setattr(synth, "ATOMS", tmp_path)
    a = synth.atoms()
    assert a["Columns"]["kind"] == "sign" and a["Columns"]["type"] == "Directional"
    assert a["Fire"]["kind"] == "sigil"
    assert a["odd"]["kind"] == "unknown"
    assert a["fire2"] == {**a["fire2"], "kind": "sigil", "note": "fire with curled tips"}
    assert all(v["mask"].mode == "L" for v in a.values())


def test_seal_layout_and_drawing_agree():
    a = synth.atoms()
    rng = random.Random(1)
    for _ in range(20):
        layout = synth.seal(rng, a)
        assert layout["ring"] in ("closed", "open")
        assert 1 <= len(layout["signs"]) <= 12
        img = synth.draw(layout, a, 256)
        assert img.size == (256, 256) and img.mode == "L"


def test_describe_names_every_sign_and_ring_state():
    layout = {"ring": "open", "double": False, "sigils": [{"name": "Fire", "scale": 0.4}],
              "signs": [{"name": "Columns", "angle": 90, "radius": 0.7, "facing": "center", "inverted": False, "scale": 0.2}],
              "subseals": []}
    text = synth.describe(layout)
    assert "ring open" in text.lower() and "Fire" in text and "Columns" in text and "east" in text


def test_compass():
    assert synth.compass(0) == "north" and synth.compass(90) == "east" and synth.compass(225) == "southwest"


def test_synth_targets_carry_exact_layout_text(tmp_path, monkeypatch):
    layout = {"ring": "closed", "double": False, "sigils": [], "signs": [{"name": "Columns", "angle": 0, "radius": 0.7, "facing": "ring", "inverted": False, "scale": 0.2}], "subseals": []}
    (tmp_path / "layouts.jsonl").write_text(json.dumps({"id": "synth/images/s0_00000.png", "layout": layout}) + "\n")
    monkeypatch.setattr(synth, "LAYOUTS", tmp_path / "layouts.jsonl")
    t = synth.synth_targets()
    assert t[0]["kind"] == "synth" and "Columns" in t[0]["text"] and "north" in t[0]["text"]


def test_messiness_is_per_element_and_described():
    a = synth.atoms()
    layouts = [synth.seal(random.Random(i), a) for i in range(30)]
    assert all(0 <= s["messy"] <= 1 for l in layouts for s in l["signs"] + l["sigils"]) and all(0 <= l["ring_messy"] <= 1 for l in layouts)
    assert any(len({s["messy"] > 0.5 for s in l["signs"]}) == 2 for l in layouts)  # mixed neat and rough in one seal
    layout = {"ring": "closed", "ring_messy": 0.9, "double": False, "sigils": [], "subseals": [],
              "signs": [{"name": "Columns", "angle": 0, "radius": 0.7, "facing": "ring", "inverted": False, "scale": 0.2, "messy": 0.9}]}
    assert "roughly" in synth.describe(layout)


def test_mask_treats_transparent_as_background(tmp_path):
    im = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    im.paste((0, 0, 0, 255), (5, 8, 15, 12))  # a 10x4 stroke; transparent pixels must not count as ink
    im.save(tmp_path / "t.png")
    m = synth.mask(tmp_path / "t.png")
    assert abs(m.width / m.height - 2.5) < 0.1


def test_mutants_are_reproducible_and_described():
    a = synth.atoms()
    rng = random.Random(5)
    spec = synth.mutant("Fire", a, rng)
    assert spec["base"] == "Fire" and spec["op"] in synth.OPS
    m1, m2 = synth.glyph(spec, a), synth.glyph(spec, a)
    assert m1.tobytes() == m2.tobytes() and m1.tobytes() != a["Fire"]["mask"].resize(m1.size).tobytes()
    layout = {"ring": "closed", "ring_messy": 0, "double": False, "sigils": [{**spec, "scale": 0.4, "messy": 0}], "signs": [], "subseals": []}
    text = synth.describe(layout, a)
    assert "unnamed glyph" in text and "Fire" in text
    assert synth.draw(layout, a, 128).size == (128, 128)


def test_novel_glyphs_appear_in_a_minority_of_seals():
    a = synth.atoms()
    novel = sum(any("op" in g or a[g["name"]]["kind"] == "unknown" for g in l["signs"] + l["sigils"])
                for l in (synth.seal(random.Random(i), a) for i in range(200)))
    assert 15 <= novel <= 70
