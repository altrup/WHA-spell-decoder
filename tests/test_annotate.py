from PIL import Image

from wha_spell_decoder.data import annotate


def test_media_type_sniffs_bytes_not_extension():
    assert annotate.media_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert annotate.media_type(b"\x89PNG\r\n\x1a\n") == "image/png"
    assert annotate.media_type(b"\xff\xd8\xff\xe0") == "image/jpeg"


def test_sign_sections_parse_image_and_type():
    signs = annotate.sign_sections()
    assert len(signs) == 47
    assert signs["Pulling"]["image"] == "Pull.png"
    assert signs["Pulling"]["type"] == "Directional"
    assert "pulled towards the seal" in signs["Pulling"]["text"]


def test_targets_cover_both_sites_and_skip_non_spell_pages():
    targets = annotate.targets()
    kinds = {t["kind"] for t in targets}
    assert kinds == {"sign", "spell"}
    names = {t["name"] for t in targets}
    assert "Sand Cage" in names and "Pulling" in names
    assert not names & annotate.NOT_SPELLS
    assert {t["id"].split("/")[0] for t in targets if t["kind"] == "spell"} == {"telepedia", "fandom"}
    assert len({t["id"] for t in targets}) == len(targets)


def test_spell_text_keeps_lead_and_description_only():
    text = "{{Infobox}}\nX is a spell.\n\n==Description==\nSigil and signs.\n\n==Usage==\nUsed by Coco in Chapter 3.\n\n==References==\n"
    out = annotate.spell_text(text)
    assert "Sigil and signs." in out and "is a spell" in out
    assert "Chapter 3" not in out and "References" not in out


def test_crops_replace_their_source_pages_in_targets():
    targets = {t["id"]: t for t in annotate.targets()}
    crops = {c["out"] for c in annotate.crop_list()}
    assert crops and all(f"telepedia/images/{c}" in targets for c in crops)
    assert not any(t["id"].endswith(c["src"].split("/")[-1]) for t in targets.values() for c in annotate.crop_list())
    assert targets["telepedia/images/Archives_Vol_01_Sample6_raincleaver.png"]["name"] == "Raincleaver Spell"


def test_cut_writes_box_sized_file(tmp_path, monkeypatch):
    images = tmp_path / "telepedia" / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (50, 40)).save(images / "page.png")
    monkeypatch.setattr(annotate, "RAW", tmp_path)
    annotate.cut({"src": "telepedia/images/page.png", "box": [10, 5, 30, 25], "out": "seal.png"})
    assert Image.open(images / "seal.png").size == (20, 20)


def test_save_crops_rejects_paths_outside_images(tmp_path, monkeypatch):
    import pytest

    from wha_spell_decoder.data import review

    monkeypatch.setattr(review, "CROPS", tmp_path / "crops.json")
    monkeypatch.setattr(review, "crop_list", lambda: [])
    with pytest.raises(AssertionError):
        review.save_crops([{"src": "telepedia/images/a.png", "box": [0, 0, 1, 1], "out": "../../evil.png", "page": "x"}])
    assert not (tmp_path / "crops.json").exists()
