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
