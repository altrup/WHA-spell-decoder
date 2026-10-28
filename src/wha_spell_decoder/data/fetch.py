"""Fetch raw wiki articles and their images into data/raw/<site>/."""

import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

UA = "wha-spell-decoder/0.1 (https://github.com/altrup/WHA-spell-decoder)"
RAW = Path(__file__).parents[3] / "data" / "raw"
assert (RAW.parents[1] / "pyproject.toml").exists(), RAW
SITES = {
    "telepedia": ("https://witchhatatelier.telepedia.net/api.php", ["Signs Explained", "Sigils Explained"], "Category:Spells"),
    "fandom": ("https://witch-hat-atelier.fandom.com/api.php", ["Spells"], "Category:Spells"),
}


def get(url: str) -> bytes:
    time.sleep(0.5)  # be polite to the wikis
    with urlopen(Request(url, headers={"User-Agent": UA})) as r:
        return r.read()


def api(endpoint: str, **params: str) -> dict:
    return json.loads(get(f"{endpoint}?{urlencode({**params, 'format': 'json'})}"))


def category_members(endpoint: str, category: str) -> list[str]:
    q = api(endpoint, action="query", list="categorymembers", cmtitle=category, cmlimit="500", cmnamespace="0")
    return [m["title"] for m in q["query"]["categorymembers"]]


def image_urls(endpoint: str, names: list[str]) -> dict[str, str]:
    urls = {}
    for i in range(0, len(names), 50):
        q = api(endpoint, action="query", prop="imageinfo", iiprop="url",
                titles="|".join(f"File:{n}" for n in names[i:i + 50]))
        for p in q["query"]["pages"].values():
            if "imageinfo" in p:
                urls[p["title"].removeprefix("File:").replace(" ", "_")] = p["imageinfo"][0]["url"]
    return urls


def fetch_site(name: str, out: Path) -> None:
    endpoint, pages, category = SITES[name]
    if category:
        pages = pages + category_members(endpoint, category)
    site_dir = out / name
    (site_dir / "images").mkdir(parents=True, exist_ok=True)
    index = {}
    for title in pages:
        parsed = api(endpoint, action="parse", page=title, prop="wikitext|images")["parse"]
        path = site_dir / f"{title.replace('/', '_')}.wikitext"
        path.write_text(parsed["wikitext"]["*"])
        index[title] = {"wikitext": path.name, "images": parsed["images"]}
        print(f"{name}: {title} ({len(parsed['images'])} images)")
    names = sorted({img for entry in index.values() for img in entry["images"]})
    for img, url in image_urls(endpoint, names).items():
        dest = site_dir / "images" / img
        if not dest.exists():
            dest.write_bytes(get(url))
    (site_dir / "index.json").write_text(json.dumps(index, indent=1))


def main() -> None:
    for name in SITES:
        fetch_site(name, RAW)
