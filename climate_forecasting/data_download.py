import json
import urllib.parse
import urllib.request
from pathlib import Path


def yadisk_download(public_link: str, dst: Path) -> None:
    api = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    url = f"{api}?{urllib.parse.urlencode({'public_key': public_link})}"

    with urllib.request.urlopen(url) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    href = payload.get("href")
    if not href:
        raise RuntimeError(f"No 'href' in Yandex Disk response: {payload}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(href, dst)


def ensure_raw_data(public_link: str, dst: Path) -> None:
    if dst.exists():
        return
    yadisk_download(public_link, dst)
