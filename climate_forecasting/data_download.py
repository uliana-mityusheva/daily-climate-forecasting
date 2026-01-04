import json
import urllib.parse
import urllib.request
from pathlib import Path


def _dvc_fetch_to_path(
    dvc_repo: str | None, dvc_path: str, dst: Path, rev: str | None
) -> bool:
    """Try fetching a tracked file from DVC and write it to dst.
    Returns True if succeeded, False otherwise.
    """
    try:
        from dvc.api import open as dvc_open  # lazy import

        dst.parent.mkdir(parents=True, exist_ok=True)
        with (
            dvc_open(dvc_path, repo=dvc_repo or ".", rev=rev) as fsrc,
            open(dst, "wb") as fdst,
        ):
            fdst.write(fsrc.read())
        return True
    except Exception:
        return False


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


def download_data(public_link: str, dst: Path) -> None:
    """Download dataset from an open source link into local storage.

    This function is required by the assignment when using local storage.
    Currently uses Yandex Disk public link; can be extended to other sources.
    """
    dst = Path(dst)
    if dst.exists():
        return
    yadisk_download(public_link, dst)


def ensure_raw_data_dvc_first(
    dst: Path,
    public_link: str | None = None,
    dvc_repo: str | None = ".",
    dvc_path: str | None = None,
    dvc_rev: str | None = None,
) -> None:
    """Ensure raw data exists, trying DVC first, then fallback downloader."""
    if dst.exists():
        return

    if dvc_path:
        ok = _dvc_fetch_to_path(dvc_repo, dvc_path, dst, dvc_rev)
        if ok:
            return

    if public_link:
        download_data(public_link, dst)
        return

    raise FileNotFoundError(
        f"Could not obtain data at {dst}. Provide DVC path or public link."
    )
