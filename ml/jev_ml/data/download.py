"""Download the MovieLens `ml-latest-small` dataset from GroupLens and verify its checksum.

Source: https://grouplens.org/datasets/movielens/  (F. Maxwell Harper and Joseph A. Konstan. 2015.
The MovieLens Datasets: History and Context. ACM TiiS 5, 4.) Licensed for research/non-commercial use.
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from pathlib import Path

import requests

from jev_ml.paths import RAW_DIR

log = logging.getLogger(__name__)

DATASET_NAME = "ml-latest-small"
BASE_URL = "https://files.grouplens.org/datasets/movielens"
EXPECTED_FILES = ("movies.csv", "ratings.csv", "tags.csv", "links.csv")


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - checksum published by GroupLens, not a security use
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_movielens(dest: Path = RAW_DIR, force: bool = False, timeout: int = 60) -> Path:
    """Download + verify + extract. Returns the directory holding the CSV files."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{DATASET_NAME}.zip"
    extract_dir = dest / DATASET_NAME

    if extract_dir.exists() and all((extract_dir / f).exists() for f in EXPECTED_FILES) and not force:
        log.info("dataset already present at %s", extract_dir)
        return extract_dir

    if force or not zip_path.exists():
        url = f"{BASE_URL}/{DATASET_NAME}.zip"
        log.info("downloading %s", url)
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            tmp = zip_path.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(1 << 16):
                    fh.write(chunk)
            tmp.replace(zip_path)

    md5_resp = requests.get(f"{BASE_URL}/{DATASET_NAME}.zip.md5", timeout=timeout)
    md5_resp.raise_for_status()
    # GroupLens publishes BSD-style "MD5 (file) = <hash>"; the hash is the last token.
    expected = md5_resp.text.split()[-1].strip().lower()
    actual = _md5(zip_path)
    if expected != actual:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {zip_path.name}: expected {expected}, got {actual}")
    log.info("checksum ok (md5 %s)", actual)

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"unsafe path in archive: {member}")
        zf.extractall(dest)  # noqa: S202 - members validated above

    missing = [f for f in EXPECTED_FILES if not (extract_dir / f).exists()]
    if missing:
        raise RuntimeError(f"archive missing expected files: {missing}")
    (extract_dir / "SOURCE_MD5").write_text(actual + "\n")
    return extract_dir
