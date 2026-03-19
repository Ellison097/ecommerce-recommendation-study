from __future__ import annotations

from pathlib import Path
import ssl
import subprocess
import zipfile
import urllib.request


ML_100K_URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
ONLINE_RETAIL_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'


def _download_file(url: str, target: Path):
    try:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as resp, open(target, 'wb') as out:
            out.write(resp.read())
    except Exception:
        subprocess.run(['curl', '-L', '-k', url, '-o', str(target)], check=True)


def download_movielens_100k(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_zip = raw_dir / 'ml-100k.zip'
    extract_dir = raw_dir / 'ml-100k'

    if extract_dir.exists():
        return extract_dir

    if not target_zip.exists():
        _download_file(ML_100K_URL, target_zip)

    with zipfile.ZipFile(target_zip, 'r') as zf:
        zf.extractall(raw_dir)

    return extract_dir


def download_online_retail(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / 'online_retail.xlsx'
    if not target.exists():
        _download_file(ONLINE_RETAIL_URL, target)
    return target
