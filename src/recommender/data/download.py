from __future__ import annotations

from pathlib import Path
import ssl
import subprocess
import zipfile
import urllib.request


ML_100K_URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'


def download_movielens_100k(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_zip = raw_dir / 'ml-100k.zip'
    extract_dir = raw_dir / 'ml-100k'

    if extract_dir.exists():
        return extract_dir

    if not target_zip.exists():
        try:
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(ML_100K_URL, context=context) as resp, open(target_zip, 'wb') as out:
                out.write(resp.read())
        except Exception:
            subprocess.run(['curl', '-L', '-k', ML_100K_URL, '-o', str(target_zip)], check=True)

    with zipfile.ZipFile(target_zip, 'r') as zf:
        zf.extractall(raw_dir)

    return extract_dir
