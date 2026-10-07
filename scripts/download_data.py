"""Download the HGNN-AC / MAGNN preprocessed ACM, DBLP and IMDB data (Google Drive) into data/preprocessed.

usage: python scripts/download_data.py            # or --zip path/to/hgnnac.zip if you downloaded it by hand
The archive (~346 MB) is https://drive.google.com/file/d/1PqUjvSViICa8yOszqDrw-j96hXVJ0MHR/view
"""
import argparse
import shutil
import time
import zipfile
from pathlib import Path

FILE_ID = "1PqUjvSViICa8yOszqDrw-j96hXVJ0MHR"
NEED = ["ACM_processed", "DBLP_processed", "IMDB_processed"]
REPO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(REPO / "data" / "preprocessed"))
ap.add_argument("--zip", default=None, help="use an already downloaded archive")
args = ap.parse_args()
out = Path(args.out)
if all((out / n / "adjM.npz").exists() for n in NEED):
    raise SystemExit(f"already there: {out}")
z = Path(args.zip) if args.zip else out.parent / "hgnnac.zip"
z.parent.mkdir(parents=True, exist_ok=True)


def fetch():
    try:  # gdown first, then the direct usercontent URL (works where drive.google.com is blocked)
        import gdown
        gdown.download(id=FILE_ID, output=str(z), quiet=False)
        if zipfile.is_zipfile(z):
            return
    except Exception as e:  # noqa: BLE001
        print("gdown failed:", e)
    import requests
    url = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(z, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


if not zipfile.is_zipfile(z):
    for attempt in range(3):
        fetch()
        if zipfile.is_zipfile(z):
            break
        time.sleep(5)
    else:
        raise SystemExit(f"download failed; get it in a browser from the URL above and pass --zip")
tmp = out.parent / "_unzip"
with zipfile.ZipFile(z) as f:
    f.extractall(tmp)
out.mkdir(parents=True, exist_ok=True)
for n in NEED:
    src = next(p for p in tmp.rglob(n) if p.is_dir())
    shutil.move(str(src), out / n)
shutil.rmtree(tmp)
print("data ready:", out)
