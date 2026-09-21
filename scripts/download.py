"""Download and extract the ASVspoof 2019 Logical Access dataset.

    python scripts/download.py                  # download, then extract
    python scripts/download.py --skip-download  # extract a zip you already have

The download is ~7.1 GB and will take hours. It resumes if interrupted, so it
is safe to stop it with Ctrl-C and run it again later.

Do NOT download PA.zip (16.4 GB) - replay attacks are out of scope.
"""

import os
import sys

# scripts/ is first on sys.path when you run a script from it, and this folder
# contains inspect.py - which shadows Python's standard library `inspect`
# module. numpy, librosa and matplotlib all import `inspect` internally, so
# without this they would find our file instead and fail with a confusing
# "circular import" error. Drop the script directory, add the repo root.
# The same block is in scripts/inspect.py, which explains it at length.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or os.getcwd()) != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

import argparse                                             # noqa: E402
import logging                                              # noqa: E402
import zipfile                                              # noqa: E402
from pathlib import Path                                    # noqa: E402
import requests                                             # noqa: E402
from tqdm import tqdm                                       # noqa: E402
from src.utils import REPO_ROOT, setup_logging              # noqa: E402

URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
ZIP_NAME = "LA.zip"
# Exact size, read from the server's Content-Range header on 2026-09-21.
# 7,640,952,520 bytes = 7.12 GiB, which is the "7.1 GB" BUILD.md quotes.
# An exact figure makes truncation detectable; a rounded one does not.
EXPECTED_BYTES = 7_640_952_520
CHUNK = 1024 * 1024   # 1 MiB

EXPECTED_DIRS = [
    "LA/ASVspoof2019_LA_train/flac",
    "LA/ASVspoof2019_LA_dev/flac",
    "LA/ASVspoof2019_LA_eval/flac",
    "LA/ASVspoof2019_LA_cm_protocols",
]


def download(url, dest):
    """Stream `url` to `dest`, resuming from a partial file if there is one."""
    have = dest.stat().st_size if dest.exists() else 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    response = requests.get(url, headers=headers, stream=True, timeout=60)

    if response.status_code == 416:
        # Range not satisfiable: we already have the whole file.
        logging.info("file already complete (%.2f GB)", have / 1e9)
        response.close()
        return

    if have and response.status_code == 200:
        # The server ignored our Range header and is sending the file from the
        # start. Appending would corrupt it, so throw away what we have.
        logging.warning("server does not support resume - restarting download")
        have = 0
        dest.unlink()
    elif have and response.status_code == 206:
        logging.info("resuming from %.2f GB", have / 1e9)

    response.raise_for_status()

    remaining = int(response.headers.get("Content-Length", 0))
    total = have + remaining if remaining else None

    mode = "ab" if have else "wb"
    with open(dest, mode) as f, tqdm(
        total=total, initial=have, unit="B", unit_scale=True,
        unit_divisor=1024, desc=dest.name,
    ) as bar:
        for chunk in response.iter_content(chunk_size=CHUNK):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    size = dest.stat().st_size
    logging.info("downloaded %.2f GB to %s", size / 1e9, dest)
    if size != EXPECTED_BYTES:
        logging.warning(
            "size is %d bytes, expected %d (difference %+d). The file is "
            "probably truncated - re-run this script and it will resume.",
            size, EXPECTED_BYTES, size - EXPECTED_BYTES,
        )
    else:
        logging.info("size matches the expected byte count exactly")


def extract(zip_path, data_dir):
    """Check the zip opens, then extract it into data_dir."""
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Run without --skip-download, or put the "
            f"file there yourself."
        )

    # Open and read the index before extracting. A truncated download usually
    # fails right here, which is much better than failing 40 minutes into
    # writing 121,000 files.
    logging.info("checking the zip opens...")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        logging.info("zip is readable, %d entries", len(names))

        data_dir.mkdir(parents=True, exist_ok=True)
        logging.info("extracting into %s (this takes a while)...", data_dir)
        for name in tqdm(names, unit="file", desc="extracting"):
            zf.extract(name, data_dir)

    logging.info("extraction complete")


def report_tree(data_dir):
    """Print what we ended up with, and whether it looks right."""
    print("\nResulting folder structure:")
    ok = True
    for rel in EXPECTED_DIRS:
        path = data_dir / rel
        if path.is_dir():
            n = sum(1 for _ in path.iterdir())
            print(f"  OK       {rel}  ({n} entries)")
        else:
            print(f"  MISSING  {rel}")
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download", action="store_true",
        help="skip the download and just extract an existing LA.zip",
    )
    args = parser.parse_args()

    setup_logging()
    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / ZIP_NAME

    if not args.skip_download:
        download(URL, zip_path)
    else:
        logging.info("--skip-download given, using %s", zip_path)

    extract(zip_path, data_dir)
    ok = report_tree(data_dir)

    print("\n" + "-" * 70)
    if ok:
        print("Next:  python scripts/verify.py")
        print(f"\nThen delete the zip to free ~7 GB:  del {zip_path}")
    else:
        print("Some folders are missing - the download may be incomplete.")
        print("Re-run this script; it will resume rather than start over.")
    print("-" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
