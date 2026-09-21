"""Download and extract the ASVspoof 2019 Logical Access dataset.

    python scripts/download.py                  # download, then extract
    python scripts/download.py --skip-download  # extract a zip you already have

About 7.1 GB, so it takes hours. It resumes if interrupted, so Ctrl-C and
re-running later is safe.

Do not download PA.zip (16.4 GB) - replay attacks are out of scope.
"""

import os
import sys

# scripts/inspect.py shadows the standard library's inspect module, which
# numpy and librosa import. Replace scripts/ on sys.path with the repo root.
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse                                              # noqa: E402
import logging                                               # noqa: E402
import time                                                  # noqa: E402
import zipfile                                               # noqa: E402
from pathlib import Path                                     # noqa: E402

import requests                                              # noqa: E402
from tqdm import tqdm                                        # noqa: E402

from src.utils import REPO_ROOT, setup_logging               # noqa: E402

URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
ZIP_NAME = "LA.zip"

# Exact size from the server's Content-Range header. An exact figure makes a
# truncated download detectable; a rounded one does not.
EXPECTED_BYTES = 7_640_952_520
CHUNK = 1024 * 1024

# The server drops the connection every gigabyte or so. Each attempt resumes
# from what is already on disk, so these are continuations, not restarts.
MAX_ATTEMPTS = 40
RETRY_WAIT = 10

EXPECTED_DIRS = [
    "LA/ASVspoof2019_LA_train/flac",
    "LA/ASVspoof2019_LA_dev/flac",
    "LA/ASVspoof2019_LA_eval/flac",
    "LA/ASVspoof2019_LA_cm_protocols",
]


def download(url, dest):
    """Fetch the zip, restarting the stream whenever the connection drops.

    On a 7 GB download the connection breaking is a when, not an if - we lost
    it at 1.2 GB on the first run. Each attempt sends a fresh Range header for
    the bytes we do not have yet, so nothing already downloaded is re-fetched.
    """
    no_progress = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = dest.stat().st_size if dest.exists() else 0

        if have >= EXPECTED_BYTES:
            logging.info("download complete, %.2f GB", have / 1e9)
            return

        try:
            _fetch_from(url, dest, have)
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            logging.warning("connection dropped (%s)", type(exc).__name__)

        now = dest.stat().st_size if dest.exists() else 0
        if now == have:
            # Distinguish a flaky connection from one that is simply down.
            no_progress += 1
            if no_progress >= 3:
                raise RuntimeError(
                    f"stuck at {now / 1e9:.2f} GB - three attempts in a row "
                    f"moved nothing. Check the network, then re-run.")
        else:
            no_progress = 0
            logging.info("%.2f GB of %.2f GB (%.1f%%), attempt %d",
                         now / 1e9, EXPECTED_BYTES / 1e9,
                         100.0 * now / EXPECTED_BYTES, attempt)

        time.sleep(RETRY_WAIT)

    raise RuntimeError(
        f"gave up after {MAX_ATTEMPTS} attempts. Re-run to carry on from "
        f"{dest.stat().st_size / 1e9:.2f} GB.")


def _fetch_from(url, dest, have):
    """One attempt: ask for everything from byte `have` onwards and append it."""

    headers = {"Range": f"bytes={have}-"} if have else {}
    response = requests.get(url, headers=headers, stream=True, timeout=60)

    if response.status_code == 416:
        # Range not satisfiable - we already have the whole file.
        logging.info("file already complete (%.2f GB)", have / 1e9)
        response.close()
        return

    if have and response.status_code == 200:
        # The server ignored the Range header and is sending from the start.
        # Appending would corrupt the file, so start over.
        logging.warning("server does not support resume - restarting download")
        have = 0
        dest.unlink()
    elif have and response.status_code == 206:
        logging.info("resuming from %.2f GB", have / 1e9)

    response.raise_for_status()

    remaining = int(response.headers.get("Content-Length", 0))
    total = have + remaining if remaining else None

    with open(dest, "ab" if have else "wb") as f, tqdm(
        total=total, initial=have, unit="B", unit_scale=True,
        unit_divisor=1024, desc=dest.name,
    ) as bar:
        for chunk in response.iter_content(chunk_size=CHUNK):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    logging.info("stream ended at %.2f GB", dest.stat().st_size / 1e9)


def extract(zip_path, data_dir):
    """Check the zip opens, then extract it."""
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Run without --skip-download.")

    # Read the index before extracting. A truncated download fails here,
    # which beats failing 40 minutes into writing 121,000 files.
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
    print("\nResulting folder structure:")
    ok = True
    for rel in EXPECTED_DIRS:
        path = data_dir / rel
        if path.is_dir():
            print(f"  OK       {rel}  ({sum(1 for _ in path.iterdir())} entries)")
        else:
            print(f"  MISSING  {rel}")
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true",
                        help="skip the download and just extract an existing LA.zip")
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
        print(f"\nThen delete the zip to free 7 GB:  del {zip_path}")
    else:
        print("Some folders are missing - the download may be incomplete.")
        print("Re-run this script; it resumes rather than starting over.")
    print("-" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
