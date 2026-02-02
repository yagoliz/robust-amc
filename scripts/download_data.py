#!/usr/bin/env python3
"""Download RadioML2016.10a dataset."""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


# Dataset URL (hosted on deepsig.ai)
DATASET_URL = "https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2"
DATASET_FILENAME = "RML2016.10a.tar.bz2"
EXTRACTED_FILENAME = "RML2016.10a_dict.pkl"


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> None:
    """Download file with progress bar."""
    print(f"Downloading from {url}")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def extract_tar_bz2(archive_path: Path, extract_dir: Path) -> None:
    """Extract tar.bz2 archive."""
    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(extract_dir)


def main():
    # Determine data directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    archive_path = data_dir / DATASET_FILENAME
    pkl_path = data_dir / EXTRACTED_FILENAME

    # Check if already downloaded
    if pkl_path.exists():
        print(f"Dataset already exists at {pkl_path}")
        return

    # Download if archive doesn't exist
    if not archive_path.exists():
        try:
            download_file(DATASET_URL, archive_path)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("\nAlternative download options:")
            print("1. Manual download from: https://www.deepsig.ai/datasets")
            print("2. Use Kaggle: https://www.kaggle.com/datasets/antoniotorres/radioml-dataset")
            print(f"\nPlace the file at: {pkl_path}")
            sys.exit(1)

    # Extract archive
    if archive_path.exists():
        try:
            extract_tar_bz2(archive_path, data_dir)
            print(f"Dataset extracted to {data_dir}")

            # Clean up archive
            archive_path.unlink()
            print("Removed archive file")
        except Exception as e:
            print(f"Error extracting archive: {e}")
            sys.exit(1)

    # Verify extraction
    if pkl_path.exists():
        print(f"Dataset ready at {pkl_path}")
    else:
        # Check for alternative naming
        alt_names = list(data_dir.glob("*.pkl"))
        if alt_names:
            print(f"Found pickle file: {alt_names[0]}")
            alt_names[0].rename(pkl_path)
            print(f"Renamed to {pkl_path}")
        else:
            print("Warning: Could not find extracted pickle file")


if __name__ == "__main__":
    main()
