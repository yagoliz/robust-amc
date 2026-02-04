#!/usr/bin/env python3
"""Preprocess RadioML2018.01a HDF5 to memory-mapped NumPy arrays.

This script converts the large HDF5 file to a format that can be loaded
in seconds with minimal memory footprint. The output uses memory-mapped
NumPy arrays that only load data on-demand.

Usage:
    uv run python scripts/preprocess_radioml2018.py
    uv run python scripts/preprocess_radioml2018.py --input data/GOLD_XYZ_OSC.0001_1024.hdf5
    uv run python scripts/preprocess_radioml2018.py --no-split-segments  # Keep 1024-sample format
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from robust_amc.data.radioml2018_loader import (
    MODULATION_CLASSES_2018,
    SNR_LEVELS_2018,
)


def compute_file_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def stratified_split_indices(
    labels: np.ndarray,
    snrs: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Compute stratified train/val/test split indices.

    Returns indices into the full dataset for each split.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Create stratification key from (label, snr) pairs
    snr_bins = np.round(snrs / 2) * 2  # Round to nearest 2 dB
    strat_key = labels * 1000 + snr_bins.astype(int)

    all_indices = np.arange(len(labels))

    # First split: train vs (val + test)
    idx_train, idx_temp = train_test_split(
        all_indices,
        train_size=train_ratio,
        stratify=strat_key,
        random_state=random_state,
    )

    # Second split: val vs test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    strat_key_temp = strat_key[idx_temp]

    idx_val, idx_test = train_test_split(
        idx_temp,
        train_size=val_test_ratio,
        stratify=strat_key_temp,
        random_state=random_state,
    )

    return {
        "train": np.sort(idx_train),
        "val": np.sort(idx_val),
        "test": np.sort(idx_test),
    }


def preprocess_radioml2018(
    input_path: Path,
    output_dir: Path,
    split_segments: bool = True,
    segment_length: int = 128,
    chunk_size: int = 100_000,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> None:
    """Preprocess RadioML2018.01a HDF5 to memory-mapped NumPy arrays.

    Args:
        input_path: Path to GOLD_XYZ_OSC.0001_1024.hdf5
        output_dir: Output directory for preprocessed files
        split_segments: Whether to split 1024-sample signals into 128-sample segments
        segment_length: Length of each segment (if split_segments=True)
        chunk_size: Number of samples to process at a time (memory management)
        train_ratio: Fraction for training split
        val_ratio: Fraction for validation split
        test_ratio: Fraction for testing split
        random_state: Random seed for reproducible splits
    """
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "indices").mkdir(exist_ok=True)

    # Open HDF5 file and get dimensions
    print("\n1. Reading HDF5 file structure...")
    with h5py.File(input_path, "r") as f:
        n_samples = f["X"].shape[0]
        original_seq_len = f["X"].shape[1]
        n_classes = f["Y"].shape[1]

        print(f"   Original samples: {n_samples:,}")
        print(f"   Sequence length: {original_seq_len}")
        print(f"   Classes: {n_classes}")

        # Calculate output dimensions
        if split_segments:
            n_segments = original_seq_len // segment_length
            total_samples = n_samples * n_segments
            seq_len = segment_length
            print(f"   Splitting into {n_segments} segments of {segment_length} samples each")
        else:
            total_samples = n_samples
            seq_len = original_seq_len
            n_segments = 1

        print(f"   Output samples: {total_samples:,}")
        print(f"   Output shape: ({total_samples}, 2, {seq_len})")

        # Create memory-mapped output arrays
        print("\n2. Creating output arrays...")
        data_path = output_dir / "data.npy"
        labels_path = output_dir / "labels.npy"
        snrs_path = output_dir / "snrs.npy"

        # Initialize with header for .npy format
        # We need to save then reopen as memmap
        out_data_shape = (total_samples, 2, seq_len)
        out_labels_shape = (total_samples,)
        out_snrs_shape = (total_samples,)

        # Create empty arrays and save headers
        np.lib.format.open_memmap(
            str(data_path),
            mode="w+",
            dtype=np.float32,
            shape=out_data_shape,
        )
        np.lib.format.open_memmap(
            str(labels_path),
            mode="w+",
            dtype=np.int64,
            shape=out_labels_shape,
        )
        np.lib.format.open_memmap(
            str(snrs_path),
            mode="w+",
            dtype=np.float32,
            shape=out_snrs_shape,
        )

        # Reopen for writing
        out_data = np.lib.format.open_memmap(str(data_path), mode="r+")
        out_labels = np.lib.format.open_memmap(str(labels_path), mode="r+")
        out_snrs = np.lib.format.open_memmap(str(snrs_path), mode="r+")

        # Process in chunks
        print("\n3. Processing data in chunks...")
        out_idx = 0

        for start in tqdm(range(0, n_samples, chunk_size), desc="Processing"):
            end = min(start + chunk_size, n_samples)

            # Read chunk from HDF5
            X_chunk = f["X"][start:end]  # (chunk, 1024, 2)
            Y_chunk = f["Y"][start:end]  # (chunk, 24) one-hot
            Z_chunk = f["Z"][start:end]  # (chunk,)

            # Convert one-hot to indices
            labels_chunk = np.argmax(Y_chunk, axis=1)

            # Transpose to (chunk, 2, seq_len)
            X_chunk = X_chunk.transpose(0, 2, 1).astype(np.float32)

            # Segment if requested
            if split_segments and n_segments > 1:
                chunk_n = X_chunk.shape[0]
                # (chunk, 2, 1024) -> (chunk, 2, 8, 128) -> (chunk, 8, 2, 128) -> (chunk*8, 2, 128)
                X_chunk = X_chunk[:, :, : n_segments * segment_length]
                X_chunk = X_chunk.reshape(chunk_n, 2, n_segments, segment_length)
                X_chunk = X_chunk.transpose(0, 2, 1, 3)
                X_chunk = X_chunk.reshape(-1, 2, segment_length)

                # Repeat labels and SNRs for each segment
                labels_chunk = np.repeat(labels_chunk, n_segments)
                Z_chunk = np.repeat(Z_chunk, n_segments)

            # Write to output
            chunk_out_size = X_chunk.shape[0]
            out_data[out_idx : out_idx + chunk_out_size] = X_chunk
            out_labels[out_idx : out_idx + chunk_out_size] = labels_chunk
            out_snrs[out_idx : out_idx + chunk_out_size] = Z_chunk.astype(np.float32)

            out_idx += chunk_out_size

        # Flush to disk
        del out_data, out_labels, out_snrs

    # Compute stratified splits
    print("\n4. Computing stratified splits...")
    labels_for_split = np.load(str(labels_path), mmap_mode="r")
    snrs_for_split = np.load(str(snrs_path), mmap_mode="r")

    # Convert to regular arrays for sklearn (can't stratify with memmap)
    labels_arr = np.array(labels_for_split)
    snrs_arr = np.array(snrs_for_split)

    splits = stratified_split_indices(
        labels_arr,
        snrs_arr,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    print(f"   Train: {len(splits['train']):,} samples ({train_ratio*100:.0f}%)")
    print(f"   Val:   {len(splits['val']):,} samples ({val_ratio*100:.0f}%)")
    print(f"   Test:  {len(splits['test']):,} samples ({test_ratio*100:.0f}%)")

    # Save split indices
    np.save(output_dir / "indices" / "train.npy", splits["train"])
    np.save(output_dir / "indices" / "val.npy", splits["val"])
    np.save(output_dir / "indices" / "test.npy", splits["test"])

    # Write metadata
    print("\n5. Writing metadata...")
    metadata = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "source_file": input_path.name,
        "class_names": MODULATION_CLASSES_2018,
        "snr_levels": SNR_LEVELS_2018,
        "num_samples": total_samples,
        "original_samples": n_samples,
        "sequence_length": seq_len,
        "split_segments": split_segments,
        "segment_length": segment_length if split_segments else original_seq_len,
        "segments_per_sample": n_segments,
        "split_random_state": random_state,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "data_shape": list(out_data_shape),
        "dtype": "float32",
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)

    data_size_gb = (total_samples * 2 * seq_len * 4) / (1024**3)
    print(f"\nOutput files in {output_dir}:")
    print(f"   data.npy:     {data_size_gb:.2f} GB ({out_data_shape})")
    print(f"   labels.npy:   {total_samples * 8 / (1024**2):.2f} MB")
    print(f"   snrs.npy:     {total_samples * 4 / (1024**2):.2f} MB")
    print(f"   metadata.json")
    print(f"   indices/train.npy, val.npy, test.npy")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess RadioML2018.01a for efficient loading"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/GOLD_XYZ_OSC.0001_1024.hdf5"),
        help="Path to RadioML2018.01a HDF5 file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/radioml2018_preprocessed"),
        help="Output directory for preprocessed files",
    )
    parser.add_argument(
        "--no-split-segments",
        action="store_true",
        help="Don't split 1024-sample signals into 128-sample segments",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=128,
        help="Segment length when splitting (default: 128)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Samples to process at a time (default: 100000)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Training split ratio (default: 0.6)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Please download RadioML2018.01a dataset first.")
        sys.exit(1)

    preprocess_radioml2018(
        input_path=args.input,
        output_dir=args.output,
        split_segments=not args.no_split_segments,
        segment_length=args.segment_length,
        chunk_size=args.chunk_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()