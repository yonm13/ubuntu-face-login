#!/usr/bin/env python3
"""Convert PyTorch .pt embedding files to NumPy .npy format.

Migration tool for moving from the old torch-based pipeline to the
ONNX-based pipeline that uses plain numpy arrays.

Usage:
    python scripts/convert-embeddings.py <source_dir> <dest_dir>
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Guarded import
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:
    print("\033[0;31m[✗]\033[0m torch is required for reading .pt files.")
    print()
    print("Install with:")
    print("    pip install torch")
    sys.exit(1)

import numpy as np


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/convert-embeddings.py <source_dir> <dest_dir>")
        print()
        print("  source_dir   Directory containing .pt embedding files")
        print("  dest_dir     Output directory for .npy files")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    if not src.is_dir():
        print(f"\033[0;31m[✗]\033[0m Source directory not found: {src}")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(src.glob("*.pt"))
    if not pt_files:
        print(f"\033[1;33m[!]\033[0m No .pt files found in {src}")
        sys.exit(0)

    converted = 0
    for pt_file in pt_files:
        npy_file = dst / pt_file.with_suffix(".npy").name

        try:
            tensor = torch.load(pt_file, map_location="cpu", weights_only=True)
            arr = tensor.numpy()

            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr.squeeze(0)

            np.save(npy_file, arr)
            converted += 1
            print(f"  \033[0;32m[✓]\033[0m {pt_file.name} → {npy_file.name}  (shape: {arr.shape})")
        except Exception as e:
            print(f"  \033[0;31m[✗]\033[0m {pt_file.name}: {e}")

    print()
    print(f"\033[0;32m[✓]\033[0m Converted {converted}/{len(pt_files)} file(s) to {dst}")


if __name__ == "__main__":
    main()
