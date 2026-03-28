#!/usr/bin/env python3
"""Export FaceNet InceptionResnetV1 to ONNX format.

Dev-only script. Requires: torch, facenet-pytorch, onnxscript
Optional: onnx (for merging external data into a single file)

Usage:
    python scripts/export-onnx.py [output_path]
"""

import hashlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Guarded imports with helpful errors
# ---------------------------------------------------------------------------
_MISSING = []

try:
    import torch
except ImportError:
    _MISSING.append("torch")

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    _MISSING.append("facenet-pytorch")

try:
    import onnxscript  # noqa: F401 — needed for modern torch.onnx.export
except ImportError:
    _MISSING.append("onnxscript")

if _MISSING:
    print(f"\033[0;31m[✗]\033[0m Missing dependencies: {', '.join(_MISSING)}")
    print()
    print("Install them with:")
    print(f"    pip install {' '.join(_MISSING)}")
    sys.exit(1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("facenet512.onnx")

    print("Loading InceptionResnetV1(pretrained='vggface2')...")
    model = InceptionResnetV1(pretrained="vggface2").eval()

    dummy = torch.randn(1, 3, 160, 160)

    print(f"Exporting to {output}...")
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
    )

    # Check for external data file (large models sometimes produce one)
    external_data = Path(str(output) + ".data")
    if external_data.exists():
        print("External data file detected — merging into single ONNX file...")
        try:
            import onnx
            from onnx.external_data_helper import convert_model_to_external_data

            model_proto = onnx.load(str(output), load_external_data=True)
            # Save as single file (no external data)
            onnx.save_model(
                model_proto,
                str(output),
                save_as_external_data=False,
            )
            external_data.unlink()
            print("Merged into single file.")
        except ImportError:
            print("\033[1;33m[!]\033[0m 'onnx' package not installed — cannot merge external data.")
            print(f"    External data file remains: {external_data}")
            print("    Install with: pip install onnx")

    size_mb = output.stat().st_size / (1024 * 1024)
    digest = sha256_file(output)

    print()
    print(f"\033[0;32m[✓]\033[0m Exported: {output}")
    print(f"    Size:   {size_mb:.1f} MB")
    print(f"    SHA256: {digest}")


if __name__ == "__main__":
    main()
