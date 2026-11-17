#!/usr/bin/env python3
"""Download and SHA256-verify NativeSeg & baseline pretrained checkpoints from Hugging Face Hub."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dir(weights_dir: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    for item in manifest:
        pt = weights_dir / item["name"] / "trainable.pt"
        if not pt.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {pt}")
        digest = sha256_file(pt)
        if digest != item["sha256"]:
            raise ValueError(
                f"SHA256 mismatch for {pt}: expected {item['sha256']}, got {digest}"
            )
        sz_mib = item.get("size_mib", item.get("size_mb", 0.0))
        print(f"[OK] {item['name']:34s} ({sz_mib:6.2f} MiB) sha256={digest[:16]}...")


def main():
    parser = argparse.ArgumentParser(description="Download or verify NativeSeg checkpoints.")
    parser.add_argument("--repo-id", default="ChengqianMa/NativeSeg", help="Hugging Face repo ID")
    parser.add_argument("--output", default="checkpoints", help="Destination folder")
    parser.add_argument("--verify-only", action="store_true", help="Verify an already-downloaded folder")
    args = parser.parse_args()

    manifest_path = Path(__file__).resolve().parents[1] / "configs" / "weights_manifest.json"
    out_dir = Path(args.output)
    if not args.verify_only:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=args.repo_id, local_dir=str(out_dir))
    verify_dir(out_dir, manifest_path)


if __name__ == "__main__":
    main()
