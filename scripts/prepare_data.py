#!/usr/bin/env python3
from pathlib import Path

from jssp_yafs.config import load_config
from jssp_yafs.data.prepare import build_checksum_manifest, prepare_subsets


def main() -> None:
    cfg = load_config("configs/full.yaml")
    prepare_subsets(
        source_file=cfg.dataset.source_file,
        quick_instances=cfg.dataset.quick_instances,
        full_instances=cfg.dataset.full_instances,
        out_root=Path("data/processed"),
    )
    build_checksum_manifest("data/raw", "data/manifests/raw_checksums.csv")
    build_checksum_manifest("data/processed", "data/manifests/processed_checksums.csv")


if __name__ == "__main__":
    main()
