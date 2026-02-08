from __future__ import annotations

from pathlib import Path

import pandas as pd

from .loader import parse_instance_corpus, save_instance_txt, sha256_file


def prepare_subsets(
    source_file: str | Path,
    quick_instances: list[str],
    full_instances: list[str],
    out_root: str | Path,
) -> None:
    corpus = parse_instance_corpus(source_file)
    out_root = Path(out_root)
    quick_dir = out_root / "quick"
    full_dir = out_root / "full"
    quick_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)

    for name in quick_instances:
        save_instance_txt(corpus[name], quick_dir / f"{name}.txt")

    for name in full_instances:
        save_instance_txt(corpus[name], full_dir / f"{name}.txt")



def build_checksum_manifest(data_root: str | Path, manifest_path: str | Path) -> None:
    data_root = Path(data_root)
    entries: list[dict[str, str]] = []
    for file_path in sorted(data_root.rglob("*.txt")):
        entries.append(
            {
                "path": str(file_path),
                "sha256": sha256_file(file_path),
            }
        )

    df = pd.DataFrame(entries)
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
