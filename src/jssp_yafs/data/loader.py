from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .models import JSSPInstance

_INSTANCE_RE = re.compile(r"^instance\s+([a-zA-Z0-9_]+)\s*$")
_DIM_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")


class DatasetError(RuntimeError):
    """Raised when the benchmark dataset cannot be parsed."""



def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def verify_checksums(manifest_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    statuses: list[bool] = []
    observed: list[str] = []
    for _, row in df.iterrows():
        digest = sha256_file(row["path"])
        observed.append(digest)
        statuses.append(digest == row["sha256"])
    out = df.copy()
    out["observed_sha256"] = observed
    out["ok"] = statuses
    return out



def parse_instance_corpus(path: str | Path) -> dict[str, JSSPInstance]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    instances: dict[str, JSSPInstance] = {}

    i = 0
    while i < len(lines):
        match = _INSTANCE_RE.match(lines[i].strip())
        if not match:
            i += 1
            continue

        name = match.group(1)
        i += 1

        while i < len(lines) and _DIM_RE.match(lines[i]) is None:
            i += 1
        if i >= len(lines):
            raise DatasetError(f"Could not find dimensions for instance {name}")

        dim_match = _DIM_RE.match(lines[i])
        if dim_match is None:
            raise DatasetError(f"Malformed dimensions for instance {name}")
        n_jobs = int(dim_match.group(1))
        n_machines = int(dim_match.group(2))
        i += 1

        rows: list[list[int]] = []
        while i < len(lines) and len(rows) < n_jobs:
            tokens = re.findall(r"-?\d+", lines[i])
            if len(tokens) >= 2 * n_machines:
                vals = list(map(int, tokens[: 2 * n_machines]))
                rows.append(vals)
            i += 1

        if len(rows) != n_jobs:
            raise DatasetError(f"Could not read {n_jobs} job rows for instance {name}")

        machine = np.zeros((n_jobs, n_machines), dtype=np.int64)
        proc = np.zeros((n_jobs, n_machines), dtype=np.float64)
        for j, row in enumerate(rows):
            machine[j, :] = np.array(row[::2], dtype=np.int64)
            proc[j, :] = np.array(row[1::2], dtype=np.float64)

        instances[name] = JSSPInstance(
            name=name,
            n_jobs=n_jobs,
            n_machines=n_machines,
            machine_matrix=machine,
            processing_matrix=proc,
        )

    if not instances:
        raise DatasetError("No instances parsed from corpus")

    return instances



def save_instance_txt(instance: JSSPInstance, target: str | Path) -> None:
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{instance.n_jobs} {instance.n_machines}"]
    for j in range(instance.n_jobs):
        pairs: list[str] = []
        for k in range(instance.n_machines):
            pairs.extend(
                [str(int(instance.machine_matrix[j, k])), str(int(instance.processing_matrix[j, k]))]
            )
        lines.append(" ".join(pairs))
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def load_instance_txt(path: str | Path, name: str | None = None) -> JSSPInstance:
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    first = lines[0].split()
    n_jobs, n_machines = int(first[0]), int(first[1])
    machine = np.zeros((n_jobs, n_machines), dtype=np.int64)
    proc = np.zeros((n_jobs, n_machines), dtype=np.float64)

    if len(lines) < n_jobs + 1:
        raise DatasetError(f"File {path} missing job rows")

    for j in range(n_jobs):
        tokens = list(map(int, lines[j + 1].split()))
        if len(tokens) != 2 * n_machines:
            raise DatasetError(f"Row {j+1} in {path} has {len(tokens)} values; expected {2*n_machines}")
        machine[j, :] = np.array(tokens[::2], dtype=np.int64)
        proc[j, :] = np.array(tokens[1::2], dtype=np.float64)

    return JSSPInstance(
        name=name or Path(path).stem,
        n_jobs=n_jobs,
        n_machines=n_machines,
        machine_matrix=machine,
        processing_matrix=proc,
    )



def load_instances_from_folder(folder: str | Path, names: list[str]) -> dict[str, JSSPInstance]:
    out: dict[str, JSSPInstance] = {}
    for name in names:
        out[name] = load_instance_txt(Path(folder) / f"{name}.txt", name=name)
    return out
