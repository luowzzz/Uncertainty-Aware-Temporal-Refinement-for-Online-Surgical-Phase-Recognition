from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    source_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_inference_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_fresh_output_dir(path: str | Path, allow_overwrite: bool = False) -> Path:
    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(
            "Refusing to overwrite a non-empty output directory. Choose a new output "
            f"directory or explicitly allow overwrite: {output_dir}"
        )
    return output_dir
