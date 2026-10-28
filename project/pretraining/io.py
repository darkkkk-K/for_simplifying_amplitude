"""Paths and explicit, read-only loading of trusted historical embeddings."""
from __future__ import annotations

import hashlib
import io
import json
import pickle
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = {
    "ex": "testdata_256_temp15_sl_final_lite.pkl",
    "dim": "testdatadim_256_temp15_sl_final_lite.pkl",
    "sim": "testdatasim_256_temp15_sl_final_lite.pkl",
}


def resolve_input(value):
    """Explicit relative paths are repository-relative, never cwd-dependent."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def default_input(kind):
    from project.paths import asset_path
    return asset_path("outputs/test_data/" + DEFAULT_INPUTS[kind])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


class CPUUnpickler(pickle.Unpickler):
    """Map embedded Torch storage to CPU. Pickle input must still be trusted."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda payload: torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        return super().find_class(module, name)


def load_embeddings(path, kind):
    """Read legacy pickle or the new non-pickle NPZ. No model is constructed."""
    import numpy as np
    path = resolve_input(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"].item()))
            if metadata.get("kind") != kind:
                raise ValueError(f"Expected {kind} embeddings, got {metadata.get('kind')}")
            names = metadata["class_names"]
            groups = [saved[f"class_{i:03d}"].copy() for i in range(len(names))]
        return groups, names, metadata
    if path.suffix.lower() not in (".pkl", ".pickle"):
        raise ValueError("Expected trusted .pkl/.pickle or an embeddings.npz file")
    with path.open("rb") as stream:
        payload = CPUUnpickler(stream).load()
    metadata = {"kind": kind, "format": "legacy_pickle", "checkpoint": "unknown: not recorded in old cache"}
    if kind == "sim":
        if not isinstance(payload, (list, tuple)) or len(payload) != 4:
            raise ValueError("Similarity cache must contain (embeddings, initial_terms, n_points, dimensions)")
        groups, initial, points, dimensions = payload
        if not all(len(x) == len(groups) for x in (initial, points, dimensions)):
            raise ValueError("Similarity metadata lengths do not match embedding classes")
        names = [f"group={i}, initial_terms={initial[i]}, n={points[i]}, mass_dim={dimensions[i]}"
                 for i in range(len(groups))]
    else:
        groups = payload
        if not isinstance(groups, (list, tuple)) or not groups:
            raise ValueError("Legacy cache must be a nonempty list of embedding tensors")
        start = 3 if kind == "ex" else 2
        metadata["class_values"] = list(range(start, start + len(groups)))
        metadata["label_source"] = "legacy eval script ordering; not independently stored in cache"
        names = [f"n={v}" if kind == "ex" else f"mass_dim={v}" for v in metadata["class_values"]]
    return groups, names, metadata
