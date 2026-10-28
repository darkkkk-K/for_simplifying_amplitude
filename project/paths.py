"""Read-only path resolution for new code; frozen legacy modules stay unchanged."""
import json
from pathlib import Path, PurePosixPath
import re

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "project/layout/catalog.json"


def _relative(value):
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or ":" in text or path.is_absolute() or any(p in (".", "..") for p in text.split("/")):
        raise ValueError("expected a canonical relative repository path")
    return text


def _inside(relative, repo):
    root = Path(repo).resolve()
    result = (root / _relative(relative)).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError("resolved path escapes repository")
    return result


def asset_path(legacy_name, *, repo=REPO, catalog=None, require_exists=True):
    """Resolve a catalogued legacy asset or descendant to physical storage.

    Snapshot the specific resolved input paths/hashes in an experiment; do not
    freeze the whole mutable global catalog as a dependency of every new run.
    """
    query = _relative(legacy_name)
    root = Path(repo).resolve()
    catalog_path = Path(catalog) if catalog is not None else root / "project/layout/catalog.json"
    with catalog_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    choices = [r for r in data["entries"] if query == r["legacy"] or query.startswith(r["legacy"] + "/")]
    if not choices:
        raise KeyError(f"asset is not catalogued: {query}")
    row = max(choices, key=lambda r: len(r["legacy"]))
    suffix = query[len(row["legacy"]):].lstrip("/")
    if suffix and row["kind"] != "directory_move":
        raise ValueError("standalone file has no asset descendants")
    relative = row["physical"] + ("/" + suffix if suffix else "")
    result = _inside(relative, root)
    if require_exists and not result.exists():
        raise FileNotFoundError(result)
    return result


def run_path(experiment_id, *, repo=REPO):
    """Return, but never create, the new large-artifact directory."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,95}", experiment_id):
        raise ValueError("use a unique lower-case experiment ID")
    return _inside(f"artifacts/runs/{experiment_id}", repo)


def contract_path(experiment_id, *, repo=REPO):
    run_path(experiment_id, repo=repo)
    return _inside(f"experiments/{experiment_id}", repo)
