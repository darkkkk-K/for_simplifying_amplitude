"""Shared t-SNE viewer. Import/help never load data, models or plotting backends."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import inspect
import json
import math
import os
import pickle
import re
from pathlib import Path
import uuid
import warnings

from .io import REPO, default_input, digest, load_embeddings, resolve_input, write_json


def as_matrix(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim not in (2, 3) or array.shape[-1] == 0:
        raise ValueError(f"Expected [N,D] or [B,T,D], got {array.shape}")
    array = array.reshape(-1, array.shape[-1])
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Empty/nonfinite embedding class; fix its input instead of plotting it")
    return array


def prepare_tsne_data(tensor_list):
    """Compatibility API retained by the three original plotting scripts."""
    import numpy as np
    if not tensor_list:
        raise ValueError("No embedding classes")
    matrices = [as_matrix(x) for x in tensor_list]
    if len({x.shape[1] for x in matrices}) != 1:
        raise ValueError("Embedding dimensions differ between classes")
    labels, indices, offset = [], {}, 0
    for i, matrix in enumerate(matrices):
        labels.append(np.full(len(matrix), i, dtype=np.int64))
        indices[i] = (offset, offset + len(matrix))
        offset += len(matrix)
    return np.concatenate(matrices), np.concatenate(labels), indices


def group_dimensions(groups, names, metadata, mode):
    if mode == "individual":
        return groups, names
    import numpy as np
    values = metadata.get("class_values")
    if values != list(range(2, 12)) or len(groups) != 10:
        raise ValueError("legacy_bins requires exactly dimensions 2..11 in order; use --dim-grouping individual")
    bins = ((0,), (1, 2, 3), (4, 5, 6), (7, 8, 9))
    # Flatten before concatenation: old caches can have different term axes.
    return ([np.concatenate([as_matrix(groups[i]) for i in ids]) for ids in bins],
            ["mass_dim=2", "mass_dim=3..5", "mass_dim=6..8", "mass_dim=9..11"])


def select_points(groups, limit, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    output, selections, dropped = [], [], []
    for group in groups:
        matrix = as_matrix(group)
        # Exact zero padding vectors have no direction; report their removal.
        ids = np.flatnonzero(np.any(matrix != 0, axis=1))
        dropped.append(len(matrix) - len(ids))
        if not len(ids):
            raise ValueError("A class contains only zero/padding vectors")
        if limit and len(ids) > limit:
            ids = np.sort(rng.choice(ids, size=limit, replace=False))
        output.append(matrix[ids])
        selections.append(ids)
    return output, selections, dropped


def tsne_kwargs(tsne_class, *, samples, perplexity, iterations, metric, seed):
    if samples < 3:
        raise ValueError("t-SNE requires at least 3 nonzero points for this viewer")
    if perplexity <= 0 or not math.isfinite(perplexity):
        raise ValueError("perplexity must be positive and finite")
    if iterations < 250:
        raise ValueError("iterations must be at least 250")
    effective = min(float(perplexity), max(1., (samples - 1) / 3)) if perplexity >= samples else float(perplexity)
    if effective != perplexity:
        warnings.warn(f"perplexity={perplexity} >= N={samples}; using {effective:g}", stacklevel=2)
    kwargs = dict(n_components=2, random_state=seed, perplexity=effective,
                  metric=metric, learning_rate="auto", init="random")
    # sklearn renamed n_iter to max_iter in 1.5, then removed n_iter.
    key = "max_iter" if "max_iter" in inspect.signature(tsne_class).parameters else "n_iter"
    kwargs[key] = iterations
    return kwargs


def run_tsne_analysis(tensor_list, class_names=None, perplexity=30, metric="euclidean",
                      *, iterations=768, seed=42, show=True, output=None, title=None):
    # On newer Windows, loky's physical-core probe may invoke removed WMIC.
    # The viewer already limits numeric threads to one; avoid that probe too.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    import numpy as np
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits
    embeddings, labels, indices = prepare_tsne_data(tensor_list)
    names = class_names or [f"Class {i}" for i in range(len(tensor_list))]
    if len(names) != len(tensor_list):
        raise ValueError("class_names must match the number of embedding groups")
    if len(names) == 1:
        warnings.warn("Only one class: this plot cannot measure separation between expression groups", stacklevel=2)
    kwargs = tsne_kwargs(TSNE, samples=len(embeddings), perplexity=perplexity,
                         iterations=iterations, metric=metric, seed=seed)
    print(f"Points={len(embeddings)}, embedding_dim={embeddings.shape[1]}, classes={len(names)}, perplexity={kwargs['perplexity']:g}")
    with threadpool_limits(limits=1):
        coords = TSNE(**kwargs).fit_transform(embeddings)
    if not np.isfinite(coords).all():
        raise ValueError("t-SNE produced nonfinite coordinates")
    os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".cache/matplotlib"))
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, name in enumerate(names):
        points = coords[labels == i]
        ax.scatter(points[:, 0], points[:, 1], label=name, alpha=.7, s=4)
    ax.set(title=title or f"t-SNE: D={embeddings.shape[1]}, metric={metric}, perplexity={kwargs['perplexity']:g}",
           xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    try:
        if output is not None:
            with open(output, "xb") as stream:
                fig.savefig(stream, format="png", dpi=160)
        if show:
            plt.show()
    finally:
        plt.close(fig)
    return coords, labels, indices


def build_parser(kind):
    parser = argparse.ArgumentParser(description=(
        f"{kind} encoder t-SNE: --model FILE generates data, encodes and plots in one command; "
        "without --model, view existing embeddings."))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", help="Trusted .pkl or embeddings.npz; cache-only mode")
    source.add_argument("--checkpoint", "--model", dest="checkpoint", help="Model path or basename in pretrained folder; run and plot directly")
    parser.add_argument("--list-inputs", action="store_true", help="List historical caches without loading them")
    parser.add_argument("--inspect", action="store_true", help="Inspect cache or checkpoint only; no generation/forward/t-SNE")
    parser.add_argument("--dry-run", action="store_true", help="With --model, preview budgets without loading/generating")
    parser.add_argument("--samples", "--samples-per-class", dest="samples_per_class", type=int,
                        help="ex/dim: expressions per class (model-mode default 100); each term is one plotted point")
    parser.add_argument("--groups", "--sim-groups", dest="sim_groups", type=int,
                        help="sim: number of independent expression groups (model-mode default 4)")
    parser.add_argument("--rounds", "--scramble-rounds", dest="scramble_rounds", type=int,
                        help="sim: scramble rounds per expression (default 64); more rounds usually produce more terms")
    parser.add_argument("--inputs", help="Optional saved inputs.json to reuse the exact same expressions with another model")
    parser.add_argument("--num-heads", type=int, help="Override attention heads; otherwise use metadata, or historical default 8 with a notice")
    parser.add_argument("--rope-mode", choices=("shared_learnable", "unshared_learnable", "shared_fixed", "unshared_fixed"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--external-legs", nargs="+", type=int)
    parser.add_argument("--dimensions", nargs="+", type=int)
    parser.add_argument("--max-attempts-per-class", type=int, help="Default=max(64,8*samples) for ex/dim; 64 for sim")
    parser.add_argument("--min-terms", type=int)
    parser.add_argument("--max-terms", type=int)
    parser.add_argument("--max-tokens-per-term", type=int)
    parser.add_argument("--perplexity", type=float, default=50)
    parser.add_argument("--metric", choices=("euclidean", "cosine"), default="euclidean")
    parser.add_argument("--iterations", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points-per-class", type=int,
                        help="Display limit only, not generation: 0=all; default all in model mode, 1000 in cache mode")
    parser.add_argument("--dim-grouping", choices=("individual", "legacy_bins"), default="individual")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--run-id", help="Optional output name; model mode automatically creates a unique ID if omitted")
    return parser


def automatic_run_id(kind, checkpoint):
    stem = re.sub(r"[^a-z0-9]+", "_", Path(checkpoint).stem.lower()).strip("_")[:24] or "encoder"
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    return f"pretrain_{kind}_{stem}_{stamp}_{uuid.uuid4().hex[:6]}"


def evaluation_arguments(kind, args, run_id):
    """Translate the convenient viewer CLI into the existing bounded evaluator."""
    phase = "inspect" if args.inspect else "plan" if args.dry_run else "run"
    command = ["--phase", phase, "--checkpoint", args.checkpoint,
               "--device", args.device, "--seed", str(args.seed)]
    if run_id is not None:
        command += ["--run-id", run_id]
    if args.num_heads is None:
        # This is an explicitly reported historical default, not shape inference.
        command += ["--fallback-num-heads", "8"]
    else:
        command += ["--num-heads", str(args.num_heads)]
    samples = args.samples_per_class if args.samples_per_class is not None else 100
    attempts = args.max_attempts_per_class
    if attempts is None:
        attempts = 64 if kind == "sim" else max(64, samples * 8)
    command += ["--max-attempts-per-class", str(attempts)]
    if kind == "sim":
        command += ["--sim-groups", str(args.sim_groups if args.sim_groups is not None else 4),
                    "--scramble-rounds", str(args.scramble_rounds if args.scramble_rounds is not None else 64)]
    else:
        command += ["--samples-per-class", str(samples)]
    for name in ("inputs", "rope_mode", "min_terms", "max_terms", "max_tokens_per_term"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    for name in ("external_legs", "dimensions"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), *map(str, value)]
    return command


def main(kind, argv=None):
    parser = build_parser(kind)
    args = parser.parse_args(argv)
    if args.checkpoint and args.list_inputs:
        parser.error("--list-inputs cannot be combined with --model/--checkpoint")
    if args.inspect and args.dry_run:
        parser.error("Choose either --inspect or --dry-run")
    generation_fields = ("inputs", "samples_per_class", "sim_groups", "scramble_rounds", "num_heads",
                         "rope_mode", "external_legs", "dimensions", "max_attempts_per_class", "min_terms",
                         "max_terms", "max_tokens_per_term")
    if not args.checkpoint and (args.dry_run or any(getattr(args, key) is not None for key in generation_fields)):
        parser.error("Generation options require --model/--checkpoint; use --max-points-per-class to display more cached points")
    if kind == "sim" and args.samples_per_class is not None:
        parser.error("sim uses one expression per group: use --groups and --rounds, not --samples")
    if kind != "sim" and (args.sim_groups is not None or args.scramble_rounds is not None):
        parser.error("--groups/--rounds apply to visualize_tsne_sim.py only; ex/dim use --samples")
    if args.inputs and (args.samples_per_class is not None or args.sim_groups is not None or args.scramble_rounds is not None):
        parser.error("--inputs reuses fixed expressions; do not combine it with new sample/group/round counts")
    if args.max_points_per_class is None:
        args.max_points_per_class = 0 if args.checkpoint else 1000
    if args.list_inputs:
        from project.paths import asset_path
        prefix = {"ex": "testdata_", "dim": "testdatadim_", "sim": "testdatasim_"}[kind]
        for path in sorted(asset_path("outputs/test_data").glob(prefix + "*.pkl")):
            print(path)
        return 0
    if args.max_points_per_class < 0:
        parser.error("--max-points-per-class must be nonnegative")
    if not math.isfinite(args.perplexity) or args.perplexity <= 0 or args.iterations < 250:
        parser.error("perplexity must be positive/finite and iterations >=250")
    output = contract = None
    plot_owned = False
    try:
        if args.checkpoint:
            from . import evaluation
            from project.paths import run_path, contract_path
            if kind == "dim" and args.dim_grouping == "legacy_bins":
                values = args.dimensions
                if args.inputs:
                    values = json.loads(resolve_input(args.inputs).read_text(encoding="utf-8")).get("class_values")
                if values is not None and values != list(range(2, 12)):
                    raise ValueError("legacy_bins requires dimensions 2..11; choose individual")
            if args.inspect or args.dry_run:
                return evaluation.main(kind, evaluation_arguments(kind, args, args.run_id))
            args.run_id = args.run_id or automatic_run_id(kind, args.checkpoint)
            print(f"One-command evaluation: {args.run_id}", flush=True)
            evaluation.main(kind, evaluation_arguments(kind, args, args.run_id))
            output, contract = run_path(args.run_id), contract_path(args.run_id)
            path = output / "embeddings.npz"
            plot_owned = True
            print(f"Reusable inputs: {output / 'inputs.json'}", flush=True)
        else:
            path = resolve_input(args.input) if args.input else default_input(kind)
        groups, names, metadata = load_embeddings(path, kind)
        if kind == "dim":
            groups, names = group_dimensions(groups, names, metadata, args.dim_grouping)
        matrices = [as_matrix(x) for x in groups]
        if len({x.shape[1] for x in matrices}) != 1:
            raise ValueError("Embedding dimensions differ between groups")
        print(json.dumps({"input": str(path), "shapes": [list(x.shape) for x in matrices],
                          "class_names": names, "metadata": metadata}, ensure_ascii=False, indent=2))
        if args.inspect:
            return 0
        groups, selections, dropped = select_points(matrices, args.max_points_per_class, args.seed)
        print(f"Zero/padding rows removed per class: {dropped}; selected points: {[len(x) for x in groups]}")
        if args.run_id and not args.checkpoint:
            from project.paths import run_path, contract_path
            output, contract = run_path(args.run_id), contract_path(args.run_id)
            if output.exists() or contract.exists():
                raise FileExistsError("Run ID already exists; choose a new --run-id")
            output.mkdir(parents=True); contract.mkdir(parents=True)
            plot_owned = True
        if output is not None:
            protocol_name = "plot_protocol.json" if args.checkpoint else "protocol.json"
            write_json(contract / protocol_name, {"kind": "tsne_view", "args": vars(args),
                       "input": str(path), "input_sha256": digest(path), "source_metadata": metadata,
                       "source_sha256": {p: digest(REPO / p) for p in (
                           "project/pretraining/visualization.py", f"scripts/visualize_tsne_{kind}.py")}})
        coords, labels, _ = run_tsne_analysis(groups, names, args.perplexity, args.metric,
            iterations=args.iterations, seed=args.seed, show=not args.no_show,
            output=None if output is None else output / "tsne.png")
        if output is not None:
            import numpy as np
            with (output / "projection.npz").open("xb") as stream:
                np.savez_compressed(stream, coordinates=coords, labels=labels,
                                    **{f"selected_{i}": ids for i, ids in enumerate(selections)})
            summary_name = "plot_summary.json" if args.checkpoint else "summary.json"
            write_json(contract / summary_name, {"status": "completed", "scope": "plotting_only", "embedding_dim": matrices[0].shape[1],
                       "points": len(coords), "zero_rows_removed": dropped, "output": str(output),
                       "model_forwards": 0, "symbolic_calls": 0})
            print(f"Plot saved: {output / 'tsne.png'}", flush=True)
        return 0
    except (ValueError, OSError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        if plot_owned and output is not None and not (output / "plot_failed.json").exists():
            write_json(output / "plot_failed.json", {"status": "failed", "scope": "plotting_only", "error": repr(exc)})
        parser.exit(2, f"Error: {exc}\nSee docs/PRETRAIN_EVALUATION.md\n")
