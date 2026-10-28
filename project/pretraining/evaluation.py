"""Manual encoder diagnostics: explicit budgets, reusable inputs, no training."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path

from .io import REPO, digest, resolve_input, write_json
from .encoder import construct_encoder, load_definitions, read_checkpoint, resolve_checkpoint


def build_parser(kind):
    p = argparse.ArgumentParser(description=f"Pretrained encoder {kind} evaluation. Default=plan; no implicit model/generation.")
    p.add_argument("--phase", choices=("plan", "inspect", "generate", "run"), default="plan")
    p.add_argument("--checkpoint", help="Standalone pretrained .pth/.pt; repository-relative or absolute")
    p.add_argument("--num-heads", type=int, help="Required when checkpoint metadata/shape cannot determine head count")
    p.add_argument("--fallback-num-heads", type=int, help="Explicit fallback used only when metadata/shape do not determine heads")
    p.add_argument("--rope-mode", choices=("shared_learnable", "unshared_learnable", "shared_fixed", "unshared_fixed"))
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--inputs", help="Reuse inputs.json from a previous run; zero generation calls")
    p.add_argument("--run-id", help="Unique ID for experiments/<id> and artifacts/runs/<id>")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--samples-per-class", type=int, default=8)
    p.add_argument("--external-legs", nargs="+", type=int, default=[3, 4, 5, 6])
    p.add_argument("--dimensions", nargs="+", type=int, default=list(range(4 if kind == "ex" else 2, 12)))
    p.add_argument("--sim-groups", type=int, default=2)
    p.add_argument("--scramble-rounds", type=int, default=64, help="sim: each round=Schouten then momentum conservation")
    p.add_argument("--max-attempts-per-class", type=int, default=64)
    p.add_argument("--min-terms", type=int, default=1)
    p.add_argument("--max-terms", type=int, default=512)
    p.add_argument("--max-tokens-per-term", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p


def validate_args(args):
    if not set(args.external_legs).issubset({3, 4, 5, 6}) or len(set(args.external_legs)) != len(args.external_legs):
        raise ValueError("--external-legs must be distinct values from 3,4,5,6")
    if any(x < 2 for x in args.dimensions) or len(set(args.dimensions)) != len(args.dimensions):
        raise ValueError("--dimensions must contain distinct integers >=2")
    if min(args.samples_per_class, args.sim_groups, args.max_attempts_per_class,
           args.min_terms, args.max_terms, args.max_tokens_per_term) <= 0:
        raise ValueError("Sample counts and capacity bounds must be positive")
    if args.min_terms > args.max_terms or args.scramble_rounds < 0:
        raise ValueError("Invalid term range/scramble rounds")


def planned_caps(args, kind, corpus=None):
    if corpus is not None:
        return {"generation_attempts": 0, "config_generations": 0, "monomial_calls": 0,
                "identity_calls": 0, "encoder_forwards": len(corpus["records"]) if args.phase == "run" else 0,
                "checkpoint_loads": int(args.phase == "run")}
    classes = len(args.external_legs) if kind == "ex" else len(args.dimensions) if kind == "dim" else args.sim_groups
    per_class = 1 if kind == "sim" else args.samples_per_class
    if args.max_attempts_per_class < per_class:
        raise ValueError("max-attempts-per-class cannot be smaller than requested samples")
    attempts = classes * args.max_attempts_per_class
    return {"generation_attempts": attempts, "config_generations": attempts,
            "monomial_calls": attempts * (4 if kind == "sim" else 1),
            "identity_calls": attempts * (2 * args.scramble_rounds if kind == "sim" else 3),
            "encoder_forwards": classes * per_class if args.phase == "run" else 0,
            "checkpoint_loads": int(args.phase == "run")}


class CallCounter:
    def __init__(self, caps, event_path=None):
        self.caps = dict(caps)
        self.counts = {key: 0 for key in caps}
        self.event_path = event_path

    def event(self, row):
        if self.event_path is not None:
            with self.event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def tick(self, key):
        if self.counts[key] >= self.caps[key]:
            raise RuntimeError(f"Budget exhausted: {key}")
        self.counts[key] += 1

    def wrapped(self, function, key, name):
        def invoke(*args):
            self.tick(key)
            row = {"api": name, "category": key, "serial": self.counts[key]}
            if name == "random_apply_momentum_conservation":
                row["n_point"] = int(args[1])
            self.event({**row, "status": "started"})
            try:
                result = function(*args)
            except Exception as exc:
                self.event({**row, "status": "failed", "error": repr(exc)})
                raise
            self.event({**row, "status": "completed"})
            return result
        return invoke


def generation_functions(api, counts):
    """Original pretraining recipe, without importing data/__init__.py."""
    import numpy as np
    namespace = {"__name__": "pretraining_generation_definitions", "np": np,
                 "random": random, "math": math, "Tuple": tuple, "List": list}
    for name, category in (("random_monomial", "monomial_calls"),
                           ("random_apply_schouten", "identity_calls"),
                           ("random_apply_momentum_conservation", "identity_calls")):
        namespace[name] = counts.wrapped(getattr(api, name), category, name)
    load_definitions("data/generation.py", ["amplitude_config_gen", "randomly_scramble"], namespace)
    return namespace


def similarity_sample(config, functions, rounds):
    helicities, dimension = config
    # Old eval_sim used len(config)==2 here. It must be the number of legs.
    n_point = len(helicities)
    expr = sum(random.randint(1, 10) * functions["random_monomial"](helicities, dimension)
               for _ in range(4))
    initial_terms = len(expr.terms())
    if not expr.tokenize():
        return [], initial_terms
    for _ in range(rounds):
        expr = functions["random_apply_schouten"](expr, 1)[1]
        expr = functions["random_apply_momentum_conservation"](expr, n_point, 1)[1]
    return expr.tokenize(), initial_terms


def validate_tokens(tokens, vocab, args):
    if not isinstance(tokens, list) or not args.min_terms <= len(tokens) <= args.max_terms:
        return "empty_or_term_capacity"
    for term in tokens:
        if not isinstance(term, list) or not 0 < len(term) <= args.max_tokens_per_term:
            return "token_capacity"
        if not all(isinstance(token, str) and token in vocab for token in term):
            raise ValueError("Unknown token in generated/reused input; refusing silent replacement")
    return None


def generate_corpus(args, kind, counts, api, vocab):
    import numpy as np
    functions = generation_functions(api, counts)
    values = args.external_legs if kind == "ex" else args.dimensions if kind == "dim" else list(range(args.sim_groups))
    groups, records = [], []
    target = 1 if kind == "sim" else args.samples_per_class
    for class_index, value in enumerate(values):
        name = f"n={value}" if kind == "ex" else f"mass_dim={value}" if kind == "dim" else f"group={value}"
        groups.append(name)
        accepted = 0
        print(f"Generating {name}: target={target}, attempt_limit={args.max_attempts_per_class}", flush=True)
        for attempt in range(args.max_attempts_per_class):
            if accepted == target:
                break
            counts.tick("generation_attempts")
            counts.tick("config_generations")
            if kind == "sim":
                weights = np.exp(np.array(args.external_legs, dtype=float) / 1.8)
                n_point = int(np.random.choice(args.external_legs, p=weights / weights.sum()))
            else:
                n_point = int(value if kind == "ex" else random.choice(args.external_legs))
            dimension = int(value if kind == "dim" else random.choice(args.dimensions))
            config = functions["amplitude_config_gen"](n_point, dimension)
            if kind == "sim":
                tokens, initial = similarity_sample(config, functions, args.scramble_rounds)
            else:
                # Exactly one scramble per attempt. Never call it again in an if-clause.
                tokens, initial = functions["randomly_scramble"](config), None
            reason = validate_tokens(tokens, vocab, args)
            counts.event({"type": "sample", "class_index": class_index, "attempt": attempt,
                          "accepted": reason is None, "reason": reason, "n_point": n_point,
                          "mass_dimension": dimension, "term_count": len(tokens)})
            if reason is not None:
                continue
            row = {"class_index": class_index, "n_point": n_point, "mass_dimension": dimension,
                   "helicities": config[0], "tokens": tokens,
                   "source_id": f"class_{class_index:03d}_sample_{accepted:04d}"}
            if kind == "sim":
                row["initial_terms_for_plot_label_only"] = initial
                groups[-1] = f"group={value}, initial_terms={initial}, n={n_point}, mass_dim={dimension}"
            records.append(row)
            accepted += 1
        if accepted != target:
            raise RuntimeError(f"Class {name}: accepted {accepted}/{target} within fixed attempt cap; no automatic retry")
        print(f"Generated {name}: {accepted} expressions", flush=True)
    return {"schema": "pretraining_inputs_v1", "kind": kind, "class_names": groups,
            "class_values": list(values), "records": records,
            "seed_note": "Python/NumPy seeded; native Rust RNG is not guaranteed. Reuse these tokens for comparisons."}


def read_corpus(path, kind, args, vocab):
    path = resolve_input(path)
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if corpus.get("schema") != "pretraining_inputs_v1" or corpus.get("kind") != kind:
        raise ValueError("Input corpus schema/task mismatch")
    names, records = corpus.get("class_names"), corpus.get("records")
    values = corpus.get("class_values")
    if (not isinstance(names, list) or not names or not isinstance(records, list) or not records
            or not all(isinstance(name, str) for name in names)
            or not isinstance(values, list) or len(values) != len(names)):
        raise ValueError("Empty/malformed input corpus")
    populated = set()
    for row in records:
        if type(row.get("class_index")) is not int or not 0 <= row["class_index"] < len(names):
            raise ValueError("Invalid class index")
        if row.get("n_point") not in (3, 4, 5, 6) or validate_tokens(row["tokens"], vocab, args):
            raise ValueError("Reused input violates fixed particle/token/term constraints")
        if kind == "ex" and row["n_point"] != values[row["class_index"]]:
            raise ValueError("External-leg class label does not match the input")
        if kind == "dim" and row.get("mass_dimension") != values[row["class_index"]]:
            raise ValueError("Mass-dimension class label does not match the input")
        populated.add(row["class_index"])
    if populated != set(range(len(names))):
        raise ValueError("Some input classes contain no records")
    return corpus


def encode_corpus(corpus, model, vocab, config, device, counts):
    import numpy as np
    import torch
    groups = [[] for _ in corpus["class_names"]]
    for record_index, row in enumerate(corpus["records"]):
        tokens = row["tokens"]
        ids = [torch.tensor([vocab[token] for token in term], dtype=torch.long) for term in tokens]
        src = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0).unsqueeze(0).to(device)
        counts.tick("encoder_forwards")
        with torch.inference_mode():
            output = model(src)
        if tuple(output.shape) != (1, len(tokens), config["emb_dim"]) or not torch.isfinite(output).all():
            raise ValueError("Encoder returned wrong shape/nonfinite embeddings")
        if (output.norm(dim=-1) == 0).any():
            raise ValueError("Encoder returned a zero embedding for a real term")
        groups[row["class_index"]].append(output[0].float().cpu().numpy())
        if (record_index + 1) % max(1, len(corpus["records"]) // 10) == 0 or record_index + 1 == len(corpus["records"]):
            print(f"Encoded {record_index + 1}/{len(corpus['records'])} expressions (D={config['emb_dim']})", flush=True)
    return [np.concatenate(group, axis=0) for group in groups]


def main(kind, argv=None):
    parser = build_parser(kind)
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        if args.list_models:
            from project.paths import asset_path
            for path in sorted(asset_path("checkpoints").iterdir()):
                if path.suffix in (".pth", ".pt"):
                    print(path)
            return 0
        if args.phase == "plan":
            planned_args = argparse.Namespace(**{**vars(args), "phase": "run"})
            planned_corpus = None
            if args.inputs:
                vocab = json.loads((REPO / "data/tokenizor.json").read_text(encoding="utf-8"))
                planned_corpus = read_corpus(args.inputs, kind, args, vocab)
            print(json.dumps({"phase": "plan", "kind": kind, "args": vars(args),
                              "caps_if_run": planned_caps(planned_args, kind, planned_corpus),
                              "note": "No work executed. Use --phase inspect, generate, or run. See docs/PRETRAIN_EVALUATION.md."}, indent=2))
            return 0
        if args.phase in ("inspect", "run") and not args.checkpoint:
            raise ValueError("--checkpoint is required; no checkpoint is silently selected")
        if args.phase in ("inspect", "run"):
            args.checkpoint = str(resolve_checkpoint(args.checkpoint))
        if args.phase == "inspect":
            _, config = read_checkpoint(args.checkpoint, num_heads=args.num_heads, rope_mode=args.rope_mode,
                                        fallback_num_heads=args.fallback_num_heads)
            print(json.dumps({"checkpoint": str(resolve_input(args.checkpoint)), "config": config,
                              "checkpoint_loads": 1, "encoder_forwards": 0, "generation_calls": 0}, indent=2))
            return 0
        if not args.run_id:
            raise ValueError("--run-id is required for generate/run; existing runs are never overwritten")
        if args.phase == "generate" and args.inputs:
            raise ValueError("generate does not accept --inputs; use run to encode an existing corpus")
        from project.paths import run_path, contract_path
        out, contract = run_path(args.run_id), contract_path(args.run_id)
        if out.exists() or contract.exists():
            raise FileExistsError("Run ID already exists; choose a new ID, do not overwrite/retry partial output")
        vocab = json.loads((REPO / "data/tokenizor.json").read_text(encoding="utf-8"))
        corpus = read_corpus(args.inputs, kind, args, vocab) if args.inputs else None
        caps = planned_caps(args, kind, corpus)
        if args.device == "auto":
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        source_files = ["models/encoder.py", "models/custom_encoder.py", "models/rope_bias_a.py",
                        "models/rope_bias_b.py", "models/rope_bias_c.py", "data/generation.py",
                        "data/tokenizor.json", "project/pretraining/encoder.py", "project/pretraining/evaluation.py",
                        "project/pretraining/io.py", f"scripts/eval_{kind}.py"]
        protocol = {"kind": kind, "args": vars(args), "caps": caps,
                    "source_sha256": {p: digest(REPO / p) for p in source_files},
                    "checkpoint_sha256": digest(resolve_input(args.checkpoint)) if args.phase == "run" else None,
                    "inputs_sha256": digest(resolve_input(args.inputs)) if args.inputs else None}
        out.mkdir(parents=True); contract.mkdir(parents=True)
        write_json(contract / "protocol.json", protocol)
        counts = CallCounter(caps, out / "events.jsonl")
        try:
            import numpy as np
            random.seed(args.seed); np.random.seed(args.seed)
            model = config = None
            if args.phase == "run":
                import torch
                torch.manual_seed(args.seed)
                counts.tick("checkpoint_loads")
                state, config = read_checkpoint(args.checkpoint, num_heads=args.num_heads, rope_mode=args.rope_mode,
                                                fallback_num_heads=args.fallback_num_heads)
                print(json.dumps({"loaded_encoder": config}, indent=2))
                if max(vocab.values()) >= config["vocab_size"]:
                    raise ValueError("Checkpoint vocabulary is smaller than the current tokenizer; refusing invalid embedding indices")
                model = construct_encoder(state, config, args.device)
                del state
            if corpus is None:
                import forsythia_hep
                corpus = generate_corpus(args, kind, counts, forsythia_hep, vocab)
            write_json(out / "inputs.json", corpus)
            metadata = {"kind": kind, "class_names": corpus["class_names"], "class_values": corpus["class_values"],
                        "config": config, "checkpoint": args.checkpoint if model is not None else None,
                        "checkpoint_sha256": protocol["checkpoint_sha256"], "inputs_sha256": digest(out / "inputs.json"),
                        "schema": "pretraining_embeddings_v1"}
            if model is not None:
                groups = encode_corpus(corpus, model, vocab, config, args.device, counts)
                with (out / "embeddings.npz").open("xb") as stream:
                    np.savez_compressed(stream, metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
                                        **{f"class_{i:03d}": group for i, group in enumerate(groups)})
                metadata["embedding_shapes"] = [list(group.shape) for group in groups]
            summary = {"status": "completed", **metadata, "counts": counts.counts,
                       "output": str(out), "training_updates": 0}
            write_json(contract / "summary.json", summary)
            write_json(out / "completed.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        except Exception as exc:
            write_json(out / "failed.json", {"error": repr(exc), "counts": counts.counts, "status": "failed"})
            raise
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, pickle.UnpicklingError) as exc:
        parser.exit(2, f"Error: {exc}\nSee docs/PRETRAIN_EVALUATION.md\n")
