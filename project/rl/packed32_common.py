"""Audited <=32-term experiment utilities; imports do not load models or engines."""
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import statistics

REPO = Path(__file__).resolve().parents[2]
RUN_ID = "packed32_20261023_v1"
TRAIN_SEEDS = (103, 211, 307)
EVAL_SEEDS = (401, 503)
ROLLOUTS, CALLS, VISITS = 4, 64, 4


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def key(*values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def seed_for(*values): return int(key(RUN_ID, *values)[:15], 16)


def locations():
    from project.paths import run_path, contract_path
    return run_path(RUN_ID), contract_path(RUN_ID)


def expression_from_tokens(tokens):
    import forsythia_hep as api
    from .legal_actions import parse_terms
    result = 0 * api.abk(1, 2)
    for row in parse_terms(tokens):
        term = row.coefficient
        for factor in row.factors:
            constructor = api.abk if factor.kind == "angle" else api.sbk
            term = term * constructor(*factor.labels) ** factor.power
        result = result + term
    if result.tokenize() != tokens:
        raise ValueError("Exact token reconstruction changed order/content")
    return result


def tensor_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def paired(candidate, reference):
    differences = [a-b for a, b in zip(candidate, reference)]
    rng = random.Random(2026102301)
    boot = sorted(statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    return {"mean": statistics.fmean(differences), "ci95": [boot[249], boot[9749]], "sources": len(differences)}


def advantages(episodes):
    if len(episodes) != ROLLOUTS or len({e["seed"] for e in episodes}) != ROLLOUTS:
        raise ValueError("Four independent rollout seeds required")
    if len({(e["initial"], e["initial_terms"]) for e in episodes}) != 1:
        raise ValueError("Initial inputs differ")
    rtg = [[(r["before"]-e["final_terms"])/e["initial_terms"] for r in e["rounds"]] for e in episodes]
    return [[value - sum(rtg[j][t] if t < len(rtg[j]) else 0. for j in range(ROLLOUTS) if j != i)/(ROLLOUTS-1)
             for t, value in enumerate(row)] for i, row in enumerate(rtg)]


class Ledger:
    def __init__(self, caps): self.caps, self.counts = caps, Counter()
    def hit(self, name, amount=1):
        if name not in self.caps or self.counts[name] + amount > self.caps[name]:
            raise RuntimeError("Budget exceeded: " + name)
        self.counts[name] += amount


def source_hashes(extra=()):
    names = ["project/rl/legal_actions.py", "project/rl/packed_policy.py", "project/rl/packed_sampling.py",
             "project/rl/packed_factory.py", "project/rl/packed32_common.py", "project/paths.py",
             "project/pretraining/encoder.py", "project/pretraining/io.py", "data/tokenizor.json",
             "models/encoder.py", "models/custom_encoder.py", "models/rl_policy.py",
             "models/rope_bias_a.py", "models/rope_bias_b.py", "models/rope_bias_c.py"]
    return {name: digest(REPO / name) for name in sorted(set(names) | set(extra))}


def check_sources(protocol):
    for name, expected in protocol["code_sha256"].items():
        if digest(REPO / name) != expected: raise ValueError("Frozen source changed: " + name)


def freeze_stage(name, caps, extra=(), **settings):
    out, contract = locations()
    directory, protocol_file = out / name, contract / (name + "_protocol.json")
    if directory.exists() or protocol_file.exists(): raise FileExistsError("Do not replay existing stage: " + name)
    core = read_json(contract / "protocol.json")
    if digest(out / "inputs.json") != core["inputs_sha256"]: raise ValueError("Inputs changed")
    import importlib.util
    engine_file = Path(importlib.util.find_spec("forsythia_hep").origin)
    protocol = {"stage": name, "caps": caps, "code_sha256": source_hashes(extra),
                "core_protocol_sha256": digest(contract / "protocol.json"),
                "engine_module": str(engine_file), "engine_module_sha256": digest(engine_file), **settings}
    write_json(protocol_file, protocol)
    directory.mkdir()
    write_json(directory / "started.json", {"protocol_sha256": digest(protocol_file), "pid": os.getpid()})
    return directory, protocol, Ledger(caps)


def complete_stage(directory, protocol, ledger, report):
    check_sources(protocol)
    report = {**report, "counts": dict(ledger.counts)}
    write_json(directory / "report.json", report)
    write_json(directory / "completed.json", {"report_sha256": digest(directory / "report.json"), "counts": dict(ledger.counts)})
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    return report


def prepare():
    """Input-only local groups; no identity, random generator, or model call."""
    import pickle
    from .legal_actions import build_policy_input
    from project.paths import asset_path
    out, contract = locations()
    if out.exists() or contract.exists(): raise FileExistsError("Existing experiment must not be overwritten")
    meta_path = REPO / "datasets/unknown_online_rl_v1/train/samples.jsonl"
    shard = REPO / "datasets/unknown_online_rl_v1/train/input_data/unknown_amp_part000.pkl.gz"
    metadata = [json.loads(line) for line in meta_path.read_text().splitlines() if line]
    metadata.sort(key=lambda r: key(RUN_ID, "source", r["source_id"]))
    chosen = metadata[:96]
    if len(chosen) != 96 or len({r["source_id"] for r in chosen}) != 96: raise ValueError("Need 96 independent origins")
    # Freeze selection rule before any reconstruction; input-only local slicing is
    # not a random amplitude/identity generator and does not inspect outcomes.
    out.mkdir(parents=True); contract.mkdir(parents=True)
    write_json(contract / "preparation_protocol.json", {
        "selection": "sort original sources by sha256(RUN_ID,source,source_id), first32 train,next64 research-validation",
        "local_sizes": [8,16,24,32], "term_selection": "sort original term ordinals by sha256(RUN_ID,terms,source_id,term_tokens); take fixed size",
        "selected_metadata": chosen, "metadata_sha256": digest(meta_path), "shard_sha256": digest(shard),
        "local_constructions_cap": 96, "model_loads": 0, "forwards": 0, "identity_calls": 0, "random_generation": 0})
    with gzip.open(shard, "rb") as f: raw = pickle.load(f)
    records = {}
    for sample in raw:
        if not isinstance(sample, (list, tuple)) or len(sample) != 2: raise ValueError("Target-bearing shard rejected")
        expression, n_point = sample
        records[hashlib.sha256(repr(expression.tokenize()).encode()).hexdigest()] = (expression, n_point)
    rows = []
    for index, meta in enumerate(chosen):
        expression, n = records[meta["sample_id"]]
        if n != 4: raise ValueError("Expected physical four-point context")
        size = (8,16,24,32)[index % 4]
        tokens = expression.tokenize()
        indices = sorted(range(len(tokens)), key=lambda t: key(RUN_ID, "terms", meta["source_id"], tokens[t]))[:size]
        local = 0 * expression
        for t in indices:
            term, coefficient = expression.get_term_at(t)
            local = local + term * coefficient
        local_tokens = local.tokenize()
        if len(local_tokens) != size: raise ValueError("Local group size changed, no replacement")
        build_policy_input([local_tokens], [n], [CALLS], [CALLS], max_terms=32)
        rows.append({"index": index, "split": "train" if index < 32 else "validation", "source_id": meta["source_id"],
                     "original_input_id": meta["sample_id"], "original_term_indices": indices,
                     "tokens": local_tokens, "n_point": n, "terms": size, "id": key(local_tokens)})
    if len({r["id"] for r in rows}) != 96: raise ValueError("Duplicate local inputs; no replacement")
    write_json(out / "inputs.json", rows)
    encoder = asset_path("checkpoints/temp15_final_model_sl_256.pth")
    baseline = asset_path("checkpoints_p1_depth2_full160_160u_seed42/grpo_update_0096.pth")
    protocol = {"id": RUN_ID, "authorization": "user requested <=32 terms, autonomous stages1-4 and moderately more rollouts",
                "max_terms": 32, "max_tokens": 128, "max_actions": 32768, "n_point": 4,
                "training_sources": 32, "validation_sources": 64, "train_seeds": TRAIN_SEEDS, "eval_seeds": EVAL_SEEDS,
                "rollouts_per_update": ROLLOUTS, "visits": VISITS, "calls_per_rollout": CALLS, "candidates_per_round": 4,
                "inputs_sha256": digest(out / "inputs.json"), "preparation_sha256": digest(contract / "preparation_protocol.json"),
                "encoder_sha256": digest(encoder), "baseline_sha256": digest(baseline),
                "scope": "input-only local subexpressions from historical training-side sources, not fresh independent full amplitudes",
                "data_generation_calls": 0, "identity_caps": {"stage1":1353,"stage2":8192,"stage4_training":196608,"stage4_evaluation":65536},
                "total_identity_cap": 271689, "lr":3e-4,"optimizer":"AdamW","grad_clip":1.,"post_update_KL_stop":.1,
                "stopping": "mechanical failure stops; stage2 significant deterioration below -2pp stops; no tuning/retry; no automatic deployment"}
    write_json(contract / "protocol.json", protocol)
    print(json.dumps({"prepared":96,"train":32,"validation":64,"terms":[8,16,24,32],"random_generation":0,"identity_calls":0}), flush=True)


def load_inputs(split):
    out, contract = locations()
    if digest(out / "inputs.json") != read_json(contract / "protocol.json")["inputs_sha256"]: raise ValueError("Input content changed")
    return [r for r in read_json(out / "inputs.json") if r["split"] == split]


def reference_scores(reference, batch):
    """Existing product32 law plus actual encoder mask; one old forward, no new law."""
    import torch
    original = reference._bracket_policy_head
    captured = {}
    def observe(*args, **kwargs):
        captured["mask"] = kwargs["validity_invalid_mask"]
        return original(*args, **kwargs)
    if "_bracket_policy_head" in reference.__dict__: raise ValueError("Unexpected existing method override")
    reference._bracket_policy_head = observe
    try:
        with torch.no_grad(): term, bracket, *_ = reference(batch.token_ids)
    finally:
        delattr(reference, "_bracket_policy_head")
    rows, S = batch.actions, batch.token_ids.shape[-1]
    t = rows.action_indices[:,0]
    a, b = rows.token_positions.unbind(1)
    flat = a * (2*S-a-3)//2 + b-1
    values = (term[0,t] * bracket[0,t,flat]).double()
    allowed = ~captured["mask"][0,t,flat].bool() & rows.query_allowed
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()): raise ValueError("Invalid legacy scores")
    return values, allowed
