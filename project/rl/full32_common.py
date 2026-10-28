"""New complete-amplitude experiment; old subset experiment remains immutable."""
from pathlib import Path
import importlib.util
import os
from .packed32_common import (REPO, TRAIN_SEEDS, EVAL_SEEDS, ROLLOUTS, CALLS, VISITS,
    digest, read_json, write_json, append, key, tensor_digest, paired, advantages,
    Ledger, expression_from_tokens, reference_scores, source_hashes, check_sources)

EXPERIMENT="packed_full32_rl_20261023_v1"
DATASET="packed_full32_20261023_v1"


def locations():
    from project.paths import run_path, contract_path
    return run_path(EXPERIMENT),contract_path(EXPERIMENT)


def seed_for(*values):return int(key(EXPERIMENT,*values)[:15],16)


def load_inputs(split):
    import hashlib
    folder=REPO/"datasets"/DATASET
    manifest=read_json(folder/"manifest.json")
    if manifest["status"]!="completed":raise ValueError("Incomplete generation")
    path=folder/(split+".json")
    if digest(path)!=manifest["files_sha256"][path.name]:raise ValueError("Input file changed")
    rows=read_json(path)
    for r in rows:
        if set(r)!={"index","split","id","source_id","n_point","terms","tokens"}:raise ValueError("Target/path-bearing record rejected")
        if r["id"]!=hashlib.sha256(repr(r["tokens"]).encode()).hexdigest() or r["terms"]!=len(r["tokens"]) or not 1<=r["terms"]<=32 or r["n_point"]!=4 or r["split"]!=split:
            raise ValueError("Input metadata mismatch")
    return rows


def freeze(name,caps,extra=(),**settings):
    from project.paths import asset_path
    out,contract=locations();directory=out/name;file=contract/(name+"_protocol.json")
    if directory.exists() or file.exists():raise FileExistsError("Do not replay or overwrite "+name)
    data=REPO/"datasets"/DATASET;manifest=read_json(data/"manifest.json")
    engine=Path(importlib.util.find_spec("forsythia_hep").origin)
    code=source_hashes((*extra,"project/rl/full32_common.py","scripts/run_full32.py"))
    protocol={"stage":name,"caps":caps,"code_sha256":code,"dataset":DATASET,"dataset_manifest_sha256":digest(data/"manifest.json"),
        "input_sha256":manifest["files_sha256"],"device":"cuda","max_terms":32,"max_tokens":128,"max_actions":32768,
        "encoder_sha256":digest(asset_path("checkpoints/temp15_final_model_sl_256.pth")),
        "baseline_sha256":digest(asset_path("checkpoints_p1_depth2_full160_160u_seed42/grpo_update_0096.pth")),
        "engine_sha256":{str(p):digest(p) for p in [engine,*engine.parent.glob("*.pyd")]},
        "supersedes":"user rejected arbitrary subsets; old run preserved; old two engine attempts charged separately",**settings}
    out.mkdir(parents=True,exist_ok=True);contract.mkdir(parents=True,exist_ok=True)
    write_json(file,protocol);directory.mkdir()
    write_json(directory/"started.json",{"protocol_sha256":digest(file),"pid":os.getpid()})
    return directory,protocol,Ledger(caps)


def require_stage(name):
    out,contract=locations();folder=out/name
    done=read_json(folder/"completed.json")
    if digest(folder/"report.json")!=done["report_sha256"]:raise ValueError("Prior stage report changed")
    report=read_json(folder/"report.json")
    if not report["passed"]:raise RuntimeError("Prerequisite stage did not pass: "+name)
    check_sources(read_json(contract/(name+"_protocol.json")))
    return report


def complete(directory,protocol,ledger,report):
    check_sources(protocol)
    report={**report,"counts":dict(ledger.counts)}
    write_json(directory/"report.json",report)
    write_json(directory/"completed.json",{"report_sha256":digest(directory/"report.json"),"counts":dict(ledger.counts)})
    import json
    print(json.dumps(report,indent=2),flush=True)
    return report


def failure(directory,ledger,exc):
    write_json(directory/"failed.json",{"error":repr(exc),"counts":dict(ledger.counts),"automatic_retry":False})


def configure_torch():
    import torch
    if not torch.cuda.is_available():raise RuntimeError("Frozen CUDA device unavailable")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return torch
