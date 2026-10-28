"""Generate complete <=32-term inputs from <=4-term sources; no inverse labels."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random

from project.rl.packed32_common import REPO, digest, write_json, append, Ledger

DATA_ID = "packed_full32_20261023_v1"
ROUNDS = (4, 8, 12, 16)
CAPS = {"attempts":960, "monomial_calls":3840, "random_identity_calls":15360,
        "model_loads":0, "forwards":0, "optimizer":0}


def token_id(tokens): return hashlib.sha256(repr(tokens).encode("utf-8")).hexdigest()


def accept_lengths(source_terms, input_terms):
    return 1 <= source_terms <= 4 and source_terms < input_terms <= 32


def public_record(index, split, tokens, source_id):
    # Deliberately exclude clean tokens, clean length, reverse actions and path.
    return {"index":index, "split":split, "id":token_id(tokens), "source_id":source_id,
            "n_point":4, "terms":len(tokens), "tokens":tokens}


def run():
    from project.paths import contract_path, run_path
    from project.rl.legal_actions import build_policy_input
    dataset, out, contract = REPO / "datasets" / DATA_ID, run_path(DATA_ID), contract_path(DATA_ID)
    if dataset.exists() or out.exists() or contract.exists(): raise FileExistsError("Existing generation must not be overwritten")
    metadata_paths=sorted((REPO/"datasets").glob("**/samples.jsonl"))
    excluded_sources, excluded_inputs=set(),set()
    for path in metadata_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:continue
            row=json.loads(line)
            if row.get("source_id"):excluded_sources.add(row["source_id"])
            if row.get("sample_id"):excluded_inputs.add(row["sample_id"])
    engine=Path(importlib.util.find_spec("forsythia_hep").origin)
    files=[Path(__file__).resolve(),REPO/"scripts/generate_full32.py",REPO/"project/rl/packed32_common.py",REPO/"project/rl/legal_actions.py"]
    protocol={"id":DATA_ID,"purpose":"complete scrambled inputs, never selected subsets",
        "train":32,"validation":64,"n_point":4,"source_monomials":[1,4],"acceptance":"1<=source_terms<=4 and source_terms<input_terms<=32",
        "scramble_calls_per_attempt":ROUNDS,"schedule":"accepted index modulo four, same cycle in each split",
        "helicity_options":[[-1,-1,1,1],[0,0,0,0]],"mass_dimensions":[6,8,10],
        "coefficient_options":[-3,-2,-1,1,2,3],"caps":CAPS,"max_intermediate_terms":128,"max_token_length":128,
        "python_seed":2026102307,"engine_rng":"native engine random seed is not exposed; saved exact inputs are authoritative",
        "code_sha256":{p.relative_to(REPO).as_posix():digest(p) for p in files},
        "excluded_metadata_sha256":{p.relative_to(REPO).as_posix():digest(p) for p in metadata_paths},
        "engine_sha256":{str(p):digest(p) for p in [engine,*engine.parent.glob("*.pyd")]},
        "split":"one accepted input per distinct original source; train and validation disjoint before model use",
        "target_use":"source only constructs and filters generation; no clean target, target count, or inverse action in policy inputs",
        "storage_cap_bytes":256*1024**2,"automatic_retry_after_failure":False}
    dataset.mkdir();out.mkdir(parents=True);contract.mkdir(parents=True)
    write_json(contract/"protocol.json",protocol)
    write_json(out/"started.json",{"protocol_sha256":digest(contract/"protocol.json")})
    ledger=Ledger(CAPS);rng=random.Random(protocol["python_seed"]);accepted=[];attempts=[]
    try:
        import forsythia_hep as api
        while len(accepted)<96 and ledger.counts["attempts"]<CAPS["attempts"]:
            ledger.hit("attempts");serial=ledger.counts["attempts"]
            requested=rng.randint(1,4);helicities=rng.choice(protocol["helicity_options"]);dimension=rng.choice(protocol["mass_dimensions"])
            steps=ROUNDS[len(accepted)%4];split="train" if len(accepted)<32 else "validation"
            source=0*api.abk(1,2)
            for monomial in range(requested):
                ledger.hit("monomial_calls")
                append(out/"calls.jsonl",{"event":"start","kind":"monomial","serial":ledger.counts["monomial_calls"],"attempt":serial,"helicities":helicities,"dimension":dimension})
                term=api.random_monomial(helicities,dimension)
                source=source+rng.choice(protocol["coefficient_options"])*term
                append(out/"calls.jsonl",{"event":"end","kind":"monomial","serial":ledger.counts["monomial_calls"],"attempt":serial,"term_id":token_id(term.tokenize())})
            source_tokens=source.tokenize();source_terms=len(source_tokens);sid=token_id(source_tokens)
            reason=None
            if not 1<=source_terms<=4:reason="invalid_source_length"
            elif sid in excluded_sources:reason="duplicate_source"
            current=source;executed=0
            if reason is None:
                for step in range(steps):
                    kind=rng.choice(("schouten","momentum"));before=token_id(current.tokenize())
                    ledger.hit("random_identity_calls");call=ledger.counts["random_identity_calls"]
                    append(out/"calls.jsonl",{"event":"start","kind":kind,"serial":call,"attempt":serial,"step":step,"parent":before})
                    try:
                        result=api.random_apply_schouten(current,1) if kind=="schouten" else api.random_apply_momentum_conservation(current,4,1)
                        current=result[1];error=None
                    except RuntimeError as exc:
                        if str(exc)!="no nontrivial identity can be applied":raise
                        error=str(exc)  # A failed random attempt is charged, not replaced.
                    executed+=1;tokens=current.tokenize()
                    append(out/"calls.jsonl",{"event":"end","kind":kind,"serial":call,"attempt":serial,"child":token_id(tokens),"terms":len(tokens),"error":error})
                    if len(tokens)>128 or max(map(len,tokens),default=0)>128:
                        reason="intermediate_capacity";break
            tokens=current.tokenize();iid=token_id(tokens)
            if reason is None and not accept_lengths(source_terms,len(tokens)):reason="length_or_no_growth"
            if reason is None and iid in excluded_inputs:reason="duplicate_input"
            if reason is None:
                build_policy_input([tokens],[4],[64],[64],max_terms=32)
                accepted.append(public_record(len(accepted),split,tokens,sid))
                excluded_sources.add(sid);excluded_inputs.add(iid)
            audit={"attempt":serial,"accepted":reason is None,"reason":reason,"split":split,"source_id":sid,
                   "source_terms":source_terms,"input_id":iid,"input_terms":len(tokens),"random_calls":executed,"planned_calls":steps,
                   "helicities":helicities,"dimension":dimension}
            append(out/"attempts.jsonl",audit);attempts.append(audit)
            if reason is None and len(accepted)%16==0:print(f"generated {len(accepted)}/96; attempts={serial}; random_identity_calls={ledger.counts['random_identity_calls']}",flush=True)
            if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>protocol["storage_cap_bytes"]:raise RuntimeError("Generation storage cap")
        if len(accepted)!=96:raise RuntimeError("Insufficient accepted inputs within frozen generation cap")
        for split in ("train","validation"):write_json(dataset/(split+".json"),[r for r in accepted if r["split"]==split])
        if any(digest(REPO/p)!=sha for p,sha in protocol["code_sha256"].items()):raise ValueError("Generation source changed")
        report={"status":"completed","inputs":96,"train":32,"validation":64,"counts":dict(ledger.counts),
                "input_terms":{"min":min(r["terms"] for r in accepted),"max":max(r["terms"] for r in accepted),"mean":sum(r["terms"] for r in accepted)/96},
                "length_histogram":dict(Counter(r["terms"] for r in accepted)),"rejections":dict(Counter(a["reason"] for a in attempts if not a["accepted"])),
                "source_overlap":0,"source_count":len({r["source_id"] for r in accepted}),"input_only":True,
                "files_sha256":{split+".json":digest(dataset/(split+".json")) for split in ("train","validation")}}
        write_json(dataset/"manifest.json",report);write_json(out/"completed.json",report);write_json(contract/"summary.json",report)
        print(json.dumps(report,indent=2),flush=True)
        return report
    except BaseException as exc:
        write_json(out/"failed.json",{"error":repr(exc),"counts":dict(ledger.counts),"accepted":len(accepted),"automatic_retry":False})
        raise
