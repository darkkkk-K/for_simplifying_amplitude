"""Lazy factory: default-off does not import Torch or load any checkpoints."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import typing

REPO = Path(__file__).resolve().parents[2]
BASELINE_SHA = "1b381dd47f0bc665d1ff298781d07bada79e8c298920bd41a656451996517a46"


@dataclass
class LoadedCandidatePolicy:
    policy: object
    reference: object
    metadata: dict


def policy_class(encoder):
    """Load the existing class unchanged, excluding module-level checkpoint/data I/O."""
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from project.pretraining.encoder import load_definitions
    vocab = json.loads((REPO / "data/tokenizor.json").read_text(encoding="utf-8"))
    namespace = {"__name__": "readonly_policy_definition", "torch": torch, "nn": nn, "F": F,
                 "np": np, "math": math, "TransformerEncoder": nn.TransformerEncoder,
                 "TransformerEncoderLayer": nn.TransformerEncoderLayer,
                 "model_clustring": encoder, "model1_config": {"emb_dim": encoder.emb_dim},
                 "token_id_of_bracket": list(range(16, 46)), "tokenizor": vocab,
                 "reverse_tokenizor": {v: k for k, v in vocab.items() if 16 <= v <= 45},
                 **{name: getattr(typing, name) for name in ("List", "Tuple", "Optional", "Dict", "Any", "Callable", "Literal")}}
    load_definitions("models/rl_policy.py", ["extract_upper_triangular_no_diag", "policy_machine2"], namespace)
    return namespace["policy_machine2"]


def load_candidate_policy(*, mode="off", device="cpu", config=None, seed=103):
    if mode == "off":
        return None
    if mode != "packed":
        raise ValueError("mode must be off or packed")
    import torch
    from project.paths import asset_path
    from project.pretraining.encoder import read_checkpoint, construct_encoder
    from project.pretraining.io import digest
    from .packed_policy import CandidateConfig, attach_candidate_policy
    encoder_path = asset_path("checkpoints/temp15_final_model_sl_256.pth")
    checkpoint_path = asset_path("checkpoints_p1_depth2_full160_160u_seed42/grpo_update_0096.pth")
    if digest(checkpoint_path) != BASELINE_SHA:
        raise ValueError("Reference checkpoint changed; do not silently use a different backbone")
    encoder_state, encoder_config = read_checkpoint(encoder_path, num_heads=8)
    if (encoder_config["emb_dim"], encoder_config["numenc1"], encoder_config["fnn_dim"], encoder_config["vocab_size"]) != (768, 3, 3072, 46):
        raise ValueError("Selected encoder is not the agreed 768-dimensional temp15 model")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = saved.get("model_state_dict", saved)
    if not isinstance(state, dict):
        raise ValueError("Expected dictionary reference checkpoint")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    embedded = {key.removeprefix("embdding."): value for key, value in state.items() if key.startswith("embdding.")}
    if set(embedded) != set(encoder_state) or any(not torch.equal(embedded[key], value) for key, value in encoder_state.items()):
        raise ValueError("The reference policy's embedded encoder differs from the user-selected encoder")
    # Preserve caller's global RNG; only the adapter initialization uses this seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = construct_encoder(encoder_state, encoder_config, "cpu")
        cls = policy_class(encoder)
        reference = cls(vocab_size=46, emb_dim=512, numhead=8, numenc1=1, fnn_dim=1024,
                        head_dim=256, dropout=0., use_delta_head=False, action_head_mode="factorized")
        reference.load_state_dict(state, strict=True)
        reference.requires_grad_(False).eval().to(device)
        candidate = attach_candidate_policy(reference, mode="packed", config=config or CandidateConfig())
    metadata = {"encoder": str(encoder_path), "encoder_sha256": digest(encoder_path),
                "reference": str(checkpoint_path), "reference_sha256": BASELINE_SHA,
                "encoder_config": encoder_config, "checkpoint_loads": 2,
                "trainable_parameters": sum(p.numel() for p in candidate.parameters() if p.requires_grad),
                "seed": seed}
    if metadata["trainable_parameters"] != 64545:
        raise ValueError("New parameter count differs from the design")
    return LoadedCandidatePolicy(candidate, reference, metadata)
