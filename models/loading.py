import re
from .encoder import ClusteringMachine1_RoPE
def unpack_checkpoint(payload):
    import torch
    if not isinstance(payload, dict):
        raise ValueError("Expected state_dict or a checkpoint dictionary; whole pickled model objects are not accepted")
    metadata = payload.get("config", {})
    if not isinstance(metadata, dict):
        metadata = {}
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Checkpoint does not contain a tensor state_dict")
    keys = list(state)
    if all(k.startswith("module.") for k in keys):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    if "embdding.weight" not in state:
        raise ValueError("Not a standalone pretrained encoder (missing embdding.weight); do not pass an RL policy checkpoint")
    return state, metadata


def infer_config(state, metadata=None, *, num_heads=None, rope_mode=None):
    metadata = metadata or {}
    embedding = state["embdding.weight"]
    if embedding.ndim != 2:
        raise ValueError("Malformed embedding weight")
    vocab_size, width = map(int, embedding.shape)
    layer_ids = sorted({int(m.group(1)) for key in state
                        if (m := re.match(r"transformer\.layers\.(\d+)\.", key))})
    if not layer_ids or layer_ids != list(range(len(layer_ids))):
        raise ValueError("Transformer layer indices must be contiguous")
    if "transformer.layers.0.self_attn.q_proj.weight" not in state:
        raise ValueError(f"Embedding width={width}, but this is not the supported ClusteringMachine1_RoPE architecture")
    ff = int(state["transformer.layers.0.linear1.weight"].shape[0])
    for i in layer_ids:
        if tuple(state[f"transformer.layers.{i}.linear1.weight"].shape) != (ff, width):
            raise ValueError("Inconsistent feed-forward shapes")
    for key, inferred in (("emb_dim", width), ("numenc", len(layer_ids)), ("fnn_dim", ff), ("vocab_size", vocab_size)):
        if key in metadata and int(metadata[key]) != inferred:
            raise ValueError(f"Checkpoint metadata {key} contradicts tensor shapes")
    theta = state.get("transformer.layers.0.rope.log_theta")
    inferred_mode = inferred_heads = None
    if theta is not None:
        if theta.ndim == 1 and theta.numel() == width // 2:
            inferred_mode = "shared_learnable"
        elif theta.ndim == 2 and theta.numel() == width // 2:
            inferred_mode, inferred_heads = "unshared_learnable", int(theta.shape[0])
        else:
            raise ValueError("Unsupported learnable RoPE parameter shape")
    config_heads = metadata.get("numhead", metadata.get("num_heads"))
    head_candidates = [int(v) for v in (num_heads, config_heads, inferred_heads) if v is not None]
    if len(set(head_candidates)) > 1:
        raise ValueError("--num-heads conflicts with checkpoint metadata/shape")
    heads = head_candidates[0] if head_candidates else None
    if heads is not None and (heads <= 0 or width % heads or (width // heads) % 2):
        raise ValueError("Attention head count must divide embedding width with even head_dim")
    mode_candidates = [v for v in (rope_mode, metadata.get("RoPE_mode"), inferred_mode) if v is not None]
    if len(set(mode_candidates)) > 1:
        raise ValueError("--rope-mode conflicts with checkpoint metadata/shape")
    mode = mode_candidates[0] if mode_candidates else None
    if mode not in (None, "shared_learnable", "unshared_learnable", "shared_fixed", "unshared_fixed"):
        raise ValueError("Unsupported RoPE mode")
    return {"model_type": "ClusteringMachine1_RoPE", "vocab_size": vocab_size, "emb_dim": width,
            "numenc1": len(layer_ids), "fnn_dim": ff, "numhead": heads, "RoPE_mode": mode,
            "numhead_source": "explicit --num-heads" if num_heads is not None else
                              "checkpoint metadata/shape" if heads is not None else "unknown; specify --num-heads",
            "width_source": "embdding.weight.shape[1]"}

"""Load the released encoder and expression-only actor–critic."""
from pathlib import Path
import torch
from .encoder import ClusteringMachine1_RoPE


def read_checkpoint(path, *, num_heads=8, rope_mode=None):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    state, metadata = unpack_checkpoint(payload)
    return state, infer_config(state, metadata, num_heads=num_heads, rope_mode=rope_mode)


def construct_encoder(state, config, device='cpu'):
    model = ClusteringMachine1_RoPE(**{k: config[k] for k in
        ('vocab_size', 'emb_dim', 'numhead', 'numenc1', 'fnn_dim', 'RoPE_mode')})
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_policy(encoder_path, policy_path=None, *, device='cpu', top_k=8):
    from .policy import RestrictedActorCritic
    state, config = read_checkpoint(encoder_path)
    encoder = construct_encoder(state, config, device)
    payload = torch.load(policy_path, map_location='cpu', weights_only=True) if policy_path else None
    model = RestrictedActorCritic(encoder, enabled=True, top_k=payload['top_k'] if payload else top_k).to(device)
    if payload:
        model.load_checkpoint_payload(payload)
    return model
