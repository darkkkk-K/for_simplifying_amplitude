"""Use the current expression-only PPO policy; return the best visited SpExpr.

Run this file for one on-the-fly demonstration, or import simplify_expression.
Importing it does not load weights, generate amplitudes, or run inference.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import random
import sys
from typing import TYPE_CHECKING
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.pretraining.encoder import read_checkpoint, construct_encoder
from project.reducibility.features import structures, validate
from project.reducibility.generation import amplitude_config
from project.rl.early_fusion_policy import sample_one
from project.rl.legal_actions import UnsupportedExpression
from project.rl.restricted_inference import select_argmax
from project.rl.restricted_policy import RestrictedActorCritic, SCHEMA, build_observation

if TYPE_CHECKING:
    from forsythia_hep import SpExpr

# ---------------- 用户可以直接修改的示例参数 ----------------
DEMO_SEED = 103
DEMO_N_POINT = None            # None：在4/5/6点中等概率随机选择；整数：固定点数。
DEMO_DIMENSION = None          # None：在下方闭区间随机选择；整数：固定量纲。
DEMO_DIMENSION_MIN = 2
DEMO_DIMENSION_MAX = 12        # 默认示例与后续五/六点低量纲课程范围一致。
DEMO_INITIAL_TERMS = 5           # 生成1–5个单项式后相加；合并同类项后可能更少。
DEMO_SCRAMBLE_STEPS = 16        # 扰乱尝试次数，不保证每次都改变表达式。
DEMO_TOPK = 1                  # 1 = argmax；>1 = top-k内按softmax概率采样。
DEMO_BUDGET_STEPS = 36
# 五点联合60轮后，再以5.2:4.3:4.2=24:4:4追加36轮的终点；旧模型可用--policy指定。
DEFAULT_POLICY = ROOT / "artifacts/backups/course52_resume36_20261027_v1/course52_resume_round036/round_036.pt"
DEFAULT_ENCODER = ROOT / "artifacts/models/pretrained/temp15_final_model_sl_256.pth"
MAX_TERMS = 128
MAX_TOKENS_PER_TERM = 128


def _engine():
    import forsythia_hep
    return forsythia_hep


def _check_arguments(n_point, topk, budget_steps, seed):
    if type(n_point) is not int or n_point not in (4, 5, 6):
        raise ValueError("n_point必须为4、5或6；接口支持不代表对应构型的效果已验证。")
    if type(topk) is not int or topk < 1:
        raise ValueError("topk必须是正整数；topk=1表示argmax。")
    if type(budget_steps) is not int or budget_steps < 0:
        raise ValueError("budget_steps必须是非负整数。")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError("seed必须为None或[0, 2**63)内的整数。")


def _over_capacity(tokens):
    return len(tokens) > MAX_TERMS or max(map(len, tokens), default=0) > MAX_TOKENS_PER_TERM


class PolicySimplifier:
    """Load once, simplify many SpExpr inputs. No classifier, critic or rewards.

    model.top_k keeps its checkpoint/training meaning. The topk argument to
    simplify controls inference sampling explicitly and never mutates it.
    """

    def __init__(self, policy_path=DEFAULT_POLICY, encoder_path=DEFAULT_ENCODER, *, device=None):
        self.policy_path = Path(policy_path).expanduser().resolve()
        self.encoder_path = Path(encoder_path).expanduser().resolve()
        for path in (self.policy_path, self.encoder_path):
            if not path.is_file():
                raise FileNotFoundError(f"模型文件不存在：{path}。请显式指定路径，不自动改用其他模型。")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(self.policy_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("observation_schema") != SCHEMA:
            raise ValueError("仅支持expression_best_v1策略检查点，不隐式转换旧架构。")
        training_topk = payload.get("top_k")
        if type(training_topk) is not int or training_topk < 1:
            raise ValueError("策略检查点缺少有效的训练top_k。")
        state, config = read_checkpoint(self.encoder_path, num_heads=8)
        if config["emb_dim"] != 768:
            raise ValueError("该策略要求配套的768维预训练encoder。")
        encoder = construct_encoder(state, config, "cpu")
        self.model = RestrictedActorCritic(encoder, enabled=True, top_k=training_topk)
        self.model.load_checkpoint_payload(payload)
        self.model.to(self.device).eval().requires_grad_(False)
        self.last_info = None

    @torch.inference_mode()
    def simplify(self, expression: SpExpr, n_point: int, topk: int = 1,
                 budget_steps: int = 36, *, seed: int | None = None, verbose=False) -> SpExpr:
        """Return a shortest visited expression, including the normalized input.

        topk=1 selects the first argmax; otherwise sample one action from the
        min(topk, legal_action_count) largest logits using the PPO sampler.
        Temporary expansion is accepted. We keep searching from the successor,
        never reset to best_expression, and keep the earliest state on ties.
        Budget/step/reward history are never inputs to the network.
        """
        _check_arguments(n_point, topk, budget_steps, seed)
        api = _engine()
        if not isinstance(expression, api.SpExpr):
            raise TypeError("expression必须是forsythia_hep.SpExpr，不是字符串或tokens。")
        self.last_info = None
        if n_point == 6:
            warnings.warn("六点可调用，但默认策略尚未经过六点课程训练或效果验证；自定义权重请以其训练记录为准。", stacklevel=2)
        current = expression.normal_order().expand()
        tokens = current.tokenize()
        if _over_capacity(tokens):
            raise UnsupportedExpression("输入超过128项或每项128 tokens；不会静默截断。")
        _, _, dimensions, weights = structures(tokens, n_point)
        dimension = dimensions[0] if dimensions else 0
        helicities = [-w for w in weights[0]] if weights else [0] * n_point
        best_expression, best_terms = current, len(tokens)
        generator = torch.Generator(device="cpu")
        generator.seed() if seed is None else generator.manual_seed(seed)
        self.model.eval()
        history = []
        reason = "zero_expression" if not tokens else "budget"
        forwards = draws = 0
        initial_terms = len(tokens)
        for step in range(budget_steps):
            if not tokens:
                reason = "zero_expression"
                break
            observation = build_observation(tokens, n_point, best_terms)
            if not len(observation.actions):
                reason = "no_legal_actions"
                break
            state = self.model.prepare(observation)
            logits, mask = self.model.policy_logits([state])
            forwards += 1
            if topk == 1:
                index, _ = select_argmax(logits, mask)
            else:
                index, _, _ = sample_one(logits, mask, topk, generator)
                draws += 1
            term, first, second = state.actions[index].tolist()
            action = (term, (first, second))
            child = api.apply_identity_at(current, action, n_point)[0]
            if current.tokenize() != tokens:
                raise RuntimeError("符号引擎修改了父表达式；无法安全保存最好结果。")
            next_tokens = child.tokenize()
            validate(next_tokens, n_point, dimension, helicities)
            after = len(next_tokens)
            if after < best_terms:
                best_expression, best_terms = child, after
            history.append({"step": step + 1, "before": len(tokens), "after": after,
                            "best_terms": best_terms, "action": action})
            if verbose:
                print(f"step {step + 1:02d}: {len(tokens)} -> {after}; best={best_terms}")
            current, tokens = child, next_tokens
            if not tokens:
                reason = "zero_expression"
                break
            if _over_capacity(tokens):
                reason = "capacity"
                break
        self.last_info = {"initial_terms": initial_terms, "final_terms": len(tokens),
                          "best_terms": best_terms, "steps": len(history), "reason": reason,
                          "encoder_forwards": forwards, "policy_forwards": forwards,
                          "identity_calls": len(history), "sampling_draws": draws,
                          "value_forwards": 0, "classifier_forwards": 0,
                          "history": history}
        return best_expression


@lru_cache(maxsize=1)
def _default_simplifier():
    return PolicySimplifier()


def simplify_expression(expression: SpExpr, n_point: int, topk: int = 1,
                        budget_steps: int = 36, *, seed: int | None = None,
                        simplifier: PolicySimplifier | None = None) -> SpExpr:
    """Four-argument convenience API; lazily reuse the default loaded model.

    Pass a PolicySimplifier via the optional keyword to select weights/device.
    Its last_info provides actual steps, terminal length and per-step lengths.
    """
    _check_arguments(n_point, topk, budget_steps, seed)
    runner = simplifier if simplifier is not None else _default_simplifier()
    return runner.simplify(expression, n_point, topk, budget_steps, seed=seed)


def _check_demo_configuration(n_point, dimension, dimension_min, dimension_max):
    if n_point is not None and (type(n_point) is not int or n_point not in (4, 5, 6)):
        raise ValueError("示例n_point必须为None或4/5/6。")
    if dimension is not None and (type(dimension) is not int or dimension < 2):
        raise ValueError("示例dimension必须为None或不小于2的整数。")
    if (type(dimension_min) is not int or type(dimension_max) is not int
            or not 2 <= dimension_min <= dimension_max):
        raise ValueError("量纲随机区间要求2 <= dimension_min <= dimension_max。")


def sample_demo_configuration(rng, nrng, *, n_point=None, dimension=None,
                              dimension_min=DEMO_DIMENSION_MIN, dimension_max=DEMO_DIMENSION_MAX):
    """Choose one shared physical configuration; no engine/model calls.

    Fixing point and dimension consumes no extra RNG draws compared with the
    original fixed-point demo, preserving its helicity/coefficient RNG sequence.
    """
    _check_demo_configuration(n_point, dimension, dimension_min, dimension_max)
    point = rng.choice((4, 5, 6)) if n_point is None else n_point
    dim = rng.randint(dimension_min, dimension_max) if dimension is None else dimension
    helicities = amplitude_config(point, dim, rng, nrng)
    return {"n_point": point, "dimension": dim, "helicities_twice": helicities}


def make_demo_expression(n_point, dimension, initial_terms, scramble_steps, seed, *,
                         dimension_min=DEMO_DIMENSION_MIN, dimension_max=DEMO_DIMENSION_MAX):
    """One bounded example, no rejection sampling, no reverse-trajectory target."""
    _check_arguments(4 if n_point is None else n_point, 1, scramble_steps, seed)
    if type(initial_terms) is not int or not 1 <= initial_terms <= 5:
        raise ValueError("示例初态由1–5个单项式组成。")
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    configuration = sample_demo_configuration(rng, nrng, n_point=n_point, dimension=dimension,
                                             dimension_min=dimension_min, dimension_max=dimension_max)
    n_point, dimension, helicities = (configuration[key] for key in
                                    ("n_point", "dimension", "helicities_twice"))
    api = _engine()
    monomials = []
    for _ in range(initial_terms):
        mono = api.random_monomial(helicities, dimension)
        validate(mono.tokenize(), n_point, dimension, helicities)
        monomials.append(rng.randint(1, 3) * mono)
    original = sum(monomials).normal_order().expand()
    original_tokens = original.tokenize()
    if not 1 <= len(original_tokens) <= initial_terms:
        raise RuntimeError("本次构型未生成非零短表达式；不自动重新生成，请手动检查参数。")
    validate(original_tokens, n_point, dimension, helicities)
    scrambled, changed = original, 0
    for _ in range(scramble_steps):
        before = scrambled.tokenize()
        if rng.random() < .5:
            scrambled = api.random_apply_schouten(scrambled, 1)[1]
        else:
            scrambled = api.random_apply_momentum_conservation(scrambled, n_point, 1)[1]
        scrambled = scrambled.normal_order().expand()
        tokens = scrambled.tokenize()
        validate(tokens, n_point, dimension, helicities)
        changed += int(tokens != before)
        if _over_capacity(tokens):
            raise UnsupportedExpression("扰乱超出模型容量，停止示例；不截断或自动重试。")
    return original, scrambled, {"n_point": n_point, "dimension": dimension,
                                "helicities_twice": helicities, "monomial_calls": initial_terms,
                                "scramble_calls": scramble_steps, "changed_scrambles": changed}


def main(argv=None):
    # PowerShell/log wrappers decode UTF-8; avoid locale-dependent mojibake.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--device", default=None, help="默认自动选择cuda或cpu")
    parser.add_argument("--seed", type=int, default=DEMO_SEED)
    parser.add_argument("--n-point", type=int, choices=(4, 5, 6), default=DEMO_N_POINT,
                        help="不指定时随机选择4/5/6点；指定时固定")
    parser.add_argument("--dimension", type=int, default=DEMO_DIMENSION,
                        help="不指定时在dimension-min/max区间随机选择；指定时固定")
    parser.add_argument("--dimension-min", type=int, default=DEMO_DIMENSION_MIN)
    parser.add_argument("--dimension-max", type=int, default=DEMO_DIMENSION_MAX)
    parser.add_argument("--initial-terms", type=int, default=DEMO_INITIAL_TERMS)
    parser.add_argument("--scramble-steps", type=int, default=DEMO_SCRAMBLE_STEPS)
    parser.add_argument("--topk", type=int, default=DEMO_TOPK)
    parser.add_argument("--steps", type=int, default=DEMO_BUDGET_STEPS)
    args = parser.parse_args(argv)
    _check_arguments(4 if args.n_point is None else args.n_point, args.topk, args.steps, args.seed)
    _check_demo_configuration(args.n_point, args.dimension, args.dimension_min, args.dimension_max)
    if not 1 <= args.initial_terms <= 5 or args.scramble_steps < 0:
        parser.error("要求initial-terms为1–5、scramble-steps>=0。")
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    print("注意：seed固定Python/NumPy/PyTorch随机数；当前符号引擎没有seed接口，随机生成/扰乱不保证完全复现。")
    print(f"本次上限：{args.initial_terms}次单项式生成、{args.scramble_steps}次随机扰乱、{args.steps}次策略动作；无训练。")
    runner = PolicySimplifier(args.policy, args.encoder, device=args.device)
    print(f"策略：{runner.policy_path}\n设备：{runner.device}; topk={args.topk}")
    original, scrambled, generation = make_demo_expression(
        args.n_point, args.dimension, args.initial_terms, args.scramble_steps, args.seed,
        dimension_min=args.dimension_min, dimension_max=args.dimension_max)
    print(f"构型：{generation}")
    if args.scramble_steps and not generation["changed_scrambles"]:
        print("提示：本例的扰乱尝试均未改变表达式；不自动换样本，可手动增大dimension或更换seed。")
    print(f"原始表达式（{len(original.tokenize())}项）：{original}")
    print(f"扰乱后表达式（{len(scrambled.tokenize())}项）：{scrambled}")
    # Only the scrambled expression enters inference. No original target is used.
    best = runner.simplify(scrambled, generation["n_point"], args.topk, args.steps, seed=args.seed, verbose=True)
    info = runner.last_info
    reduction = (info["initial_terms"] - info["best_terms"]) / info["initial_terms"] if info["initial_terms"] else 0.
    print(f"返回最短表达式（{info['best_terms']}项）：{best}")
    print(f"减项率={reduction:.2%}; 实际动作={info['steps']}; 末状态项数={info['final_terms']}; 结束原因={info['reason']}")
    print(f"调用计数：encoder={info['encoder_forwards']}, policy={info['policy_forwards']}, "
          f"identity={info['identity_calls']}, samples={info['sampling_draws']}, critic=0, classifier=0")
    return best


if __name__ == "__main__":
    main()
