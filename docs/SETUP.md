# 环境和复现实验

使用Python 3.12。安装PyTorch 2.7.1和requirements.txt中的依赖。符号引擎为forsythia-hep 0.1.0，使用与操作系统和Python ABI匹配的wheel：

```powershell
python -m pip install -r requirements.txt
python -m pip install /path/to/forsythia_hep-0.1.0-compatible.whl
```

配套模型位于`artifacts/models/`。运行固定1024条输入的评估：

```powershell
python -B scripts/evaluate_fourpoint.py
```

结果写入带时间戳的`artifacts/evaluations/`，包含逐步项数、返回表达式与分量纲统计。命令行可设置`--policy`、`--encoder`、`--topk`、`--steps`、`--seed`与`--output`。

接口合成测试：`python -B -m unittest tests.test_inference -q`。

生成新的四点数据：

```powershell
python -B scripts/generate_fourpoint.py --count 1024 --dimension-max 10 --min-terms 21 --max-terms 40 --output datasets/my_fourpoint
```

原始表达式保存在`origins.json`，扰乱后的输入保存在`inputs.json`。`--scramble-steps`控制每个候选的扰乱上限，`--candidate-cap`控制候选数量。调用`data/generation.py`中的原始构型生成方法，统一表达式中各项的量纲和螺旋度。

预训练入口为`scripts/pretrain.py::train_model`，配置示例位于`configs/pretrain.json`。当前PPO的策略、训练环境、GAE与更新函数分别位于`restricted_policy.py`、`restricted_environment.py`、`packed_ppo.py`和`entropy20_policy.py`。展示模型的参数与最后训练阶段记录在`configs/policy_096.json`，张量计算见`docs/ARCHITECTURE.md`。
