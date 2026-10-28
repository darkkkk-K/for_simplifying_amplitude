# 散射振幅的神经符号化简

**768维Transformer代数表示＋PPO强化学习＋符号运算。**

A neural-symbolic approach to scattering amplitude simplification: a pretrained Transformer learns algebraic representations, and a PPO policy selects valid symbolic transformations to produce shorter expressions.

## 项目背景

散射振幅中的动量守恒与Schouten恒等式，允许将外观不同的旋量括号多项式相互改写。找到更短的表达式需要选择合适的操作顺序，有时还要先扩项、再通过跨项抵消实现化简。

本项目采用两阶段方法：首先使用监督对比学习，让Transformer识别恒等式关联的项；随后冻结encoder，用PPO学习对完整表达式选择恒等式动作。程序枚举合法动作，策略网络给出概率，符号引擎执行变换。

## 四点振幅化简成果

在1024条四点随机构型振幅上，输入为21–40项、量纲≤10，由不超过5项的原始振幅扰乱生成。每个输入使用top-k=4、最多64步，返回路径上项数最少的表达式。

| 量纲 | 数量 | 平均输入项数 | 平均输出项数 | 平均减项率 |
|---|---:|---:|---:|---:|
| 全部 | 1024 | 27.24 | 4.90 | **82.24%** |
| 2–4 | 342 | 22.77 | 3.97 | **82.54%** |
| 5–7 | 341 | 28.80 | 4.18 | **85.67%** |
| 8–10 | 341 | 30.15 | 6.56 | **78.51%** |

平均项数**27.24→4.90**；总项数**27,890→5,021**，总项数减少**82.00%**。平均减项率按每个输入等权统计；21项下限使第一组实际为量纲4。

![分量纲结果与推理进展](graphs/fourpoint_reduction.png)

## 预训练表示

encoder采用768维、3层、8头、FFN3072的Transformer，使用可学习RoPE和温度0.15的监督对比损失。恒等式关联的项组成正样本组；归一化项向量进入对比目标。

**代数关联样本形成簇：**

![代数关联项聚类](graphs/clustring_diffexpr.png)

**嵌入同时组织了外线数和量纲信息：**

| 外线数 | 量纲 |
|---|---|
| ![外线数](graphs/clustering_exleg.png) | ![量纲](graphs/clustring_massdim.png) |

## 强化学习方法

- 将冻结encoder表示与括号指数、系数、外线数特征早期融合，经项间和项内Transformer形成上下文表示。
- 策略头对合法“项＋括号对”动作评分，价值头估计状态价值；encoder之后约361万参数可训练。
- 使用PPO-Clip、分轨迹GAE、价值损失和熵正则；采样与概率重算采用同一个top-k分布。
- 奖励结合减项、分段效率、potential-based shaping和检查点扩项惩罚，通过课程采样学习新构型并复习已有构型。

[当前架构与张量计算](docs/ARCHITECTURE.md) · [环境、推理、数据生成和训练](docs/USAGE.md)

## 快速使用

```powershell
python -m pip install -r requirements.txt
python -B scripts/simplify_policy.py --n-point 4 --dimension 8 --initial-terms 5 --scramble-steps 32 --topk 4 --steps 64 --seed 103
```

```python
from scripts.simplify_policy import PolicySimplifier

solver = PolicySimplifier()
simplified = solver.simplify(expression, n_point=4, topk=4, budget_steps=64, seed=103)
```

## 符号引擎支持

**Forsythia是一套主要面向高能物理的全面计算机符号代数系统，目前处于开发中，尚未开源。** 本项目通过课题组的`forsythia-hep` Python接口进行振幅构造与恒等式运算。如需相关支持，请联系：[zhangrunke@itp.ac.cn](mailto:zhangrunke@itp.ac.cn)。

## 代码导航

| 路径 | 用途 |
|---|---|
| `models/encoder.py`、`models/custom_encoder.py`、`models/rope_bias_*.py` | 预训练encoder与注意力、位置编码 |
| `models/policy.py` | 当前策略/价值网络、合法动作观测、top-k采样 |
| `models/loading.py` | 两份配套权重的读取 |
| `data/symbolic.py`、`data/features.py` | 括号对合法性、代数特征和表达式重建 |
| `data/generation.py`、`scripts/generate_data.py` | encoder正样本组与完整RL振幅生成 |
| `training/pretrain.py`、`training/loss.py` | 对比预训练与损失 |
| `training/ppo.py`、`training/rewards.py`、`training/runtime.py` | 当前PPO、奖励与rollout |
| `scripts/pretrain.py`、`scripts/train_ppo.py` | 配置驱动的训练入口 |
| `scripts/simplify_policy.py` | `SpExpr`推理接口及现场示例 |
| `configs/`、`artifacts/models/` | 当前配置与encoder/策略权重 |

技术栈：Python、PyTorch、Transformer、监督对比学习、PPO、GAE、spinor-helicity、符号代数。

## 许可证

源代码采用[MIT许可](LICENSE)。两份发布的模型权重采用[CC BY-NC 4.0许可](artifacts/models/LICENSE.txt)：允许非商业学术研究、修改和再分发；分享时须注明出处、保留许可并说明修改。商业使用需要获得单独授权。

Forsythia及`forsythia-hep`属于独立的课题组项目，其授权单独处理。相关支持与授权咨询请联系[zhangrunke@itp.ac.cn](mailto:zhangrunke@itp.ac.cn)。
