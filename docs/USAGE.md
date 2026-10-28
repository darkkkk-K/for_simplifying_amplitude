# 环境、推理与训练

## 环境

使用Python3.12，安装适合本机CPU/CUDA环境的PyTorch2.7.1及NumPy：

```powershell
python -m pip install -r requirements.txt
```

本项目还使用课题组的`forsythia-hep`。**Forsythia是一套主要面向高能物理的全面计算机符号代数系统，当前处于开发中，还未开源。** 如需`forsythia-hep`支持，请联系[zhangrunke@itp.ac.cn](mailto:zhangrunke@itp.ac.cn)，获取与操作系统及Python版本匹配的安装包。

源代码使用MIT许可；模型权重使用`artifacts/models/LICENSE.txt`中的CC BY-NC 4.0许可，允许非商业研究，商业使用需要另行授权。Forsythia的授权独立处理。

## 推理

两份配套权重位于`artifacts/models/`，`scripts/simplify_policy.py`默认直接读取它们。

```powershell
python -B scripts/simplify_policy.py --n-point 4 --dimension 8 --topk 4 --steps 64 --scramble-steps 32 --seed 103
```

Python接口：

```python
from scripts.simplify_policy import PolicySimplifier

solver = PolicySimplifier()
best_expression = solver.simplify(expression, n_point=4, topk=4, budget_steps=64, seed=103)
print(solver.last_info)
```

`last_info`包含初始/末状态/路径最小项数、每步动作和执行次数。可用`--policy`与`--encoder`指定其他配套权重。

## 数据生成

各入口默认显示配置，添加`--run`执行；输出使用新的目录。

```powershell
# 完整振幅训练数据：随机五点、量纲4–5、≤36项，从≤5项短式扰乱生成。
python -B scripts/generate_data.py --stage rl --n-point 5 --dimension-min 4 --dimension-max 5 --count 1792 --max-terms 36 --output datasets/course52 --run

# 四点复习池。
python -B scripts/generate_data.py --stage rl --n-point 4 --dimension-min 10 --dimension-max 11 --count 1024 --output datasets/course43 --run
python -B scripts/generate_data.py --stage rl --n-point 4 --dimension-min 6 --dimension-max 9 --count 1024 --output datasets/course42 --run

# 预训练关联项组：按原预训练构型生成方法构造并划分train/val。
python -B scripts/generate_data.py --stage encoder --count 10000 --output datasets/pretrain --run
```

RL的`train.json`只保存扰乱后的输入和物理构型；`origins.json`保存生成来源。量纲、外线数、长度及扰乱次数均可用命令行设置。`--candidate-cap`控制候选上限。

## 预训练encoder

```powershell
python -B scripts/pretrain.py --config configs/pretrain.json
python -B scripts/pretrain.py --config configs/pretrain.json --run
```

配置指定768维RoPE架构、对比温度、学习率、batch大小和训练轮数。预设使用1e-5学习率、AdamW及线性预热＋余弦退火，数据从`datasets/pretrain/train_data`和`val_data`读取。

## 训练当前PPO方法

```powershell
# 查看当前训练预设。
python -B scripts/train_ppo.py --config configs/ppo.json

# 新初始化策略/价值网络，冻结预训练encoder。
python -B scripts/train_ppo.py --config configs/ppo.json --output artifacts/runs/ppo_new --run

# 从发布的策略权重开始新的训练。
python -B scripts/train_ppo.py --config configs/ppo.json --checkpoint artifacts/models/rl/policy_096.pt --output artifacts/runs/ppo_finetune --run

# 从本训练入口保存的完整恢复文件继续。
python -B scripts/train_ppo.py --config configs/ppo.json --resume artifacts/runs/ppo_new/recovery.pt --output artifacts/runs/ppo_resumed --run
```

每轮32×24条真实转移、batch16、epoch2，共96次更新；预设36轮，共3456次更新。完整恢复文件包含模型、Adam状态、来源池与随机数状态；下一轮重新从数据集抽取初始表达式。训练记录每4轮报告真实轨迹末状态平均减项率。
