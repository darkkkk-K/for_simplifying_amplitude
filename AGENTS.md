# 项目开发入口

首先阅读README.md、docs/ARCHITECTURE.md、docs/RESULTS.md和docs/SETUP.md。

主线为冻结768维预训练encoder＋合法动作枚举＋PPO actor–critic。当前允许的网络上下文为外线数、当前项数、历史最小项数。模型参数与本次展示评估设置分别位于configs/policy_096.json和results/evaluation_protocol.json。

项目约定：

- 模型、PPO和数据生成改动使用独立实验配置及ID，明确训练起点、随机数、数据范围和计算预算。
- 原始符号表达式与恒等式执行结果按来源保存；策略通过实际交互奖励学习。
- 新方法提供默认关闭的可选配置，保存修改前状态，沿用固定评估输入比较。
- 临时日志放artifacts/logs/，实验大文件放artifacts/runs/，评估结果放artifacts/evaluations/。
- 当前发布权重和固定1024条展示数据作为复现资产保留；模型清理和对外发布遵循维护者的当次指令。
- 快速接口检查：python -B -m unittest tests.test_inference -q。

