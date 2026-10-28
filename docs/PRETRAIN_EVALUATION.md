# 查看预训练表示

三个脚本分别观察恒等式关联样本、量纲和外线数。`--model`指定encoder，脚本自动生成样本、计算嵌入并绘制t-SNE。

```powershell
python -B scripts/visualize_tsne_sim.py --model artifacts/models/pretrained/temp15_final_model_sl_256.pth --groups 12 --rounds 128
python -B scripts/visualize_tsne_dim.py --model artifacts/models/pretrained/temp15_final_model_sl_256.pth --samples 256
python -B scripts/visualize_tsne_ex.py --model artifacts/models/pretrained/temp15_final_model_sl_256.pth --samples 512
```

`--help`列出数据数量、t-SNE和输出参数。模型路径可以换成其他encoder权重，隐藏维度由检查点自动读取。每次运行使用独立输出目录，图像与相应嵌入、参数一并保存。

本项目首页的三张图位于`graphs/`：`clustring_diffexpr.png`按代数关联来源着色，`clustring_massdim.png`按量纲着色，`clustering_exleg.png`按外线数着色。图中每个点对应一个归一化的项向量。
