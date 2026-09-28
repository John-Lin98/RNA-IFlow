# 推理 / Inference

主模型权重目前未公开；以下命令需要用户已获授权的本地模型副本。本 GitHub 仓库只发布代码与结果资料。

完整模型目录应包含 `model.safetensors`、`config.json`、`export_manifest.json`。加载器验证权重 SHA 并严格恢复全部参数和 buffer，不需要原始 backbone 目录或 tokenizer 文件。

```bash
python scripts/load_export.py /path/to/model
python scripts/infer.py --model /path/to/model --structure '(((...)))' --candidates 8
```

默认使用 CPU；可通过 `--device cuda:0` 指定 GPU。此命令是单 target 示例，不计算正式 benchmark 汇总。

正式论文评估使用冻结的多条件协议，每条件 K=8、8 个 transition steps、ViennaRNA 2.7.2。数据集评估须保留 task-index seed 映射，不能把单 target 示例简单循环后当成同协议结果。

历史模型包的 CPU smoke 曾以目标 `(((...)))`、2 个候选验证生成和 ViennaRNA 评估链路。本次公开代码 QA 未取得权重，未重跑该项。该小样本仅是工程验证，不支持解题率或泛化 claim。

论文评分经 `scripts/paper_metrics.py` 适配：ViennaRNA 2.7.2 的 `ensemble_defect` 已归一化，不再次除以序列长度。历史训练 scorer 保持原样；新推理示例使用与论文结果整理相同的纠正口径。回归测试：`python tests/test_paper_metrics.py`，直接对照 ViennaRNA 的 NED、target probability、MFE 和 uMFE 判定。
