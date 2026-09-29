# 推理 / Inference

双模型权重在独立的 [Hugging Face 模型仓](https://huggingface.co/jojojoojooo/RNA-IFlow) 归档。该模型仓已公开，权重采用 Apache-2.0；第三方 backbone 保留其原许可说明。GitHub 仓库只保存代码与结果资料。

完整模型目录应包含 `model.safetensors`、`config.json`、`export_manifest.json`。加载器验证权重 SHA 并严格恢复全部参数和 buffer，不需要原始 backbone 目录或 tokenizer 文件。

```bash
python scripts/load_export.py /path/to/model
python scripts/infer_flow.py --model /path/to/RNA-IFlow --structure '(((...)))...' --candidates 8
python scripts/infer.py --model /path/to/RNA-IFlow-RL --structure '(((...)))...' --candidates 8
```

默认使用 CPU；可通过 `--device cuda:0` 指定 GPU。此命令是单 target 示例，不计算正式 benchmark 汇总。

正式论文评估使用冻结的多条件协议，每条件 K=8、8 个 transition steps、ViennaRNA 2.7.2。数据集评估须保留 task-index seed 映射，不能把单 target 示例简单循环后当成同协议结果。

2026-09-29 从 HF 仓全新下载 RL 权重，SHA256 与原导出一致；用 GitHub 公开代码在 CPU 上完成 K=8 单目标生成和 ViennaRNA 评分。监督模型导出后经张量及固定双样本输出核对，并完成 CPU K=8 单目标生成。两项仅是工程 smoke，不支持解题率或泛化 claim，也不是论文 benchmark 重跑。

论文评分经 `scripts/paper_metrics.py` 适配：ViennaRNA 2.7.2 的 `ensemble_defect` 已归一化，不再次除以序列长度。历史训练 scorer 保持原样；新推理示例使用与论文结果整理相同的纠正口径。回归测试：`python tests/test_paper_metrics.py`，直接对照 ViennaRNA 的 NED、target probability、MFE 和 uMFE 判定。
