# RNA-IFlow-RL 模型引用

Hugging Face: https://huggingface.co/jojojoojooo/RNA-IFlow-RL （private）。

主模型为 C3+D5 U2442。原始 checkpoint SHA256：
`198fd79e7680f4b01e758f063aadab00c0d4e6709ac4d2a220249a267a70ebe8`。

完整推理导出 SHA256：
`8a8dcf74014be2e3ab9571a19facebaad639ca4fd6574bc97d7bdf5d9ffe9f9d`。

导出包含冻结 backbone、监督参数及 U2442 更新参数，不含 optimizer/RNG。不能作为精确续训 checkpoint。原始 .pt 暂不额外上传：其权重是局部状态，精确恢复还依赖原始数据、合同、父模型与 RNG；本次优先提供独立推理模型，避免误导和重复权重。

223 个张量逐项相等，固定双样本 CPU 输出差异为 0。当前 metadata revision 为 `a3ca9b831865d0e499a6f5163e17bde3b154c2cc`；350239076-byte 权重 SHA 保持不变。独立 fresh download 后，模型及 8 个元数据文件均通过下载包内的 SHA256 清单校验。下载包已在全新 CPU 环境严格加载并完成 K=8 单目标推理与 ViennaRNA 评分；这是发布链路 smoke，不是完整 benchmark 重跑。详见 `results/provenance/hf_readback_qa.json`。

使用方法见 [推理文档](../docs/inference.md)。项目采用 Apache-2.0；第三方依赖与数据仍遵循各自条款。仓库与模型均不得擅自公开。

加载器同时固定权重与配置身份。对应上述 HF revision 的 `config.json` SHA256 为 `fd84cd17150e1ad1d9e756a05a78dc13c24463ac27bcf4c771e5c7e8f07375fb`，仓内参考副本为 `configs/u2442.json`。任何配置字节变更（包括可保持张量形状但改变输出的激活函数）均被拒绝；不能只保留权重 SHA 就声称仍为已验收主模型。
