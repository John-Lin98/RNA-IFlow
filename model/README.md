# RNA-IFlow / RNA-IFlow-RL 模型引用

两组完整推理权重由独立的 [HF 模型仓](https://huggingface.co/jojojoojooo/RNA-IFlow) 管理，不随 GitHub 代码仓存放。HF 仓已公开，模型权重采用 Apache-2.0，详见模型卡和第三方许可说明。

监督 RNA-IFlow 原始 checkpoint SHA256：
`8e221cc4c4382a5421890724c0548cf64492fe1be4556d498878e86e83056bc5`。
完整推理导出 SHA256：
`4f33b51d28aa0fa7fab7a27cd6b0422c2eed9654f204bcedb80a771851c5a151`。

主模型为 C3+D5 U2442。原始 checkpoint SHA256：
`198fd79e7680f4b01e758f063aadab00c0d4e6709ac4d2a220249a267a70ebe8`。

完整推理导出 SHA256：
`8a8dcf74014be2e3ab9571a19facebaad639ca4fd6574bc97d7bdf5d9ffe9f9d`。

导出包含冻结 backbone、监督参数及 U2442 更新参数，不含 optimizer/RNG。不能作为精确续训 checkpoint。原始 .pt 暂不额外上传：其权重是局部状态，精确恢复还依赖原始数据、合同、父模型与 RNG；本次优先提供独立推理模型，避免误导和重复权重。

两组导出各有 223 个张量逐项相等，固定双样本 CPU 输出差异为 0。RL 模型从 HF 仓全新下载后，通过 SHA256 校验并在 CPU 上严格加载、完成 K=8 单目标推理与 ViennaRNA 评分；监督模型通过 CPU K=8 单目标生成。两组权重均从该 HF 仓独立下载，SHA256 与原导出一致，并通过 CPU 单目标推理。这是发布链路 smoke，不是完整 benchmark 重跑。

使用方法见 [推理文档](../docs/inference.md)。项目代码采用 Apache-2.0；第三方依赖与数据仍遵循各自条款。

加载器同时固定两组权重与各自配置身份。仓内 RL 参考配置为 `configs/u2442.json`。配置字节变更会被拒绝；不能只保留权重 SHA 就声称仍为已验收模型。
