# RNA-IFlow-RL 模型引用

Hugging Face: https://huggingface.co/jojojoojooo/RNA-IFlow-RL （private）。

主模型为 C3+D5 U2442。原始 checkpoint SHA256：
`198fd79e7680f4b01e758f063aadab00c0d4e6709ac4d2a220249a267a70ebe8`。

完整推理导出 SHA256：
`8a8dcf74014be2e3ab9571a19facebaad639ca4fd6574bc97d7bdf5d9ffe9f9d`。

导出包含冻结 backbone、监督参数及 U2442 更新参数，不含 optimizer/RNG。不能作为精确续训 checkpoint。原始 .pt 暂不额外上传：其权重是局部状态，精确恢复还依赖原始数据、合同、父模型与 RNG；本次优先提供独立推理模型，避免误导和重复权重。

223 个张量逐项相等，固定双样本 CPU 输出差异为 0。权重已上传到 revision `774a352f55016ff32012d2936d629b12bd6045dc`，HF LFS 元数据记录的 SHA256 与上述导出 SHA 相同、大小为 350239076 bytes。fresh download SHA 与下载后 load smoke 仍待验收；远端元数据不替代实际下载验证。

使用方法见 [推理文档](../docs/inference.md)。许可证仍待完整核验，仓库与模型均不得擅自公开。
