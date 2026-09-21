# 计时适配器移植验收（阶段记录）

状态：代码与CPU编排测试完成；真实五模型CUDA preflight未运行，不能写成新论文结果。

## 已验证

- 原始 resident 计时脚本 SHA256 与正式 plan 一致：`aa4c2e13e02ec2b9b8f0b745281acc535e001f8eba8a37eb2f08911a579c161e`。
- Lead 使用原始 `rna_design_lm.py` 中的提示词、配对映射和约束 token 函数，与移植助手进行了实际 tokenizer 对比。SL、SL+RL 两个 tokenizer，各检查三个结构、共72个前缀；提示词 token、结构位置、允许 token 顺序一致。
- 该检查只验证 tokenizer 与约束助手，不包含模型 forward/generate、CUDA 数值或候选序列等价性。
- 真实 RNAinverse-pf 九核苷酸子进程 smoke 通过；超时、失败计数和 summary 字段有 CPU 单元测试。未运行完整300组计时。

## 已修复

- 五模型入口已恢复原始外层 `torch.inference_mode()`，保留resident对象生命周期，并固定双重复/4评分进程。
- 已检查 scorer 返回K条、候选逐序列身份、有限耗时与失败语义。
- baseline 配置/tokenizer JSON及外部GoForth入口文件均记录哈希；输出不嵌入私有source root。
- 17项完整CPU回归通过，其中5项专门覆盖计时顺序、候选不匹配、scorer失败/数量与CLI导入。

## 尚未通过

- 五个真实模型的同卡加载、最长目标预热、原始候选逐序列匹配。
- 便携实现与历史运行环境的完整数值/计时等价性。
- 外部 baseline 权重、源码与数据的获取及再分发权限闭环。

不得将mock计时、tokenizer助手一致或单候选smoke写成这些门已通过。
