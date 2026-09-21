# Notebook 阅读顺序

仓库保留 40 个 Notebook。它们既包含最终方案，也包含没有通过验证的尝试。建议从仓库根目录运行 `jupyter lab`，这样 Notebook 中的 `Path.cwd()` 会指向正确的项目目录。

## 1. 数据与标签

目录：[`01_data_and_labels`](01_data_and_labels/)

从原始 CSV 开始，完成数据审计、时间切分、Polarity 判断、人工复核和最终标签生成。后面的实验都依赖这一阶段输出的 `processed/model_v2/`。

## 2. 模型开发

目录：[`02_model_development`](02_model_development/)

依次比较 Party Prior、TF-IDF、各党独立文本权重、Hybrid Logistic Regression，以及2024年大选后的时间留出表现。普通 Validation 上表现不错的模型，在政权更替后明显下降。

## 3. RAG 检索

目录：[`03_rag_retrieval`](03_rag_retrieval/)

建立历史投票、Manifesto 和 Bill 知识库，逐步加入滚动时间索引、Dense Embedding、政策对象抽取和 Direct evidence 精度检查。

## 4. Agent 评测

目录：[`04_agent_evaluation`](04_agent_evaluation/)

包含12至14条小规模 LLM 回归、80条锁定验证、失败归因、证据门控修复和60条两层 Agent 评测。这里记录了拒答过少、拒答过多以及 RAG 预测低于简单基线的过程。

## 5. 最终系统

目录：[`05_final_system`](05_final_system/)

这一阶段完成 Walk-forward 模型挑战、三角色范围冻结、一次性最终 Test、最终证据防火墙和 200 条 Agent 评测。

`12_final_agent_demo_and_report_outputs.ipynb` 是冻结系统演示入口。`13_unseen_motion_agent_orchestration.ipynb` 使用 4 个外部未见 motion，演示原始文本解析、部署模型刷新、工具选择、Hybrid RAG、一次补充检索和可选 LLM 报告。`13b_interactive_debate_agent.ipynb` 在 `13` 的冻结结果上增加自然语言对话、模拟党派发言、政策效果、三个 debate points、反方观点和回应。

每个 Notebook 的具体结果与去向见 [`docs/EXPERIMENT_INDEX.md`](../docs/EXPERIMENT_INDEX.md)。
