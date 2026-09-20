# 结果快照

这里保存最终模型和 Agent 评测的少量关键结果。完整中间产物位于本地 `processed/`，可以通过 Notebook 重建。

## 最终模型

- Macro-F1：0.801
- Accuracy：0.808
- ROC-AUC：0.869
- Brier score：0.170
- Division bootstrap 95% CI：[0.772, 0.828]

三党 Macro-F1：

| 政党 | Macro-F1 | Accuracy | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
| Conservative | 0.820 | 0.832 | 0.942 | 0.180 |
| Labour | 0.839 | 0.853 | 0.951 | 0.172 |
| Liberal Democrat | 0.456 | 0.730 | 0.779 | 0.157 |

## Agent

- 最终公共预测：975 条
- 严格 Direct evidence 查询率：1.2%
- 任意外部材料查询率：23.9%
- 锁定 LLM 评测：200 条
- 200 条调用成本：约 0.0636 美元

CSV 中的最终概率来自冻结模型。RAG 和 LLM 不会覆盖概率。

## 文件说明

| 文件 | 内容 |
|---|---|
| `final_test_model_metrics_v1.csv` | 最终 Test 整体指标 |
| `final_test_party_metrics_v1.csv` | 分政党指标 |
| `final_test_year_metrics_v1.csv` | 分年份指标 |
| `final_test_predictions_public_v1.csv` | 不含真实 Test 标签的公共预测结果 |
| `final_agent_predictions_v1.csv` | 975 条最终 Agent 预测包 |
| `final_agent_evidence_v1.csv` | 最终展示层使用的证据分层结果 |
| `locked_200_agent_results_v1.csv` | 200 条锁定 LLM 回答 |
| `locked_200_agent_manual_review_40_v1.csv` | 从 200 条中抽取的人工审核队列 |
| `structured_spokesperson_test_results_v4.csv` | 结构化党派发言试验结果 |
| `final_test_confusion_matrix_v1.png` | 最终 Test 混淆矩阵 |
| `final_test_party_macro_f1_v1.png` | 分政党 Macro-F1 |
| `external_evidence_availability.png` | 外部材料覆盖情况 |

`final_test_scored_predictions_private_v1.csv` 含逐行真实标签和预测对照，只保留在本地 `processed/`，避免公开结果文件意外成为后续调参输入。
