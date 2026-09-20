# 实验与调试索引

这份索引按项目真实发生的顺序列出 Notebook。编号后的字母表示同一阶段的后续版本，不代表它一定比前一版成功。很多版本保留下来，是为了说明某个判断为什么被修改。

## 01 至 03：数据和标签

| Notebook | 做了什么 | 主要结论 |
|---|---|---|
| `01_data_audit_and_split.ipynb` | 审计原始字段，限定 Commons 和目标政党，按 division 与日期切分 | 发现 legislation 字段缺失较多；同一 division 的政党行必须一起切分 |
| `02_bill_linkage_and_polarity_review.ipynb` | 关联 Bill 信息，判断 Aye 相对政策对象的方向，建立 P0 至 P3 人工队列 | 原始 Aye/No 不能直接作为政策立场；Validation/Test 歧义项需要优先审核 |
| `03_finalize_labels_and_build_dataset.ipynb` | 合并人工审核，生成最终标签和 model-ready 数据 | 得到可监督学习的二分类标签，并隔离事后投票字段 |

## 04：模型基线与时间漂移

| Notebook | 做了什么 | 结果与去向 |
|---|---|---|
| `04_baseline_tfidf_classifier.ipynb` | 比较 Global Prior、Party Prior、结构化、TF-IDF 和组合基线 | 最好的 TF-IDF 仍低于 Party Prior，说明相同议案文字对不同政党含义不同 |
| `04b_party_aware_tfidf_validation.ipynb` | 为每个政党学习独立文本权重 | Macro-F1 提升到 0.680，但没有达到预先写下的进取门槛 |
| `04c_hybrid_party_text_institutional_validation.ipynb` | 同时使用共享文本、政党文本和制度特征 | 普通 Validation Macro-F1 0.723，三个开发门槛均通过 |
| `04d_post_election_temporal_validation.ipynb` | 把 2024 Validation 拆成大选后适应期与后续留出期 | Temporal Macro-F1 降至 0.482，发现普通 Validation 掩盖了政权更替影响 |
| `04e_smoothed_institutional_text_ensemble.ipynb` | 混合历史文本、近期 Party Prior 和平滑制度概率 | 最好方案仍未通过时间验证，停止沿这条路线微调 |

## 05 至 06h：知识库和检索

| Notebook | 做了什么 | 结果与去向 |
|---|---|---|
| `05_build_rag_knowledge_base.ipynb` | 建立历史投票、Manifesto 和 Bill 三类语料库 | 初版知识库完成，但 Green Manifesto 一度缺失 |
| `05_build_rag_knowledge_base_v2.ipynb` | 补齐四党官方 Manifesto，增加多通道检索 | 4 个政党资料齐全，并通过基本时间泄漏检查 |
| `06_build_rag_stance_agent.ipynb` | 组装检索包和第一版 Agent 输出合同 | 形成可运行原型，概率校准留给后续模型阶段 |
| `06b_rag_retrieval_audit.ipynb` | 在 24 个分层抽样 division 上审计固定配额检索 | 结构检查通过，但固定 2+5+1 会返回低相关材料 |
| `06c_rag_retrieval_v2.ipynb` | 改为自适应来源数量，减少程序词干扰 | Historical same-domain rate 从 0.265 提升到 0.425 |
| `06d_rag_temporal_retrieval_and_corpus_gap.ipynb` | 使用按查询日期滚动的 TF-IDF 索引，改进双栏 PDF 切分 | 时间安全性提高，平均证据量下降；发现政策对象定义缺口 |
| `06e_rag_semantic_retrieval_and_regression_audit.ipynb` | 加入同义表达、政策锚点和已知坏例回归检查 | 关键回归检查通过，但“相关”仍可能被误当成 Direct |
| `06f_rag_direct_evidence_precision_audit.ipynb` | 用 15 条人工 gold cases 校验证据角色 | v5 从 60% 提升到 v5.2 的 100%；Direct 候选从 412 减到 155 |
| `06g_structured_policy_object_extraction.ipynb` | 把政策对象拆成动作、对象、范围、Bill、Clause 和原文引用 | 2,087 个对象被抽取，但 1,470 个仍需复核，因此保留为诊断工具 |
| `06h_dense_hybrid_retrieval_evaluation.ipynb` | 盲测 TF-IDF、Dense 和不同权重的 Hybrid | 建立本地 384 维 embedding 对照，不使用 API |
| `06h_apply_blind_manual_labels.py` | 回填 06h 盲审结果 | 将人工判断与策略名称解盲后汇总 |

## 07：小规模 LLM 回归

| Notebook | 做了什么 | 结果与去向 |
|---|---|---|
| `07_small_capped_rag_agent_evaluation.ipynb` | 12 条低成本付费调用，检查引用、拒答和过度断言 | Direct accuracy 0.75，但 overclaim rate 0.375 |
| `07b_evidence_gate_and_agent_regression.ipynb` | 收紧证据门槛 | Overclaim 降到 0，但 Direct 案例全部拒答，规则过严 |
| `07c_historical_evidence_contract_and_agent_regression.ipynb` | 明确历史投票的政策对象、程序和角色合同 | Direct accuracy 0.5，仍存在证据角色误判 |
| `07d_source_aware_gate_rescoring.ipynb` | 不调用 API，按来源类型重新计算 07c 结果 | 14 条调试集通过，用于冻结下一版规则；不当作最终独立评测 |

## 08：锁定验证与失败分析

| Notebook | 做了什么 | 结果与去向 |
|---|---|---|
| `08_locked_validation_end_to_end_evaluation.ipynb` | 在未参与调试的 80 条查询上评测完整 Agent | Coverage 0.375，Selective accuracy 0.600，明显低于 Party Prior |
| `08b_locked_validation_failure_analysis_and_gate_repair.ipynb` | 对 12 个错误回答归因，试验更严格门控 | 错误被挡住，但正确答案也大量丢失 |
| `08b1_refined_evidence_gate_diagnostic_v2_1.ipynb` | 按同一 Bill、Clause 和实质政策对象细化规则 | Selective accuracy 0.923，coverage 0.162 |
| `08b2_instrument_scope_patch_diagnostic_v2_2.ipynb` | 修复一条法定文书范围误判 | 旧锁定集错误归零，但 coverage 仅 0.150 |
| `08c1_two_layer_agent_locked_preparation.ipynb` | 冻结新 60 条样本，准备“证据回答 + 模型估计”两层输出 | 生成与旧调试集不重叠的新锁定样本 |
| `08c2_two_layer_agent_paid_evaluation.ipynb` | 运行 60 次付费调用 | 总体 accuracy 0.469，仍低于 Party Prior 0.850；RAG 不再承担最终预测 |

## 09 至 10：最终预测模型

| Notebook | 做了什么 | 结果与去向 |
|---|---|---|
| `09_temporal_model_challenge.ipynb` | Walk-forward 比较近期 Prior、Logistic Regression 和 Time-decay XGBoost | XGBoost 平均分高，但大选后稳定性不足 |
| `09b_role_invariant_xgboost_challenge.ipynb` | 用议会角色代替固定政党身份，并与近期 Prior 混合 | 50/50 方案整体最好，但 Green 的最差政党门槛未通过 |
| `09c_three_role_scope_freeze_audit.ipynb` | 将 MVP 冻结为 Labour、Conservative、Liberal Democrat 三种角色 | Validation Macro-F1 0.753，大选后 0.719，所有三党冻结门槛通过 |
| `10_frozen_three_role_final_test.ipynb` | 验证冻结哈希后只运行一次 2025 至 2026 Test | 最终 Macro-F1 0.801；LibDem 0.456，按产品范围作为实验性较小反对党结果说明 |

## 11 至 12：最终证据层和 Agent

| Notebook | 做了什么 | 结果与去向 |
|---|---|---|
| `11a_prediction_grounded_rag_evidence.ipynb` | 为 975 条冻结预测生成独立证据包 | 检索不读取预测方向，也不读取 Test 标签 |
| `11b_role_aware_rag_evidence_audit.ipynb` | 比较历史投票发生时和当前的议会角色 | 1,418 条不同角色材料降级为背景 |
| `11c_procedure_aware_hybrid_rag_retrieval.ipynb` | 用 Dense + TF-IDF 找候选，再按 Bill、Clause、程序和角色过滤 | 严格 Direct coverage 只有 2.5%，证明完全相同的历史先例很少 |
| `11d_evidence_tiers_and_temporal_audit.ipynb` | 把材料分为 Direct、Related、Similar 和无资料 | 保留相关材料展示，同时禁止其支持方向性结论 |
| `11e_direct_evidence_firewall_and_review.ipynb` | 对全部 Direct 再做防火墙，并导出人工复核表 | Direct 查询率降为 1.2%，最终证据规则冻结 |
| `12_final_agent_demo_and_report_outputs.ipynb` | 组装最终预测、证据、LLM 解释、200 条评测和党派发言示例 | 技术流程完成；200 条 LLM 调用成本约 0.0636 美元 |

## 如何理解这些版本

Notebook 中的 acceptance gate 是开发时预先写下的产品标准，用来阻止看到结果后随意降低要求。它们不是统计学定律。报告会同时写出实际提升和是否达到当时门槛，避免把“没有通过进取门槛”误写成“模型没有改进”。

07d、08b1 和 08b2 使用已经看过的失败案例修规则，因此属于回归或诊断结果。真正独立的检查来自后续锁定样本和最终 Test。
