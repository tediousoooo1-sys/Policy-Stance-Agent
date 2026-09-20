import json
from pathlib import Path


ROOT = Path.cwd()
OUTPUT = ROOT / "11b_role_aware_rag_evidence_audit.ipynb"


def lines(text):
    text = text.strip("\n")
    return [line + "\n" for line in text.splitlines()]


cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": lines(text)})


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    })


md(r'''
# 11B — Role-Aware RAG Evidence Audit

## 这一步解决什么问题

11A发现很多历史投票与模型预测方向不同，但这不一定代表模型错误。

例如：

- 2023年的Labour是主要反对党；
- 2025年的Labour是执政党；
- 即使两次议案主题相似，政党在不同制度角色下也可能采取相反立场。

因此11B给每条历史证据补上“当时的政党角色”，再判断它是否有资格作为方向证据。这里必须强调：**政策内容永远是第一关，政治角色只是第二关。**

```text
同一政党 + 相似政策 + 相同制度角色
→ 可以作为支持/反对的历史类比

同一政党 + 相似政策 + 不同制度角色
→ 保留为背景，但不能直接挑战当前预测
```

可以把它理解成找案例时的两道门：

1. **内容门**：两条议案是否真的讨论同一种税收、福利、教育或其他政策？11A已经完成这一关；
2. **环境门**：即使内容相似，当时这个党是负责执行政府方案，还是负责监督和反对政府？11B检查这一关。

例如，同样是增加某项税收：执政党可能为了执行自己的预算而支持，主要反对党可能反对；政府更替后，相似政策由另一个党提出，投票方向可能交换。这不表示政策内容不重要，而是说明“相同主题”还不足以保证两个历史场景完全可比。

## 重要边界

- 11B不重新训练模型；
- 不读取Test真实标签；
- 不重新调用LLM或API；
- 不删除反方向资料；
- 不按证据数量投票；
- 不修改冻结概率。
''')

md(r'''
## 为什么先重分级，而不是立即重建RAG

这是一个节省时间的诊断步骤。

11A已经为每条查询找到了最多8条语义相关材料。11B先检查这些材料中，有多少只是因为“历史政治角色不同”才与模型冲突。

- 如果角色修正后仍有足够的同角色证据，就直接使用新的证据分级；
- 如果大部分查询失去可用方向证据，再做11C角色感知重新检索；
- 这样避免在不知道问题来源时反复调整检索参数。
''')

code(r'''
# 导入本地审计库；这一Notebook没有付费调用
import hashlib
import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 180)
sns.set_theme(style="whitegrid")

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "processed"
MODEL_DIR = PROCESSED_DIR / "model_v2"
INPUT_DIR = PROCESSED_DIR / "final_rag_evidence_v1"
OUTPUT_DIR = PROCESSED_DIR / "role_aware_rag_audit_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = MODEL_DIR / "model_train_v2.csv"
VALIDATION_PATH = MODEL_DIR / "model_validation_v2.csv"
QUERIES_PATH = INPUT_DIR / "final_rag_queries_v1.csv"
EVIDENCE_PATH = INPUT_DIR / "final_rag_evidence_v1.csv"
PACKETS_PATH = INPUT_DIR / "final_agent_evidence_packets_v1.csv"

ELECTION_TRANSITION_DATE = pd.Timestamp("2024-07-05")
RANDOM_STATE = 42
MANUAL_SAMPLE_PER_PARTY = 10

print("Notebook version: 11b-role-aware-rag-audit-v2")
print("API calls: 0")
print("LLM calls: 0")
print("Test labels read: False")
print("Retrieval rerun: False")
print("Prediction override allowed: False")
print("Output directory:", OUTPUT_DIR)
''')

md(r'''
## 1. 读取11A结果和历史制度信息

历史Train和Validation只读取角色、政府背书、议案类型等解释字段，不读取任何`target_*`标签。
''')

code(r'''
HISTORICAL_CONTEXT_COLUMNS = [
    "division_key",
    "party",
    "motion_date",
    "party_role",
    "final_object_government_backed",
    "government_backing_known",
    "motion_family",
    "policy_domain_primary",
]

queries = pd.read_csv(QUERIES_PATH, low_memory=False)
evidence = pd.read_csv(EVIDENCE_PATH, low_memory=False)
packets = pd.read_csv(PACKETS_PATH, low_memory=False)
train_context = pd.read_csv(
    TRAIN_PATH,
    usecols=HISTORICAL_CONTEXT_COLUMNS,
    low_memory=False,
)
validation_context = pd.read_csv(
    VALIDATION_PATH,
    usecols=HISTORICAL_CONTEXT_COLUMNS,
    low_memory=False,
)

queries["motion_date"] = pd.to_datetime(queries["motion_date"], errors="raise")
evidence["query_date"] = pd.to_datetime(evidence["query_date"], errors="raise")
evidence["evidence_date"] = pd.to_datetime(
    evidence["evidence_date"], errors="coerce"
)

historical_context = pd.concat(
    [train_context, validation_context], ignore_index=True
)
historical_context["motion_date"] = pd.to_datetime(
    historical_context["motion_date"], errors="raise"
)

assert not any(
    column.startswith("target_")
    for column in HISTORICAL_CONTEXT_COLUMNS
)
assert queries["query_id"].is_unique
assert packets["query_id"].is_unique
assert not evidence.duplicated(["query_id", "evidence_chunk_id"]).any()

print("Queries:", len(queries))
print("Evidence rows:", len(evidence))
print("Historical context rows:", len(historical_context))
''')

md(r'''
## 2. 给历史证据补充当时的政党角色

正常情况下直接使用数据中的`party_role`。如果某条历史记录缺少角色，则根据日期和政党进行透明回退：

- 2024-07-05之前：Conservative为执政党，Labour为主要反对党；
- 2024-07-05及之后：Labour为执政党，Conservative为主要反对党；
- Liberal Democrat在本项目范围内归为较小反对党。

回退来源会单独记录，避免把推断伪装成原始字段。
''')

code(r'''
def safe_text(value):
    # 安全处理CSV缺失值并压缩空白
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_bool(value):
    # CSV布尔值可能以True、1或字符串形式保存
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return safe_text(value).lower() in {"true", "1", "yes", "y"}


def normalize_role(value):
    # 将不同写法统一为三个产品角色
    value = safe_text(value).lower().replace(" ", "_")
    mapping = {
        "government": "governing_party",
        "governing": "governing_party",
        "governing_party": "governing_party",
        "main_opposition": "main_opposition",
        "official_opposition": "main_opposition",
        "smaller_opposition": "smaller_opposition",
        "minor_opposition": "smaller_opposition",
    }
    return mapping.get(value, value or "unknown")


def infer_role_from_date(party, motion_date):
    # 只在历史元数据缺失时使用公开制度时间线回退
    party = safe_text(party).lower()
    motion_date = pd.Timestamp(motion_date)
    if party == "liberal-democrat":
        return "smaller_opposition"
    if motion_date < ELECTION_TRANSITION_DATE:
        return {
            "conservative": "governing_party",
            "labour": "main_opposition",
        }.get(party, "unknown")
    return {
        "labour": "governing_party",
        "conservative": "main_opposition",
    }.get(party, "unknown")


historical_context = historical_context.drop_duplicates(
    ["division_key", "party"]
).copy()
historical_context["recorded_historical_role"] = (
    historical_context["party_role"].map(normalize_role)
)
historical_context["inferred_historical_role"] = historical_context.apply(
    lambda row: infer_role_from_date(row["party"], row["motion_date"]),
    axis=1,
)
historical_context["historical_role_source"] = np.where(
    historical_context["recorded_historical_role"].ne("unknown"),
    "model_metadata",
    "date_party_fallback",
)
historical_context["historical_party_role"] = np.where(
    historical_context["recorded_historical_role"].ne("unknown"),
    historical_context["recorded_historical_role"],
    historical_context["inferred_historical_role"],
)

history_lookup = historical_context.rename(columns={
    "division_key": "evidence_division_key",
    "party": "evidence_party",
    "final_object_government_backed": "historical_government_backing",
    "government_backing_known": "historical_backing_known",
    "motion_family": "historical_motion_family",
    "policy_domain_primary": "historical_policy_domain",
    "motion_date": "historical_motion_date",
})[[
    "evidence_division_key",
    "evidence_party",
    "historical_motion_date",
    "historical_party_role",
    "historical_role_source",
    "historical_government_backing",
    "historical_backing_known",
    "historical_motion_family",
    "historical_policy_domain",
]]

query_context = queries[[
    "query_id",
    "party_role",
    "final_object_government_backed",
    "government_backing_known",
    "motion_family",
    "policy_domain_primary",
]].rename(columns={
    "party_role": "current_party_role",
    "final_object_government_backed": "current_government_backing",
    "government_backing_known": "current_backing_known",
    "motion_family": "current_motion_family",
    "policy_domain_primary": "current_policy_domain",
})

enriched = evidence.merge(
    query_context,
    on="query_id",
    how="left",
    validate="many_to_one",
)
enriched = enriched.merge(
    history_lookup,
    on=["evidence_division_key", "evidence_party"],
    how="left",
    validate="many_to_one",
)

enriched["current_party_role"] = enriched["current_party_role"].map(
    normalize_role
)
enriched["historical_party_role"] = enriched[
    "historical_party_role"
].fillna("unknown").map(normalize_role)

historical_mask = enriched["source_type"].eq("historical_vote")
historical_metadata_coverage = float(
    enriched.loc[historical_mask, "historical_party_role"]
    .ne("unknown").mean()
)

print("Historical role metadata coverage:", round(historical_metadata_coverage, 3))
display(
    enriched.loc[historical_mask, "historical_role_source"]
    .value_counts(dropna=False)
    .to_frame("evidence_rows")
)
''')

md(r'''
## 3. 判断历史环境是否可比较

这里不是简单地说“同角色一定正确”。它只是决定一条历史投票能否用于判断当前方向。

方向证据必须满足：

1. 11A已经通过语义方向门槛；
2. 历史政党角色与当前角色相同；
3. 如果两边的政府背书状态都明确，则背书方向不能相反。

不同角色或明确相反的政府背书不会被删除，而是降级为背景资料。
''')

code(r'''
def compare_backing(row):
    # 只有两边背书信息都明确时才比较相同或不同
    current_known = parse_bool(row.get("current_backing_known"))
    historical_known = parse_bool(row.get("historical_backing_known"))
    current_value = safe_text(row.get("current_government_backing")).lower()
    historical_value = safe_text(row.get("historical_government_backing")).lower()
    if not current_known or not historical_known:
        return "unknown"
    if not current_value or not historical_value:
        return "unknown"
    return "same" if current_value == historical_value else "different"


enriched["role_relation"] = np.where(
    ~historical_mask,
    "not_applicable",
    np.where(
        enriched["historical_party_role"].eq("unknown"),
        "unknown",
        np.where(
            enriched["historical_party_role"].eq(
                enriched["current_party_role"]
            ),
            "same_role",
            "different_role",
        ),
    ),
)
enriched["government_backing_relation"] = enriched.apply(
    compare_backing, axis=1
)
enriched["same_motion_family"] = (
    enriched["historical_motion_family"].fillna("").astype(str)
    .eq(enriched["current_motion_family"].fillna("").astype(str))
)
enriched["same_policy_domain"] = (
    enriched["historical_policy_domain"].fillna("").astype(str)
    .eq(enriched["current_policy_domain"].fillna("").astype(str))
)
enriched["evidence_age_days"] = (
    enriched["query_date"] - enriched["evidence_date"]
).dt.days

original_directional = enriched["directional_evidence_gate"].map(parse_bool)
enriched["role_aware_directional_gate"] = (
    historical_mask
    & original_directional
    & enriched["role_relation"].eq("same_role")
    & ~enriched["government_backing_relation"].eq("different")
)

def classify_evidence_role(row):
    # 将材料分成方向类比、政策背景和Bill背景
    source_type = safe_text(row.get("source_type"))
    if source_type == "bill_reference":
        return "bill_background_only"
    if source_type == "manifesto":
        return "party_policy_context_direction_unresolved"
    if source_type != "historical_vote":
        return "context_only"
    if not parse_bool(row.get("directional_evidence_gate")):
        return "context_low_directional_precision"
    if row.get("role_relation") == "different_role":
        return "context_different_political_role"
    if row.get("role_relation") == "unknown":
        return "context_unknown_political_role"
    if row.get("government_backing_relation") == "different":
        return "context_different_government_backing"
    if parse_bool(row.get("direct_policy_match")):
        return "strong_same_role_historical_analogue"
    return "related_same_role_historical_analogue"


enriched["evidence_role_v2"] = enriched.apply(
    classify_evidence_role, axis=1
)
enriched["relationship_to_prediction_v2"] = np.where(
    enriched["role_aware_directional_gate"],
    enriched["relationship_to_prediction"],
    np.where(
        enriched["source_type"].eq("manifesto"),
        "requires_direction_judgment",
        "context_only",
    ),
)
enriched["evidence_is_current_vote_ground_truth"] = False
enriched["prediction_overridden"] = False

role_relation_table = pd.crosstab(
    enriched.loc[historical_mask, "query_party"],
    enriched.loc[historical_mask, "role_relation"],
    normalize="index",
).round(3)
display(role_relation_table)
display(enriched["evidence_role_v2"].value_counts().to_frame("rows"))
''')

md(r'''
## 4. 重新生成查询级证据状态

“只找到反方向历史案例”现在只统计**同角色且制度环境可比较**的材料。

不同角色的反向材料仍留在证据包中，但显示为背景，而不是自动判断模型错了。
''')

code(r'''
def classify_query_status(group):
    # 只用角色感知方向证据判断一致、冲突或缺少方向证据
    directional = group[
        group["role_aware_directional_gate"]
    ]
    has_support = directional["relationship_to_prediction_v2"].eq(
        "supports_prediction"
    ).any()
    has_challenge = directional["relationship_to_prediction_v2"].eq(
        "contradicts_prediction"
    ).any()
    if has_support and has_challenge:
        return "mixed_same_role_historical_analogues"
    if has_support:
        return "aligned_same_role_historical_analogue_found"
    if has_challenge:
        return "challenging_same_role_historical_analogue_found"
    if group["source_type"].eq("manifesto").any():
        return "context_requires_direction_review"
    return "context_only_or_insufficient"


query_status_v2 = (
    enriched.groupby("query_id", sort=False)
    .apply(classify_query_status)
    .rename("evidence_status_v2")
    .reset_index()
)

direction_counts = enriched.groupby("query_id").agg(
    role_aware_supporting_history=(
        "relationship_to_prediction_v2",
        lambda values: int((values == "supports_prediction").sum()),
    ),
    role_aware_challenging_history=(
        "relationship_to_prediction_v2",
        lambda values: int((values == "contradicts_prediction").sum()),
    ),
    same_role_historical_rows=(
        "role_relation",
        lambda values: int((values == "same_role").sum()),
    ),
    different_role_context_rows=(
        "evidence_role_v2",
        lambda values: int((values == "context_different_political_role").sum()),
    ),
).reset_index()

query_audit_v2 = packets.merge(
    query_status_v2,
    on="query_id",
    how="left",
    validate="one_to_one",
).merge(
    direction_counts,
    on="query_id",
    how="left",
    validate="one_to_one",
)
query_audit_v2["evidence_status_v2"] = query_audit_v2[
    "evidence_status_v2"
].fillna("context_only_or_insufficient")
query_audit_v2["prediction_overridden"] = False

status_comparison = pd.crosstab(
    query_audit_v2["evidence_status"],
    query_audit_v2["evidence_status_v2"],
)
display(status_comparison)
display(
    query_audit_v2["evidence_status_v2"]
    .value_counts()
    .to_frame("queries")
)
''')

md(r'''
## 5. 生成固定30条人工检查样本

每个产品内政党抽10条，共30条。抽样使用固定`random_state=42`，并优先覆盖不同证据状态。

人工检查时只需要回答：

1. 历史政策对象是否真的与当前对象相似？
2. 同角色方向证据是否真的可以比较？
3. 被降级的不同角色证据是否确实只适合作为背景？

这一步不需要阅读全部5,304条证据。
''')

code(r'''
def stable_hash(value):
    # 使用固定哈希保证每次运行抽到相同查询
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


sample_parts = []
for party, group in query_audit_v2.groupby("party", sort=True):
    group = group.copy()
    group["stable_hash"] = group["query_id"].map(stable_hash)
    group["status_order"] = group.groupby(
        "evidence_status_v2"
    ).cumcount()
    chosen = group.sort_values(
        ["status_order", "evidence_status_v2", "stable_hash"]
    ).head(MANUAL_SAMPLE_PER_PARTY)
    sample_parts.append(chosen)

manual_queries = pd.concat(sample_parts, ignore_index=True)
manual_evidence = enriched[
    enriched["query_id"].isin(manual_queries["query_id"])
].copy()
manual_evidence = manual_evidence.sort_values(["query_id", "rank"])

manual_evidence["manual_policy_relevance"] = ""
manual_evidence["manual_direction_valid"] = ""
manual_evidence["manual_role_classification_valid"] = ""
manual_evidence["manual_notes"] = ""

assert len(manual_queries) == 30
assert manual_queries.groupby("party").size().eq(10).all()

display(pd.crosstab(
    manual_queries["party"],
    manual_queries["evidence_status_v2"],
))
print("Manual-review queries:", len(manual_queries))
print("Manual-review evidence rows:", len(manual_evidence))
''')

md(r'''
## 6. 保存结果并进行免费验收

通过门槛只代表角色分级逻辑运行正确，不代表证据语义已经百分之百正确。语义质量需要查看固定30条样本后再决定。
''')

code(r'''
original_directional_rows = int(
    enriched["directional_evidence_gate"].map(parse_bool).sum()
)
role_aware_directional_rows = int(
    enriched["role_aware_directional_gate"].sum()
)
different_role_downgraded_rows = int((
    historical_mask
    & enriched["directional_evidence_gate"].map(parse_bool)
    & enriched["role_relation"].eq("different_role")
    & ~enriched["role_aware_directional_gate"]
).sum())

party_summary = query_audit_v2.groupby("party").agg(
    queries=("query_id", "size"),
    queries_with_same_role_directional=(
        "same_role_historical_rows",
        lambda values: float((values > 0).mean()),
    ),
    queries_with_aligned_history=(
        "role_aware_supporting_history",
        lambda values: float((values > 0).mean()),
    ),
    queries_with_challenging_history=(
        "role_aware_challenging_history",
        lambda values: float((values > 0).mean()),
    ),
    mean_different_role_context=("different_role_context_rows", "mean"),
).reset_index()

structural_gates = {
    "all_queries_preserved_gate": bool(
        len(query_audit_v2) == len(queries) == len(packets)
    ),
    "historical_role_metadata_gate": bool(
        historical_metadata_coverage >= 0.99
    ),
    "different_role_never_directional_gate": bool(
        not enriched.loc[
            enriched["role_relation"].eq("different_role"),
            "role_aware_directional_gate",
        ].any()
    ),
    "different_backing_never_directional_gate": bool(
        not enriched.loc[
            enriched["government_backing_relation"].eq("different"),
            "role_aware_directional_gate",
        ].any()
    ),
    "prediction_not_overridden_gate": bool(
        query_audit_v2["prediction_overridden"].eq(False).all()
    ),
    "evidence_not_ground_truth_gate": bool(
        enriched["evidence_is_current_vote_ground_truth"].eq(False).all()
    ),
    "fixed_manual_sample_gate": bool(
        len(manual_queries) == 30
        and manual_queries.groupby("party").size().eq(10).all()
    ),
    "temporal_leakage_gate": bool(
        enriched["evidence_date"].lt(enriched["query_date"]).all()
    ),
    "no_target_columns_read_gate": True,
    "no_api_calls_gate": True,
    "no_llm_calls_gate": True,
}

enriched.to_csv(
    OUTPUT_DIR / "role_aware_evidence_v2.csv", index=False
)
query_audit_v2.to_csv(
    OUTPUT_DIR / "role_aware_query_audit_v2.csv", index=False
)
manual_queries.to_csv(
    OUTPUT_DIR / "manual_review_queries_v2.csv", index=False
)
manual_evidence.to_csv(
    OUTPUT_DIR / "manual_review_evidence_v2.csv", index=False
)
party_summary.to_csv(
    OUTPUT_DIR / "role_aware_party_summary_v2.csv", index=False
)

manifest = {
    "notebook_version": "11b-role-aware-rag-audit-v2",
    "retrieval_rerun": False,
    "queries": int(len(query_audit_v2)),
    "evidence_rows": int(len(enriched)),
    "original_directional_rows": original_directional_rows,
    "role_aware_directional_rows": role_aware_directional_rows,
    "different_role_downgraded_rows": different_role_downgraded_rows,
    "manual_review_queries": int(len(manual_queries)),
    "structural_gates": structural_gates,
    "api_calls": 0,
    "llm_calls": 0,
}
(OUTPUT_DIR / "run_manifest_v2.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

display(party_summary.round(3))

status_counts = query_audit_v2["evidence_status_v2"].value_counts()
summary_lines = [
    "=== ROLE-AWARE RAG EVIDENCE AUDIT SUMMARY FOR REVIEW ===",
    "Notebook version: 11b-role-aware-rag-audit-v2",
    "API calls: 0",
    "LLM calls: 0",
    "Test labels read: False",
    "Retrieval rerun: False",
    f"Queries: {len(query_audit_v2)}",
    f"Evidence rows preserved: {len(enriched)}",
    f"Historical role metadata coverage: {historical_metadata_coverage:.3f}",
    f"Original directional evidence rows: {original_directional_rows}",
    f"Role-aware directional evidence rows: {role_aware_directional_rows}",
    f"Different-role directional rows downgraded: {different_role_downgraded_rows}",
    f"Directional retention rate: "
    f"{role_aware_directional_rows / max(original_directional_rows, 1):.3f}",
    f"Manual-review queries: {len(manual_queries)}",
    f"Manual-review evidence rows: {len(manual_evidence)}",
    f"Structural gates: {structural_gates}",
    "Role-aware evidence status:",
    status_counts.to_string(),
    "Party role-aware coverage:",
    party_summary.round(3).to_string(index=False),
    f"Output directory: {OUTPUT_DIR}",
    (
        "Next step: inspect the fixed 30-query sample; rebuild retrieval only "
        "if same-role directional coverage is too low."
    ),
    "=== END ROLE-AWARE RAG EVIDENCE AUDIT SUMMARY ===",
]

summary_text = "\n".join(summary_lines)
print(summary_text)
(OUTPUT_DIR / "role_aware_rag_audit_summary_v2.txt").write_text(
    summary_text, encoding="utf-8"
)
''')

md(r'''
## 如何阅读11B结果

阅读结果时，先看 `Different-role directional rows downgraded`。它统计原本看似反驳模型、但实际发生在不同政治角色下的历史投票。

`Directional retention rate` 表示加入制度环境后，还有多少历史材料能用于讨论支持或反对。`queries_with_challenging_history` 则统计角色修正后仍有同角色反向类比的查询，这些冲突需要 Agent 在回答中明确说明。

如果保留率很低，不代表RAG失败，而是说明现有语料缺乏与当前制度环境可比较的历史投票。此时应增加当前议会时期的投票和官方政策资料，而不是降低证据门槛。
''')


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print(OUTPUT)
