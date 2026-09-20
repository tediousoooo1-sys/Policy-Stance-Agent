import json
from pathlib import Path


ROOT = Path.cwd()
OUTPUT = ROOT / "11d_evidence_tiers_and_temporal_audit.ipynb"


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
# 11D — Evidence Tiers and Temporal Audit

## 这一步解决什么问题

11C已经能拒绝大量“文字相似、政策其实不同”的材料，但还发现三个问题：

1. 旧年份的同名Bill可能被误认成当前Bill；
2. `England`、`Northern Ireland`等地名可能被误认成政策关键词；
3. “要求政府发布报告”可能被误认成支持报告所讨论的政策本身。

11D不改变模型预测，也不让RAG重新预测。它只把材料分成三个产品层级：

```text
Direct evidence
    程序、政策对象和政党角色都可比，可用于说明历史上该党如何表态

Related policy material
    与政策主题相关，但不能证明本次投票会支持或反对

Similar historical vote
    政策领域或议会程序相似，只用于帮助用户探索类似投票
```

如果没有Direct evidence，产品仍然可以显示后两类材料，但必须明确告诉用户：

> No directly comparable evidence was found. The items below are related policy material or similar historical votes and do not prove the current prediction.
''')

md(r'''
## 为什么不直接放宽11C的门槛

放宽门槛会增加证据数量，但也会让错误证据重新出现。

这一版采用“分层”而不是“全部删掉”或“全部算证据”：

- 高精度材料进入Direct evidence；
- 有帮助但不能判断方向的材料进入Related或Similar；
- 明显无关的材料才会被删除。

这样既保持谨慎，也不会让产品在没有直接历史先例时完全没有内容。
''')

code(r'''
# 导入本地数据处理库；本Notebook不调用任何API或LLM
import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 220)
pd.set_option("display.width", 280)
pd.set_option("display.max_colwidth", 180)

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "processed"
V3_DIR = PROCESSED_DIR / "procedure_aware_rag_v3"
V2_DIR = PROCESSED_DIR / "role_aware_rag_audit_v2"
OUTPUT_DIR = PROCESSED_DIR / "evidence_tiers_v4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_PATH = V3_DIR / "procedure_aware_query_audit_v3.csv"
PACKET_PATH = V3_DIR / "procedure_aware_agent_packets_v3.csv"
V3_EVIDENCE_PATH = V3_DIR / "procedure_aware_evidence_v3.csv"
V2_EVIDENCE_PATH = V2_DIR / "role_aware_evidence_v2.csv"

MAX_DIRECT = 3
MAX_RELATED_POLICY = 2
MAX_SIMILAR_VOTES = 2
MAX_LEGISLATIVE_BACKGROUND = 1
MAX_TOTAL_EVIDENCE = 7
MAX_CURRENT_BILL_REFERENCE_AGE_DAYS = 1095
DIVISIONS_PER_QUARTER = 5
EXPECTED_QUARTERS = [
    "2025Q1", "2025Q2", "2025Q3",
    "2025Q4", "2026Q1", "2026Q2",
]

print("Notebook version: 11d-evidence-tiers-temporal-audit-v4")
print("API calls: 0")
print("LLM calls: 0")
print("Test labels read: False")
print("Prediction model changed: False")
print("Retrieval rerun: False")
print("Output directory:", OUTPUT_DIR)
''')

md(r'''
## 1. 读取11C和11B的已有产物

- 11C提供经过程序规则筛选的高精度候选；
- 11B只用于寻找可以展示为“相关政策”或“相似投票”的备选材料；
- 11B材料永远不能重新获得Direct evidence身份；
- 不读取任何`target_*`真实标签。
''')

code(r'''
queries = pd.read_csv(QUERY_PATH, low_memory=False)
packets = pd.read_csv(PACKET_PATH, low_memory=False)
v3_evidence = pd.read_csv(V3_EVIDENCE_PATH, low_memory=False)
v2_evidence = pd.read_csv(V2_EVIDENCE_PATH, low_memory=False)

queries["motion_date"] = pd.to_datetime(queries["motion_date"], errors="raise")
v3_evidence["query_date"] = pd.to_datetime(
    v3_evidence["query_date"], errors="raise"
)
v3_evidence["evidence_date"] = pd.to_datetime(
    v3_evidence["evidence_date"], errors="coerce"
)
v2_evidence["query_date"] = pd.to_datetime(
    v2_evidence["query_date"], errors="raise"
)
v2_evidence["evidence_date"] = pd.to_datetime(
    v2_evidence["evidence_date"], errors="coerce"
)

assert queries["query_id"].is_unique
assert packets["query_id"].is_unique
assert set(queries["query_id"]) == set(packets["query_id"])
assert not any(column.startswith("target_") for column in queries.columns)
assert not any(column.startswith("target_") for column in v3_evidence.columns)
assert not any(column.startswith("target_") for column in v2_evidence.columns)

print("Queries:", len(queries))
print("Unique divisions:", queries["division_key"].nunique())
print("11C evidence rows:", len(v3_evidence))
print("11B fallback candidate rows:", len(v2_evidence))
''')

md(r'''
## 2. 定义“实质政策词”和三项精度修正

这里不把所有共同单词都当成政策相似。

例如：

- `England`和`Northern Ireland`只是管辖地区；
- `Bill`和`Regulations`只是文件类型；
- `report`和`statement`通常代表监督动作，不代表同意被监督的政策。

因此系统会重点比较`deposit`、`containers`、`private school fees`等真正描述政策内容的词。
''')

code(r'''
WORD_PATTERN = re.compile(r"[a-z0-9]+")

# 这些词经常出现，但不能单独证明两个政策相同
GENERIC_POLICY_WORDS = set(ENGLISH_STOP_WORDS).union({
    "bill", "act", "order", "motion", "reading", "clause", "amendment",
    "question", "house", "commons", "approve", "proposed", "stage",
    "page", "line", "section", "part", "paragraph", "schedule",
    "regulation", "regulations", "draft", "lords", "new", "number",
    "england", "english", "wales", "welsh", "scotland", "scottish",
    "northern", "ireland", "irish", "united", "kingdom", "government",
    "national", "local", "public", "policy", "policies", "people",
    "provide", "ensure", "make", "work", "year", "years",
})

OVERSIGHT_PATTERN = re.compile(
    r"\b(report|statement|review|assessment|assess|consultation|consult|"
    r"monitor|evaluation|evaluate|publish|publication)\b",
    flags=re.IGNORECASE,
)


def safe_text(value):
    # 统一处理缺失值和多余空格
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_bool(value):
    # CSV中的布尔值可能以字符串、数字或布尔类型保存
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return safe_text(value).lower() in {"true", "1", "yes", "y"}


def normalize_text(value):
    # 只保留英文数字，便于稳定比较
    text = safe_text(value).lower().replace("__", " ")
    text = re.sub(r"[^a-z0-9£%\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def policy_tokens(value):
    # 删除程序词、地名和普通英语停用词，只保留政策实词
    return {
        token for token in WORD_PATTERN.findall(normalize_text(value))
        if len(token) > 2 and token not in GENERIC_POLICY_WORDS
    }


def token_metrics(left, right):
    # 返回共同实质词、较短文本覆盖率和Jaccard相似度
    left_tokens = policy_tokens(left)
    right_tokens = policy_tokens(right)
    if not left_tokens or not right_tokens:
        return set(), 0.0, 0.0
    shared = left_tokens & right_tokens
    containment = len(shared) / min(len(left_tokens), len(right_tokens))
    jaccard = len(shared) / len(left_tokens | right_tokens)
    return shared, containment, jaccard


def oversight_only_mismatch(query_text, evidence_text):
    # 历史材料只要求报告或审查，但当前对象不是监督动作时，不能判断政策方向
    query_is_oversight = bool(OVERSIGHT_PATTERN.search(safe_text(query_text)))
    evidence_is_oversight = bool(OVERSIGHT_PATTERN.search(safe_text(evidence_text)))
    return evidence_is_oversight and not query_is_oversight


def stable_hash(value):
    # 固定哈希保证重复运行时抽到相同样本
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
''')

md(r'''
## 3. 修正11C证据的身份

本节执行三项修正：

1. 超过三年的Bill reference不能声称是当前同一Bill；
2. Manifesto至少要共享真正的政策实词；
3. 监督、报告和评估类历史动议不能直接证明政策支持方向。

被降级不等于完全删除。仍然有帮助的材料会进入Related policy material。
''')

code(r'''
def classify_v3_row(row):
    # 返回产品层级、是否允许方向判断和具体原因
    source_type = safe_text(row.get("source_type"))
    role = safe_text(row.get("evidence_role_v3"))
    query_object = safe_text(row.get("query_policy_object"))
    evidence_text = " ".join([
        safe_text(row.get("evidence_title")),
        safe_text(row.get("historical_policy_object")),
        safe_text(row.get("evidence_text")),
    ])

    if source_type == "historical_vote":
        was_direct = role == "procedure_and_role_matched_historical_direction"
        if was_direct and oversight_only_mismatch(query_object, evidence_text):
            return (
                "related_policy_material",
                False,
                "oversight_motion_not_policy_direction",
            )
        if was_direct:
            return "direct_historical_evidence", True, "passed_11c_direct_gate"
        return (
            "related_policy_material",
            False,
            "policy_match_but_institutional_context_differs",
        )

    if source_type == "manifesto":
        shared, containment, jaccard = token_metrics(query_object, evidence_text)
        dense_score = float(row.get("dense_score", 0) or 0)
        relevant = bool(
            (len(shared) >= 2)
            or (len(shared) >= 1 and dense_score >= 0.36)
        )
        if relevant:
            return (
                "related_policy_material",
                False,
                "manifesto_substantive_policy_match",
            )
        return "reject", False, "manifesto_shared_only_generic_words"

    if source_type == "bill_reference":
        query_date = pd.Timestamp(row.get("query_date"))
        evidence_date = pd.to_datetime(row.get("evidence_date"), errors="coerce")
        age_days = (
            (query_date - evidence_date).days
            if pd.notna(evidence_date) else np.inf
        )
        if 0 <= age_days <= MAX_CURRENT_BILL_REFERENCE_AGE_DAYS:
            return (
                "legislative_background",
                False,
                "current_bill_reference_within_three_years",
            )
        return "reject", False, "old_same_title_bill_reference_rejected"

    return "reject", False, "unsupported_source_type"


v3_classified = v3_evidence.copy()
v3_classified[[
    "evidence_tier_v4", "direction_allowed_v4", "tier_reason_v4"
]] = v3_classified.apply(
    lambda row: pd.Series(classify_v3_row(row)), axis=1
)

v3_rejected = v3_classified[
    v3_classified["evidence_tier_v4"].eq("reject")
].copy()
v3_kept = v3_classified[
    ~v3_classified["evidence_tier_v4"].eq("reject")
].copy()

print("11C rows retained or downgraded:", len(v3_kept))
print("11C rows rejected after precision patch:", len(v3_rejected))
display(v3_classified["tier_reason_v4"].value_counts().rename("rows").to_frame())
''')

md(r'''
## 4. 从11B中寻找“相关政策”和“相似投票”

只有11C没有保留的历史材料才会进入这一层。

这里的历史投票必须：

- 发生在查询日期之前；
- 不是当前同一个division；
- 至少属于相同政策领域；
- 标题中共享足够的实质政策词；
- 通过最低相关性分数。

即使通过，它也只能显示为Related或Similar，永远不能成为Direct evidence。
''')

code(r'''
query_lookup = queries.set_index("query_id")
already_kept_chunks = set(v3_kept["evidence_chunk_id"].dropna().astype(str))


def classify_v2_fallback(row):
    # 11B材料只允许成为探索性材料，不能成为方向证据
    if safe_text(row.get("source_type")) != "historical_vote":
        return "reject", "fallback_only_uses_historical_votes", 0, 0.0

    query_id = row.get("query_id")
    if query_id not in query_lookup.index:
        return "reject", "query_not_found", 0, 0.0

    query = query_lookup.loc[query_id]
    query_date = pd.Timestamp(query["motion_date"])
    evidence_date = pd.to_datetime(row.get("evidence_date"), errors="coerce")
    if pd.isna(evidence_date) or evidence_date >= query_date:
        return "reject", "future_or_same_date", 0, 0.0
    if safe_text(row.get("evidence_division_key")) == safe_text(query["division_key"]):
        return "reject", "same_division", 0, 0.0

    same_domain = parse_bool(row.get("same_policy_domain"))
    same_family = parse_bool(row.get("same_motion_family"))
    adjusted_score = float(row.get("adjusted_score", 0) or 0)
    direct_policy_match = parse_bool(row.get("direct_policy_match"))
    shared, containment, jaccard = token_metrics(
        query["effective_policy_object"],
        row.get("evidence_title"),
    )
    substantive_match = bool(
        len(shared) >= 2
        and containment >= 0.25
        and adjusted_score >= 0.12
    )
    anchored_match = bool(
        direct_policy_match
        and len(shared) >= 1
        and adjusted_score >= 0.12
    )

    if not same_domain or not (substantive_match or anchored_match):
        return "reject", "fallback_policy_match_too_weak", len(shared), containment
    if same_family:
        return (
            "similar_historical_vote",
            "same_domain_and_motion_family_not_direct",
            len(shared),
            containment,
        )
    return (
        "related_policy_material",
        "same_policy_area_different_procedure",
        len(shared),
        containment,
    )


v2_fallback = v2_evidence[
    ~v2_evidence["evidence_chunk_id"].astype(str).isin(already_kept_chunks)
].copy()
v2_fallback[[
    "evidence_tier_v4", "tier_reason_v4",
    "substantive_shared_tokens_v4", "substantive_containment_v4",
]] = v2_fallback.apply(
    lambda row: pd.Series(classify_v2_fallback(row)), axis=1
)
v2_fallback = v2_fallback[
    ~v2_fallback["evidence_tier_v4"].eq("reject")
].copy()
v2_fallback["direction_allowed_v4"] = False

print("11B fallback rows retained as Related or Similar:", len(v2_fallback))
display(v2_fallback["evidence_tier_v4"].value_counts().rename("rows").to_frame())
''')

md(r'''
## 5. 合并并限制每个查询的展示数量

每个查询最多显示：

- 3条Direct evidence；
- 2条Related policy material；
- 2条Similar historical votes；
- 1条Legislative background；
- 总数不超过7条。

数量上限只是防止页面过长，不要求必须凑满。
''')

code(r'''
def convert_v3(row):
    # 把11C字段转换成统一的产品字段
    relationship = safe_text(row.get("relationship_to_prediction_v3"))
    if not parse_bool(row.get("direction_allowed_v4")):
        relationship = "context_only_not_prediction_evidence"
    return {
        "query_id": row.get("query_id"),
        "row_id": row.get("row_id"),
        "division_key": row.get("query_division_key"),
        "query_date": row.get("query_date"),
        "party": row.get("query_party"),
        "party_role": row.get("query_party_role"),
        "query_title": row.get("query_title"),
        "query_policy_object": row.get("query_policy_object"),
        "predicted_stance": row.get("predicted_stance"),
        "support_probability": row.get("support_probability"),
        "source_type": row.get("source_type"),
        "evidence_tier_v4": row.get("evidence_tier_v4"),
        "direction_allowed_v4": parse_bool(row.get("direction_allowed_v4")),
        "tier_reason_v4": row.get("tier_reason_v4"),
        "relationship_to_prediction_v4": relationship,
        "evidence_chunk_id": row.get("evidence_chunk_id"),
        "evidence_document_id": row.get("evidence_document_id"),
        "evidence_title": row.get("evidence_title"),
        "evidence_party": row.get("evidence_party"),
        "evidence_date": row.get("evidence_date"),
        "evidence_division_key": row.get("evidence_division_key"),
        "historical_party_role": row.get("historical_party_role"),
        "stance_label": row.get("stance_label"),
        "source_url": row.get("source_url"),
        "page_number": row.get("page_number"),
        "evidence_text": row.get("evidence_text"),
        "ranking_score_v4": float(row.get("hybrid_score", 0) or 0),
        "candidate_origin_v4": "11c_precision_patch",
    }


def convert_v2(row):
    # 把11B备选字段转换成统一的产品字段
    query = query_lookup.loc[row["query_id"]]
    return {
        "query_id": row.get("query_id"),
        "row_id": row.get("row_id"),
        "division_key": row.get("query_division_key"),
        "query_date": row.get("query_date"),
        "party": row.get("query_party"),
        "party_role": row.get("current_party_role"),
        "query_title": query.get("motion_title"),
        "query_policy_object": row.get("effective_policy_object"),
        "predicted_stance": row.get("predicted_stance"),
        "support_probability": row.get("support_probability"),
        "source_type": row.get("source_type"),
        "evidence_tier_v4": row.get("evidence_tier_v4"),
        "direction_allowed_v4": False,
        "tier_reason_v4": row.get("tier_reason_v4"),
        "relationship_to_prediction_v4": "context_only_not_prediction_evidence",
        "evidence_chunk_id": row.get("evidence_chunk_id"),
        "evidence_document_id": row.get("evidence_document_id"),
        "evidence_title": row.get("evidence_title"),
        "evidence_party": row.get("evidence_party"),
        "evidence_date": row.get("evidence_date"),
        "evidence_division_key": row.get("evidence_division_key"),
        "historical_party_role": row.get("historical_party_role"),
        "stance_label": row.get("stance_label"),
        "source_url": row.get("source_url"),
        "page_number": row.get("page_number"),
        "evidence_text": row.get("evidence_text"),
        "ranking_score_v4": float(row.get("adjusted_score", 0) or 0),
        "candidate_origin_v4": "11b_related_fallback",
    }


unified_rows = [convert_v3(row) for _, row in v3_kept.iterrows()]
unified_rows.extend(convert_v2(row) for _, row in v2_fallback.iterrows())
unified = pd.DataFrame(unified_rows)

TIER_ORDER = {
    "direct_historical_evidence": 0,
    "legislative_background": 1,
    "related_policy_material": 2,
    "similar_historical_vote": 3,
}
TIER_LIMITS = {
    "direct_historical_evidence": MAX_DIRECT,
    "legislative_background": MAX_LEGISLATIVE_BACKGROUND,
    "related_policy_material": MAX_RELATED_POLICY,
    "similar_historical_vote": MAX_SIMILAR_VOTES,
}


def select_for_query(group):
    # 先按层级，再按相关性选择；相似投票避免重复同一个历史division
    selected_parts = []
    used_chunks = set()
    used_historical_divisions = set()
    for tier in sorted(TIER_ORDER, key=TIER_ORDER.get):
        candidates = group[group["evidence_tier_v4"].eq(tier)].copy()
        candidates = candidates.sort_values(
            ["ranking_score_v4", "evidence_chunk_id"],
            ascending=[False, True],
        )
        accepted = []
        for _, row in candidates.iterrows():
            chunk_id = safe_text(row["evidence_chunk_id"])
            history_division = safe_text(row["evidence_division_key"])
            if chunk_id in used_chunks:
                continue
            if (
                tier == "similar_historical_vote"
                and history_division
                and history_division in used_historical_divisions
            ):
                continue
            accepted.append(row)
            used_chunks.add(chunk_id)
            if history_division:
                used_historical_divisions.add(history_division)
            if len(accepted) >= TIER_LIMITS[tier]:
                break
        if accepted:
            selected_parts.append(pd.DataFrame(accepted))
    if not selected_parts:
        return group.iloc[0:0].copy()
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["tier_order"] = selected["evidence_tier_v4"].map(TIER_ORDER)
    selected = selected.sort_values(
        ["tier_order", "ranking_score_v4"], ascending=[True, False]
    ).head(MAX_TOTAL_EVIDENCE)
    return selected.drop(columns="tier_order")


selected_groups = []
for query_id, group in unified.groupby("query_id", sort=False):
    selected_groups.append(select_for_query(group))

final_evidence = (
    pd.concat(selected_groups, ignore_index=True)
    if selected_groups else unified.iloc[0:0].copy()
)
final_evidence["rank_v4"] = final_evidence.groupby("query_id").cumcount() + 1
final_evidence["public_evidence_id"] = final_evidence["rank_v4"].map(
    lambda value: f"E{int(value)}"
)

print("Final evidence rows:", len(final_evidence))
display(final_evidence["evidence_tier_v4"].value_counts().rename("rows").to_frame())
''')

md(r'''
## 6. 构造产品可以直接使用的查询状态和提示语

RAG不修改冻结概率。它只决定产品应该显示哪一种说明：

- 有Direct evidence：显示直接可比证据；
- 没有Direct，但有相关材料：显示相关政策或相似投票，并加醒目限制说明；
- 什么都没有：仍然展示模型概率和模型特征解释，同时诚实说明没有找到可用外部材料。
''')

code(r'''
tier_counts = final_evidence.pivot_table(
    index="query_id",
    columns="evidence_tier_v4",
    values="evidence_chunk_id",
    aggfunc="count",
    fill_value=0,
)
for tier in TIER_ORDER:
    if tier not in tier_counts.columns:
        tier_counts[tier] = 0
tier_counts = tier_counts.reset_index()

query_audit = queries.merge(tier_counts, on="query_id", how="left")
for tier in TIER_ORDER:
    query_audit[tier] = query_audit[tier].fillna(0).astype(int)


def product_status(row):
    if row["direct_historical_evidence"] > 0:
        return "direct_evidence_available"
    if row["related_policy_material"] > 0 or row["legislative_background"] > 0:
        return "related_policy_material_only"
    if row["similar_historical_vote"] > 0:
        return "similar_votes_only"
    return "no_external_material"


def product_message(status):
    messages = {
        "direct_evidence_available": (
            "Directly comparable historical evidence is available. It may explain "
            "past party behaviour but does not replace the frozen model prediction."
        ),
        "related_policy_material_only": (
            "No directly comparable evidence was found. The material below concerns "
            "a related policy and does not prove the current prediction."
        ),
        "similar_votes_only": (
            "No directly comparable evidence was found. The votes below are similar "
            "historical examples for exploration only and cannot establish the party's "
            "stance on the current motion."
        ),
        "no_external_material": (
            "No reliable external material was found. The probability is based on the "
            "frozen prediction model and should be presented with its model drivers only."
        ),
    }
    return messages[status]


query_audit["product_evidence_status_v4"] = query_audit.apply(
    product_status, axis=1
)
query_audit["product_evidence_message_v4"] = query_audit[
    "product_evidence_status_v4"
].map(product_message)
query_audit["rag_changes_probability"] = False
query_audit["rag_produces_separate_prediction_v4"] = False
query_audit["prediction_overridden_v4"] = False


def join_ids(group, tier):
    rows = group[group["evidence_tier_v4"].eq(tier)]
    return " | ".join(rows["public_evidence_id"].astype(str)) or pd.NA


id_rows = []
for query_id, group in final_evidence.groupby("query_id"):
    id_rows.append({
        "query_id": query_id,
        "direct_evidence_ids_v4": join_ids(group, "direct_historical_evidence"),
        "related_policy_ids_v4": join_ids(group, "related_policy_material"),
        "similar_vote_ids_v4": join_ids(group, "similar_historical_vote"),
        "legislative_background_ids_v4": join_ids(group, "legislative_background"),
        "all_evidence_ids_v4": " | ".join(group["public_evidence_id"].astype(str)),
    })
id_frame = pd.DataFrame(id_rows)

agent_packets = packets.drop(columns=[
    column for column in packets.columns
    if column.endswith("_v4")
], errors="ignore").merge(
    query_audit[[
        "query_id", "product_evidence_status_v4",
        "product_evidence_message_v4", "rag_changes_probability",
        "rag_produces_separate_prediction_v4", "prediction_overridden_v4",
    ]],
    on="query_id",
    how="left",
).merge(id_frame, on="query_id", how="left")

agent_packets["agent_instruction_v4"] = (
    "Explain the frozen probability and model drivers first. Direct evidence may be "
    "described as a comparable historical party position. Related policy material, "
    "legislative background and similar votes must be shown in separate sections and "
    "must not be used to change the probability or claim the current stance. If no "
    "direct evidence exists, say so explicitly. Never vote by evidence count."
)

display(query_audit["product_evidence_status_v4"].value_counts().rename("queries").to_frame())
''')

md(r'''
## 7. 建立覆盖完整时间范围的固定人工样本

11C的30个样本全部落在2025年上半年。11D改成：

- 2025 Q1、Q2、Q3、Q4各5个division；
- 2026 Q1、Q2各5个division；
- 总共30个division；
- 每个季度尽量同时包含有Direct、只有Related/Similar和完全无材料的议案。

这个样本用于检查RAG材料质量，不使用真实投票标签。
''')

code(r'''
query_audit["time_quarter"] = query_audit["motion_date"].dt.to_period("Q").astype(str)

STATUS_ORDER = {
    "direct_evidence_available": 0,
    "related_policy_material_only": 1,
    "similar_votes_only": 2,
    "no_external_material": 3,
}

division_pool = query_audit.groupby("division_key", as_index=False).agg(
    motion_date=("motion_date", "first"),
    motion_title=("motion_title", "first"),
    policy_domain_primary=("policy_domain_primary", "first"),
    motion_family=("motion_family", "first"),
    time_quarter=("time_quarter", "first"),
    party_queries=("query_id", "size"),
    direct_queries=(
        "product_evidence_status_v4",
        lambda values: int((values == "direct_evidence_available").sum()),
    ),
    related_queries=(
        "product_evidence_status_v4",
        lambda values: int((values == "related_policy_material_only").sum()),
    ),
    similar_queries=(
        "product_evidence_status_v4",
        lambda values: int((values == "similar_votes_only").sum()),
    ),
)


def division_status(row):
    if row["direct_queries"] > 0:
        return "direct_evidence_available"
    if row["related_queries"] > 0:
        return "related_policy_material_only"
    if row["similar_queries"] > 0:
        return "similar_votes_only"
    return "no_external_material"


division_pool["division_product_status"] = division_pool.apply(
    division_status, axis=1
)
division_pool["stable_hash"] = division_pool["division_key"].map(stable_hash)


def choose_quarter_sample(group):
    # 先从每种状态各取一条，再按固定哈希补足到5条
    selected_keys = []
    for status in STATUS_ORDER:
        candidates = group[
            group["division_product_status"].eq(status)
        ].sort_values("stable_hash")
        if len(candidates):
            selected_keys.append(candidates.iloc[0]["division_key"])
    remaining = group[
        ~group["division_key"].isin(selected_keys)
    ].sort_values("stable_hash")
    needed = DIVISIONS_PER_QUARTER - len(selected_keys)
    if needed > 0:
        selected_keys.extend(
            remaining.head(needed)["division_key"].tolist()
        )
    return group[group["division_key"].isin(selected_keys)].copy()


sample_parts = []
for quarter in EXPECTED_QUARTERS:
    quarter_group = division_pool[
        division_pool["time_quarter"].eq(quarter)
    ]
    if len(quarter_group) < DIVISIONS_PER_QUARTER:
        raise ValueError(f"{quarter}可用division不足5个。")
    sample_parts.append(choose_quarter_sample(quarter_group))

manual_divisions = pd.concat(sample_parts, ignore_index=True).sort_values(
    ["time_quarter", "division_product_status", "stable_hash"]
)
manual_queries = query_audit[
    query_audit["division_key"].isin(manual_divisions["division_key"])
].copy()
manual_evidence = final_evidence[
    final_evidence["division_key"].isin(manual_divisions["division_key"])
].copy()

# 空白列留给人工审核；不会自动假装已经完成语义审核
manual_evidence["manual_policy_relevance"] = pd.NA
manual_evidence["manual_tier_correct"] = pd.NA
manual_evidence["manual_direction_valid"] = pd.NA
manual_evidence["manual_notes"] = pd.NA
manual_queries["manual_gap_acceptable"] = pd.NA
manual_queries["manual_missing_source_needed"] = pd.NA
manual_queries["manual_notes"] = pd.NA

quarter_sample_counts = manual_divisions.groupby("time_quarter").agg(
    divisions=("division_key", "nunique"),
    direct_divisions=(
        "division_product_status",
        lambda values: int((values == "direct_evidence_available").sum()),
    ),
    related_divisions=(
        "division_product_status",
        lambda values: int((values == "related_policy_material_only").sum()),
    ),
    similar_divisions=(
        "division_product_status",
        lambda values: int((values == "similar_votes_only").sum()),
    ),
    no_material_divisions=(
        "division_product_status",
        lambda values: int((values == "no_external_material").sum()),
    ),
).reset_index()

display(quarter_sample_counts)
print("Manual-review divisions:", manual_divisions["division_key"].nunique())
print("Manual-review party queries:", len(manual_queries))
print("Manual-review evidence rows:", len(manual_evidence))
''')

md(r'''
## 8. 结构检查

这些检查回答的是“流程有没有偷看答案、证据有没有越权”。

它们不能替代人工语义检查。人工仍要确认：

- Direct evidence是否真的可以判断方向；
- Related材料是否真的与政策有关；
- Similar vote是否足够相似而不会误导用户；
- 没有材料的查询是否需要补充新的官方语料。
''')

code(r'''
direct_rows = final_evidence[
    final_evidence["evidence_tier_v4"].eq("direct_historical_evidence")
]
bill_rows = final_evidence[
    final_evidence["evidence_tier_v4"].eq("legislative_background")
].copy()
if len(bill_rows):
    bill_rows["bill_age_days"] = (
        pd.to_datetime(bill_rows["query_date"])
        - pd.to_datetime(bill_rows["evidence_date"])
    ).dt.days

STRUCTURAL_GATES = {
    "all_queries_preserved_gate": set(query_audit["query_id"]) == set(queries["query_id"]),
    "maximum_seven_results_gate": bool(
        final_evidence.groupby("query_id").size().max() <= MAX_TOTAL_EVIDENCE
    ),
    "direct_only_historical_vote_gate": bool(
        direct_rows["source_type"].eq("historical_vote").all()
    ),
    "oversight_not_direct_gate": bool(
        ~direct_rows.apply(
            lambda row: oversight_only_mismatch(
                row["query_policy_object"],
                " ".join([
                    safe_text(row["evidence_title"]),
                    safe_text(row["evidence_text"]),
                ]),
            ),
            axis=1,
        ).any()
    ),
    "old_bill_reference_not_current_gate": bool(
        len(bill_rows) == 0
        or bill_rows["bill_age_days"].between(
            0, MAX_CURRENT_BILL_REFERENCE_AGE_DAYS
        ).all()
    ),
    "related_material_never_directional_gate": bool(
        final_evidence.loc[
            ~final_evidence["evidence_tier_v4"].eq("direct_historical_evidence"),
            "direction_allowed_v4",
        ].eq(False).all()
    ),
    "strict_temporal_gate": bool(
        (
            final_evidence["evidence_date"].isna()
            | (
                pd.to_datetime(final_evidence["evidence_date"])
                < pd.to_datetime(final_evidence["query_date"])
            )
        ).all()
    ),
    "same_division_gate": bool(
        final_evidence["evidence_division_key"].fillna("").astype(str).ne(
            final_evidence["division_key"].fillna("").astype(str)
        ).all()
    ),
    "thirty_division_sample_gate": manual_divisions["division_key"].nunique() == 30,
    "five_divisions_per_quarter_gate": bool(
        quarter_sample_counts["divisions"].eq(DIVISIONS_PER_QUARTER).all()
    ),
    "all_six_quarters_gate": set(quarter_sample_counts["time_quarter"]) == set(EXPECTED_QUARTERS),
    "prediction_not_overridden_gate": bool(
        query_audit["prediction_overridden_v4"].eq(False).all()
    ),
    "rag_not_second_predictor_gate": bool(
        query_audit["rag_produces_separate_prediction_v4"].eq(False).all()
    ),
    "no_target_columns_read_gate": not any(
        column.startswith("target_")
        for frame in [queries, v3_evidence, v2_evidence]
        for column in frame.columns
    ),
    "no_api_calls_gate": True,
    "no_llm_calls_gate": True,
}

display(pd.Series(STRUCTURAL_GATES, name="passed").to_frame())
if not all(bool(value) for value in STRUCTURAL_GATES.values()):
    raise ValueError("结构检查未全部通过，请先修复后再使用输出。")
''')

md(r'''
## 9. 保存输出

最重要的三个文件是：

- `agent_evidence_packets_v4.csv`：产品每个预测可直接使用的证据状态和提示语；
- `manual_review_evidence_v4.csv`：30个跨季度样本中的证据；
- `manual_review_queries_v4.csv`：包括没有证据的样本，用于判断是否需要补语料。
''')

code(r'''
def json_safe(value):
    # 将Numpy和Pandas类型转换成JSON可以保存的原生类型
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


final_evidence.to_csv(OUTPUT_DIR / "evidence_tiers_v4.csv", index=False)
query_audit.to_csv(OUTPUT_DIR / "query_evidence_status_v4.csv", index=False)
agent_packets.to_csv(OUTPUT_DIR / "agent_evidence_packets_v4.csv", index=False)
v3_rejected.to_csv(OUTPUT_DIR / "precision_patch_rejections_v4.csv", index=False)
manual_divisions.to_csv(OUTPUT_DIR / "manual_review_divisions_v4.csv", index=False)
manual_queries.to_csv(OUTPUT_DIR / "manual_review_queries_v4.csv", index=False)
manual_evidence.to_csv(OUTPUT_DIR / "manual_review_evidence_v4.csv", index=False)
quarter_sample_counts.to_csv(OUTPUT_DIR / "manual_review_quarter_coverage_v4.csv", index=False)

status_counts = query_audit["product_evidence_status_v4"].value_counts()
tier_counts_output = final_evidence["evidence_tier_v4"].value_counts()
party_summary = query_audit.groupby("party").agg(
    queries=("query_id", "size"),
    direct_evidence_rate=(
        "product_evidence_status_v4",
        lambda values: (values == "direct_evidence_available").mean(),
    ),
    related_only_rate=(
        "product_evidence_status_v4",
        lambda values: (values == "related_policy_material_only").mean(),
    ),
    similar_only_rate=(
        "product_evidence_status_v4",
        lambda values: (values == "similar_votes_only").mean(),
    ),
    no_external_material_rate=(
        "product_evidence_status_v4",
        lambda values: (values == "no_external_material").mean(),
    ),
).reset_index()
party_summary.to_csv(OUTPUT_DIR / "party_evidence_coverage_v4.csv", index=False)

manifest = {
    "notebook_version": "11d-evidence-tiers-temporal-audit-v4",
    "api_calls": 0,
    "llm_calls": 0,
    "test_labels_read": False,
    "prediction_model_changed": False,
    "retrieval_rerun": False,
    "queries": len(query_audit),
    "unique_divisions": int(query_audit["division_key"].nunique()),
    "final_evidence_rows": len(final_evidence),
    "evidence_tier_counts": tier_counts_output.to_dict(),
    "query_status_counts": status_counts.to_dict(),
    "manual_review_divisions": int(manual_divisions["division_key"].nunique()),
    "manual_review_quarters": EXPECTED_QUARTERS,
    "structural_gates": STRUCTURAL_GATES,
    "ready_for_llm_explanation": False,
}
with (OUTPUT_DIR / "run_manifest_v4.json").open("w", encoding="utf-8") as handle:
    json.dump(json_safe(manifest), handle, ensure_ascii=False, indent=2)

summary_lines = [
    "=== EVIDENCE TIERS AND TEMPORAL AUDIT SUMMARY FOR REVIEW ===",
    "Notebook version: 11d-evidence-tiers-temporal-audit-v4",
    "API calls: 0",
    "LLM calls: 0",
    "Test labels read: False",
    "Prediction model changed: False",
    "Retrieval rerun: False",
    f"Queries: {len(query_audit)}",
    f"Unique divisions: {query_audit['division_key'].nunique()}",
    f"Final evidence rows: {len(final_evidence)}",
    f"Queries with direct evidence rate: {(query_audit['product_evidence_status_v4'] == 'direct_evidence_available').mean():.3f}",
    f"Queries with related-policy-only rate: {(query_audit['product_evidence_status_v4'] == 'related_policy_material_only').mean():.3f}",
    f"Queries with similar-votes-only rate: {(query_audit['product_evidence_status_v4'] == 'similar_votes_only').mean():.3f}",
    f"Queries with no external material rate: {(query_audit['product_evidence_status_v4'] == 'no_external_material').mean():.3f}",
    f"Manual-review divisions: {manual_divisions['division_key'].nunique()}",
    f"Manual-review party queries: {len(manual_queries)}",
    f"Manual-review evidence rows: {len(manual_evidence)}",
    f"Manual-review quarters: {EXPECTED_QUARTERS}",
    f"Structural gates: {json_safe(STRUCTURAL_GATES)}",
    "Ready for LLM explanation: False",
    "Next step: Codex reviews the temporally balanced manual sample before any LLM explanation calls.",
    f"Output directory: {OUTPUT_DIR}",
    "=== END EVIDENCE TIERS AND TEMPORAL AUDIT SUMMARY ===",
]
summary_text = "\n".join(summary_lines)
(OUTPUT_DIR / "evidence_tiers_temporal_audit_summary_v4.txt").write_text(
    summary_text, encoding="utf-8"
)
print(summary_text)
print("\nEvidence tier counts:")
display(tier_counts_output.rename("rows").to_frame())
print("\nQuery product status:")
display(status_counts.rename("queries").to_frame())
print("\nParty evidence coverage:")
display(party_summary)
print("\nManual-review quarter coverage:")
display(quarter_sample_counts)
''')

md(r'''
## 如何理解11D的输出

### Direct evidence很少，不一定代表系统失败

某个新Clause可能从未在历史上出现过。此时系统不应该假装有直接先例。

### Related和Similar不是预测依据

它们的作用是帮助用户回答：

- 这个议案大致属于什么政策问题？
- 过去有哪些相近议题或议会程序？
- 如果我想继续研究，应该从哪些历史材料开始？

它们不能回答“因此该党这次一定支持或反对”。冻结概率仍然只来自预测模型。

### 下一步判断标准

先人工查看跨六个季度的固定样本：

- Direct evidence目标精度至少90%；
- Related和Similar材料目标相关率至少80%；
- 明显无关内容必须被删除；
- 如果精度合格但覆盖率仍低，再根据`manual_missing_source_needed`补充官方语料。
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
print(f"Wrote {OUTPUT}")
