import json
from pathlib import Path


ROOT = Path.cwd()
OUTPUT = ROOT / "11e_direct_evidence_firewall_and_review.ipynb"


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
# 11E — Direct Evidence Firewall and Full Review

## 这一步解决什么问题

11D证明“证据分层”方向正确，但Direct evidence仍有一个严重问题：

```text
2026年的Finance (No. 2) Bill
被匹配到2016、2018和2021年的同名Finance Bill
```

名称和投票阶段相同，不代表它们是同一部法案。每年的Finance Bill、Budget、Supply Bill等都会重复出现。

11E增加四道防火墙：

1. **重复性Bill防火墙**：Finance等年度法案必须属于同一合理年度周期；
2. **议会届次防火墙**：大选前已经终止的Bill不能冒充大选后的同一Bill；
3. **Clause和Amendment防火墙**：编号相同只有在同一Bill中才有意义；
4. **实质政策防火墙**：两个“要求发布报告”的动议还必须讨论同一具体政策。

这一步不改变预测模型、不重新检索、不调用API，也不读取Test真实标签。
''')

md(r'''
## Direct、Related和Similar的通俗区别

- **Direct evidence**：历史投票可以直接说明该党过去如何对待同一个政策对象；
- **Related policy material**：政策主题相关，但不能直接推出本次立场；
- **Similar historical vote**：投票形式或大政策领域相似，适合用户继续探索；
- **Reject**：相似主要来自编号、程序词或同名法案，展示会造成误导。

被Direct防火墙挡下的材料不会自动删除。系统会根据剩余信息把它降为Related、Similar或Reject。
''')

code(r'''
# 导入本地数据处理库；不调用任何API或LLM
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
pd.set_option("display.max_columns", 240)
pd.set_option("display.width", 300)
pd.set_option("display.max_colwidth", 180)

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "processed"
V3_DIR = PROCESSED_DIR / "procedure_aware_rag_v3"
V4_DIR = PROCESSED_DIR / "evidence_tiers_v4"
OUTPUT_DIR = PROCESSED_DIR / "evidence_precision_v5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_PATH = V4_DIR / "query_evidence_status_v4.csv"
PACKET_PATH = V4_DIR / "agent_evidence_packets_v4.csv"
EVIDENCE_PATH = V4_DIR / "evidence_tiers_v4.csv"
V3_METADATA_PATH = V3_DIR / "procedure_aware_evidence_v3.csv"

ELECTION_TRANSITION_DATE = pd.Timestamp("2024-07-05")
MAX_ORDINARY_WHOLE_BILL_AGE_DAYS = 730
MAX_RECURRING_BILL_AGE_DAYS = 400
MAX_DIRECT = 3
MAX_RELATED_POLICY = 2
MAX_SIMILAR_VOTES = 2
MAX_LEGISLATIVE_BACKGROUND = 1
MAX_TOTAL_EVIDENCE = 7
DIVISIONS_PER_QUARTER = 5
EXPECTED_QUARTERS = [
    "2025Q1", "2025Q2", "2025Q3",
    "2025Q4", "2026Q1", "2026Q2",
]

print("Notebook version: 11e-direct-evidence-firewall-v5")
print("API calls: 0")
print("LLM calls: 0")
print("Test labels read: False")
print("Prediction model changed: False")
print("Retrieval rerun: False")
print("Output directory:", OUTPUT_DIR)
''')

md(r'''
## 1. 读取11D证据和11C程序元数据

11D提供最终展示层级。11C补充历史Bill名称、程序判断和政策对象字段。

这里只读取预测、查询和证据，不读取任何`target_*`列。
''')

code(r'''
queries = pd.read_csv(QUERY_PATH, low_memory=False)
packets = pd.read_csv(PACKET_PATH, low_memory=False)
evidence = pd.read_csv(EVIDENCE_PATH, low_memory=False)
v3_metadata = pd.read_csv(V3_METADATA_PATH, low_memory=False)

queries["motion_date"] = pd.to_datetime(queries["motion_date"], errors="raise")
evidence["query_date"] = pd.to_datetime(evidence["query_date"], errors="raise")
evidence["evidence_date"] = pd.to_datetime(
    evidence["evidence_date"], errors="coerce"
)
v3_metadata["query_date"] = pd.to_datetime(
    v3_metadata["query_date"], errors="raise"
)
v3_metadata["evidence_date"] = pd.to_datetime(
    v3_metadata["evidence_date"], errors="coerce"
)

metadata_columns = [
    "query_id", "evidence_chunk_id", "historical_policy_object",
    "historical_legislation_name", "procedure_reason", "same_bill",
    "shared_policy_tokens", "policy_containment", "policy_jaccard",
]
metadata = v3_metadata[metadata_columns].drop_duplicates(
    ["query_id", "evidence_chunk_id"]
)
evidence = evidence.merge(
    metadata,
    on=["query_id", "evidence_chunk_id"],
    how="left",
)

assert queries["query_id"].is_unique
assert packets["query_id"].is_unique
assert set(queries["query_id"]) == set(packets["query_id"])
assert not any(column.startswith("target_") for column in queries.columns)
assert not any(column.startswith("target_") for column in evidence.columns)
assert not any(column.startswith("target_") for column in v3_metadata.columns)

print("Queries:", len(queries))
print("Unique divisions:", queries["division_key"].nunique())
print("11D evidence rows:", len(evidence))
print("11D direct rows:", int(evidence["evidence_tier_v4"].eq("direct_historical_evidence").sum()))
''')

md(r'''
## 2. 建立法案、条款和政策内容的比较工具

这一节把文本转换成几个简单问题：

- 是整部Bill、具体Clause、Amendment还是其他政策？
- 是否属于每年重复出现的法案？
- 两边是否位于同一次大选之后的议会环境？
- Clause或Amendment编号是否一致？
- 除去`review`、`report`等程序词后，还剩多少共同政策实词？
''')

code(r'''
WORD_PATTERN = re.compile(r"[a-z0-9]+")
YEAR_PATTERN = re.compile(r"\b(20\d{2})(?:[-–](\d{2}|20\d{2}))?\b")
CLAUSE_PATTERN = re.compile(
    r"(?:new\s+)?clause(?:\s+no\.?|\s+number)?\s*(\d+[a-z]?)",
    flags=re.IGNORECASE,
)
AMENDMENT_PATTERN = re.compile(
    r"(?:lords?\s+)?amendment(?:\s+no\.?|\s+number)?\s*(\d+[a-z]?)",
    flags=re.IGNORECASE,
)
RECURRING_BILL_PATTERN = re.compile(
    r"\b(finance(?:\s*\(no\.\s*\d+\))?\s+bill|budget|supply bill|"
    r"appropriation bill|national insurance contributions bill)\b",
    flags=re.IGNORECASE,
)
OVERSIGHT_PATTERN = re.compile(
    r"\b(report|statement|review|assessment|assess|consultation|consult|"
    r"monitor|evaluation|evaluate|publish|publication)\b",
    flags=re.IGNORECASE,
)

GENERIC_POLICY_WORDS = set(ENGLISH_STOP_WORDS).union({
    "bill", "act", "order", "motion", "reading", "clause", "amendment",
    "question", "house", "commons", "approve", "proposed", "stage",
    "page", "line", "section", "sections", "part", "paragraph", "schedule",
    "regulation", "regulations", "draft", "lords", "new", "number",
    "england", "english", "wales", "welsh", "scotland", "scottish",
    "northern", "ireland", "irish", "united", "kingdom", "government",
    "national", "local", "public", "policy", "policies", "people",
    "provide", "ensure", "make", "work", "year", "years", "disagree",
})

# 比较监督动议时，再删除这些固定程序表达
OVERSIGHT_GENERIC_WORDS = GENERIC_POLICY_WORDS.union({
    "report", "statement", "review", "assessment", "assess", "publish",
    "publication", "impact", "measure", "measures", "effect", "effects",
    "chancellor", "exchequer", "secretary", "state", "months", "month",
    "passing", "passed", "contained", "within", "thereafter",
})

BILL_NAME_STOP_WORDS = GENERIC_POLICY_WORDS.union({
    "second", "third", "read", "time", "reasoned", "programme",
    "commencement", "remaining", "no",
})


def safe_text(value):
    # 统一处理缺失值和空格
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_bool(value):
    # CSV中的布尔值可能保存为字符串或数字
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return safe_text(value).lower() in {"true", "1", "yes", "y"}


def normalize_text(value):
    # 删除标点并转为小写，方便稳定比较
    text = safe_text(value).lower().replace("__", " ")
    text = re.sub(r"[^a-z0-9£%\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(value, stop_words=GENERIC_POLICY_WORDS):
    # 提取政策实词
    return {
        token for token in WORD_PATTERN.findall(normalize_text(value))
        if len(token) > 2 and token not in stop_words
    }


def token_metrics(left, right, stop_words=GENERIC_POLICY_WORDS):
    # 返回共同词、短文本覆盖率和Jaccard相似度
    left_tokens = tokens(left, stop_words)
    right_tokens = tokens(right, stop_words)
    if not left_tokens or not right_tokens:
        return set(), 0.0, 0.0
    shared = left_tokens & right_tokens
    containment = len(shared) / min(len(left_tokens), len(right_tokens))
    jaccard = len(shared) / len(left_tokens | right_tokens)
    return shared, containment, jaccard


def scope_level(title):
    # 将查询分成具体条款、整部Bill、法定文书和一般政策
    lowered = normalize_text(title)
    if re.search(r"\b((?:new\s+)?clause|amendment)\b", lowered):
        return "specific_provision"
    if re.search(r"\b(second reading|third reading)\b", lowered):
        return "whole_bill"
    if re.search(r"\b(regulations?|statutory instrument|order)\b", lowered):
        return "instrument"
    return "general_policy"


def extract_identifiers(value):
    # 提取Clause和Amendment编号
    text = safe_text(value)
    return {
        "clauses": {item.lower() for item in CLAUSE_PATTERN.findall(text)},
        "amendments": {item.lower() for item in AMENDMENT_PATTERN.findall(text)},
    }


def extract_years(value):
    # 同时识别2028和2028-29一类年份
    result = set()
    for first, second in YEAR_PATTERN.findall(safe_text(value)):
        result.add(first)
        if second:
            if len(second) == 2:
                result.add(first[:2] + second)
            else:
                result.add(second)
    return result


def bill_signature(title, legislation=""):
    # 优先使用结构化法案名，否则使用标题中的名称实词
    value = safe_text(legislation) or safe_text(title)
    return tokens(value, BILL_NAME_STOP_WORDS)


def signature_similarity(left, right):
    # 比较两个Bill名称实词集合
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def same_election_era(query_date, evidence_date):
    # 大选前终止的Bill不能当作大选后当前同一Bill
    return bool(
        (query_date >= ELECTION_TRANSITION_DATE)
        == (evidence_date >= ELECTION_TRANSITION_DATE)
    )


def stable_hash(value):
    # 固定哈希保证重复运行抽到相同样本
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
''')

md(r'''
## 3. Direct evidence防火墙

Direct evidence必须通过以下判断：

### 整部Bill

- Bill名称实质词高度一致；
- 位于同一届议会环境；
- 普通Bill相隔不超过两年；
- Finance等重复性法案相隔不超过400天。

### 具体Clause或Amendment

- 查询必须包含足够的实质政策内容；
- Clause或Amendment编号冲突时不能直接比较；
- 不同税务年度不能冒充同一个条款；
- 如果都是报告或审查动议，删除程序套话后仍要共享具体政策词。
''')

code(r'''
def direct_firewall(row):
    # 返回新层级、是否允许方向判断和审核原因
    query_date = pd.Timestamp(row["query_date"])
    evidence_date = pd.to_datetime(row["evidence_date"], errors="coerce")
    if pd.isna(evidence_date) or evidence_date >= query_date:
        return "reject", False, "invalid_evidence_date"

    query_title = safe_text(row["query_title"])
    query_object = safe_text(row["query_policy_object"])
    evidence_title = safe_text(row["evidence_title"])
    evidence_object = safe_text(row.get("historical_policy_object"))
    if not evidence_object:
        evidence_object = " ".join([
            evidence_title,
            safe_text(row["evidence_text"]),
        ])
    evidence_legislation = safe_text(row.get("historical_legislation_name"))
    query_scope = scope_level(query_title)
    evidence_scope = scope_level(evidence_title)
    age_days = int((query_date - evidence_date).days)

    query_is_recurring = bool(RECURRING_BILL_PATTERN.search(
        " ".join([query_title, query_object])
    ))
    evidence_is_recurring = bool(RECURRING_BILL_PATTERN.search(
        " ".join([evidence_title, evidence_legislation])
    ))

    if query_scope == "whole_bill":
        query_signature = bill_signature(query_title)
        evidence_signature = bill_signature(evidence_title, evidence_legislation)
        name_similarity = signature_similarity(query_signature, evidence_signature)
        if name_similarity < 0.80:
            return "reject", False, "whole_bill_name_not_same"
        if not same_election_era(query_date, evidence_date):
            return (
                "related_policy_material",
                False,
                "same_title_bill_from_different_parliament",
            )
        if query_is_recurring or evidence_is_recurring:
            if age_days > MAX_RECURRING_BILL_AGE_DAYS:
                return (
                    "similar_historical_vote",
                    False,
                    "recurring_bill_from_different_annual_cycle",
                )
        elif age_days > MAX_ORDINARY_WHOLE_BILL_AGE_DAYS:
            return (
                "related_policy_material",
                False,
                "same_title_bill_too_old_for_current_bill",
            )
        return "direct_historical_evidence", True, "whole_bill_same_session"

    if query_scope == "specific_provision":
        query_ids = extract_identifiers(" ".join([query_title, query_object]))
        evidence_ids = extract_identifiers(" ".join([evidence_title, evidence_object]))
        query_years = extract_years(query_object)
        evidence_years = extract_years(evidence_object)

        query_oversight = bool(OVERSIGHT_PATTERN.search(query_object))
        evidence_oversight = bool(OVERSIGHT_PATTERN.search(evidence_object))
        comparison_stop_words = (
            OVERSIGHT_GENERIC_WORDS
            if query_oversight or evidence_oversight
            else GENERIC_POLICY_WORDS
        )
        shared, containment, jaccard = token_metrics(
            query_object, evidence_object, comparison_stop_words
        )

        if len(tokens(query_object, comparison_stop_words)) < 2:
            return "reject", False, "query_policy_object_too_thin"
        if query_ids["clauses"] and evidence_ids["clauses"] and (
            query_ids["clauses"] != evidence_ids["clauses"]
        ):
            if len(shared) >= 2:
                return (
                    "related_policy_material",
                    False,
                    "different_clause_number_same_policy_area",
                )
            return "reject", False, "different_clause_number"
        if query_ids["amendments"] and evidence_ids["amendments"] and (
            query_ids["amendments"] != evidence_ids["amendments"]
        ):
            return "reject", False, "different_amendment_number"
        if query_years and evidence_years and not (query_years & evidence_years):
            if len(shared) >= 2:
                return (
                    "related_policy_material",
                    False,
                    "same_policy_different_tax_years",
                )
            return "reject", False, "different_policy_years"
        if query_oversight != evidence_oversight:
            return (
                "related_policy_material",
                False,
                "oversight_and_substantive_policy_not_equivalent",
            )
        if len(shared) < 3 or containment < 0.65:
            if len(shared) >= 2:
                return (
                    "related_policy_material",
                    False,
                    "specific_policy_match_not_exact_enough",
                )
            return "reject", False, "specific_policy_match_too_weak"
        if query_is_recurring and age_days > MAX_RECURRING_BILL_AGE_DAYS:
            return (
                "related_policy_material",
                False,
                "recurring_bill_provision_from_old_cycle",
            )
        return "direct_historical_evidence", True, "exact_policy_provision_match"

    # 非Bill阶段或具体条款的历史方向证据使用保守门槛
    shared, containment, jaccard = token_metrics(query_object, evidence_object)
    if len(shared) >= 3 and containment >= 0.65 and age_days <= 730:
        return "direct_historical_evidence", True, "exact_general_policy_match"
    if len(shared) >= 2:
        return "related_policy_material", False, "general_policy_related_not_exact"
    return "reject", False, "general_policy_match_too_weak"


evidence_v5 = evidence.copy()
original_direct_mask = evidence_v5["evidence_tier_v4"].eq(
    "direct_historical_evidence"
)
direct_decisions = evidence_v5.loc[original_direct_mask].apply(
    lambda row: pd.Series(direct_firewall(row)), axis=1
)
direct_decisions.columns = [
    "evidence_tier_v5", "direction_allowed_v5", "tier_reason_v5"
]
evidence_v5.loc[original_direct_mask, direct_decisions.columns] = (
    direct_decisions.to_numpy()
)

# 非Direct材料先继承11D层级，下一节再做相关性精度检查
non_direct_mask = ~original_direct_mask
evidence_v5.loc[non_direct_mask, "evidence_tier_v5"] = evidence_v5.loc[
    non_direct_mask, "evidence_tier_v4"
]
evidence_v5.loc[non_direct_mask, "direction_allowed_v5"] = False
evidence_v5.loc[non_direct_mask, "tier_reason_v5"] = evidence_v5.loc[
    non_direct_mask, "tier_reason_v4"
]
evidence_v5["direction_allowed_v5"] = evidence_v5[
    "direction_allowed_v5"
].map(parse_bool)

direct_change_table = evidence_v5.loc[original_direct_mask].groupby(
    ["evidence_tier_v5", "tier_reason_v5"]
).size().rename("rows").to_frame()
display(direct_change_table)
''')

md(r'''
## 4. 清理Related和Similar层

Related和Similar不需要达到Direct的标准，但仍不能只靠一个普通词匹配。

本节重点删除：

- 只有`employment`一个共同词的完全不同法规；
- 只有`gains`一个共同词的不同税种；
- 不同年度的Finance Bill被误标成同一个相关政策。

这些材料本来就不影响概率，因此宁可少展示，也不应让用户误以为它们高度相关。
''')

code(r'''
def refine_context_row(row):
    # 返回修正后的非Direct层级和原因
    tier = safe_text(row["evidence_tier_v5"])
    if tier in {"direct_historical_evidence", "reject"}:
        return tier, safe_text(row["tier_reason_v5"])

    source_type = safe_text(row["source_type"])
    origin = safe_text(row["candidate_origin_v4"])
    query_text = " ".join([
        safe_text(row["query_title"]),
        safe_text(row["query_policy_object"]),
    ])
    evidence_text = " ".join([
        safe_text(row["evidence_title"]),
        safe_text(row.get("historical_policy_object")),
        safe_text(row["evidence_text"]),
    ])
    shared, containment, jaccard = token_metrics(
        row["query_policy_object"], row["evidence_title"]
    )

    if source_type in {"manifesto", "bill_reference"}:
        return tier, safe_text(row["tier_reason_v5"])

    query_is_recurring = bool(RECURRING_BILL_PATTERN.search(query_text))
    evidence_is_recurring = bool(RECURRING_BILL_PATTERN.search(evidence_text))
    if query_is_recurring and evidence_is_recurring:
        query_date = pd.Timestamp(row["query_date"])
        evidence_date = pd.to_datetime(row["evidence_date"], errors="coerce")
        age_days = (
            int((query_date - evidence_date).days)
            if pd.notna(evidence_date) else 999999
        )
        if age_days > MAX_RECURRING_BILL_AGE_DAYS:
            if len(shared) >= 2:
                return (
                    "similar_historical_vote",
                    "recurring_bill_old_cycle_similar_only",
                )
            return "reject", "recurring_bill_old_cycle_too_generic"

    # 11B回退材料必须至少有两个共同政策实词
    if origin == "11b_related_fallback" and len(shared) < 2:
        return "reject", "fallback_has_fewer_than_two_policy_terms"

    return tier, safe_text(row["tier_reason_v5"])


context_decisions = evidence_v5.apply(
    lambda row: pd.Series(refine_context_row(row)), axis=1
)
context_decisions.columns = ["evidence_tier_v5", "tier_reason_v5"]
evidence_v5[["evidence_tier_v5", "tier_reason_v5"]] = context_decisions
evidence_v5.loc[
    ~evidence_v5["evidence_tier_v5"].eq("direct_historical_evidence"),
    "direction_allowed_v5",
] = False

rejected_v5 = evidence_v5[evidence_v5["evidence_tier_v5"].eq("reject")].copy()
kept_v5 = evidence_v5[~evidence_v5["evidence_tier_v5"].eq("reject")].copy()

print("Rows retained:", len(kept_v5))
print("Rows rejected:", len(rejected_v5))
display(kept_v5["evidence_tier_v5"].value_counts().rename("rows").to_frame())
display(rejected_v5["tier_reason_v5"].value_counts().rename("rows").to_frame())
''')

md(r'''
## 5. 重新限制每个查询的展示数量

层级改变后重新排序，并重新生成E1、E2等公开证据编号。

Direct排在最前，Related和Similar分别展示，不能混成一组“支持预测的证据”。
''')

code(r'''
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
    # 每层分别选取，避免某一层占满全部位置
    selected_parts = []
    used_chunks = set()
    used_history_divisions = set()
    for tier in sorted(TIER_ORDER, key=TIER_ORDER.get):
        candidates = group[group["evidence_tier_v5"].eq(tier)].copy()
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
                and history_division in used_history_divisions
            ):
                continue
            accepted.append(row)
            used_chunks.add(chunk_id)
            if history_division:
                used_history_divisions.add(history_division)
            if len(accepted) >= TIER_LIMITS[tier]:
                break
        if accepted:
            selected_parts.append(pd.DataFrame(accepted))
    if not selected_parts:
        return group.iloc[0:0].copy()
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["tier_order_v5"] = selected["evidence_tier_v5"].map(TIER_ORDER)
    selected = selected.sort_values(
        ["tier_order_v5", "ranking_score_v4"], ascending=[True, False]
    ).head(MAX_TOTAL_EVIDENCE)
    return selected.drop(columns="tier_order_v5")


selected_parts = []
for query_id, group in kept_v5.groupby("query_id", sort=False):
    selected_parts.append(select_for_query(group))

final_evidence = (
    pd.concat(selected_parts, ignore_index=True)
    if selected_parts else kept_v5.iloc[0:0].copy()
)
final_evidence["rank_v5"] = final_evidence.groupby("query_id").cumcount() + 1
final_evidence["public_evidence_id_v5"] = final_evidence["rank_v5"].map(
    lambda value: f"E{int(value)}"
)

print("Final evidence rows:", len(final_evidence))
display(final_evidence["evidence_tier_v5"].value_counts().rename("rows").to_frame())
''')

md(r'''
## 6. 更新产品状态和Agent提示语

没有Direct evidence时，产品仍然可以展示Related或Similar，但标题和说明必须分开。

Agent必须先解释冻结模型概率，再说明外部材料属于哪个层级。任何RAG材料都不能覆盖模型概率。
''')

code(r'''
tier_counts = final_evidence.pivot_table(
    index="query_id",
    columns="evidence_tier_v5",
    values="evidence_chunk_id",
    aggfunc="count",
    fill_value=0,
)
for tier in TIER_ORDER:
    if tier not in tier_counts.columns:
        tier_counts[tier] = 0
tier_counts = tier_counts.reset_index()

base_query_columns = [
    column for column in queries.columns
    if column not in TIER_ORDER
    and not column.startswith("product_evidence_status_v5")
    and not column.startswith("product_evidence_message_v5")
]
query_audit = queries[base_query_columns].merge(
    tier_counts, on="query_id", how="left"
)
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


PRODUCT_MESSAGES = {
    "direct_evidence_available": (
        "Directly comparable historical evidence is available. It may explain past "
        "party behaviour but does not replace the frozen prediction."
    ),
    "related_policy_material_only": (
        "No directly comparable evidence was found. The material below concerns a "
        "related policy and does not prove the current prediction."
    ),
    "similar_votes_only": (
        "No directly comparable evidence was found. These are similar historical votes "
        "for exploration only and cannot establish the current stance."
    ),
    "no_external_material": (
        "No reliable external material was found. Present the frozen probability and "
        "its model drivers without inventing supporting evidence."
    ),
}

query_audit["product_evidence_status_v5"] = query_audit.apply(
    product_status, axis=1
)
query_audit["product_evidence_message_v5"] = query_audit[
    "product_evidence_status_v5"
].map(PRODUCT_MESSAGES)
query_audit["rag_changes_probability_v5"] = False
query_audit["rag_produces_separate_prediction_v5"] = False
query_audit["prediction_overridden_v5"] = False


def join_ids(group, tier):
    rows = group[group["evidence_tier_v5"].eq(tier)]
    return " | ".join(rows["public_evidence_id_v5"].astype(str)) or pd.NA


id_rows = []
for query_id, group in final_evidence.groupby("query_id"):
    id_rows.append({
        "query_id": query_id,
        "direct_evidence_ids_v5": join_ids(group, "direct_historical_evidence"),
        "related_policy_ids_v5": join_ids(group, "related_policy_material"),
        "similar_vote_ids_v5": join_ids(group, "similar_historical_vote"),
        "legislative_background_ids_v5": join_ids(group, "legislative_background"),
        "all_evidence_ids_v5": " | ".join(
            group["public_evidence_id_v5"].astype(str)
        ),
    })
id_frame = pd.DataFrame(id_rows)

packet_drop_columns = [
    column for column in packets.columns
    if column.endswith("_v5")
]
agent_packets = packets.drop(columns=packet_drop_columns, errors="ignore").merge(
    query_audit[[
        "query_id", "product_evidence_status_v5",
        "product_evidence_message_v5", "rag_changes_probability_v5",
        "rag_produces_separate_prediction_v5", "prediction_overridden_v5",
    ]],
    on="query_id",
    how="left",
).merge(id_frame, on="query_id", how="left")

agent_packets["agent_instruction_v5"] = (
    "Explain the frozen probability and model drivers first. Use Direct evidence only "
    "for comparable historical party positions. Present Related policy material, "
    "Legislative background and Similar votes in separate sections with their limits. "
    "Never use non-direct material to change the probability or claim the current stance. "
    "If Direct evidence is absent, say so explicitly."
)

display(query_audit["product_evidence_status_v5"].value_counts().rename("queries").to_frame())
''')

md(r'''
## 7. 审核全部Direct evidence

11D只有36条Direct evidence，数量不大。因此11E不再抽样，而是导出全部剩余Direct证据供人工检查。

人工需要填写：

- `manual_same_policy_object`：是否真的是同一政策对象；
- `manual_procedure_comparable`：程序是否可比较；
- `manual_direction_valid`：历史支持或反对能否用于当前对象；
- `manual_notes`：错误原因或补充说明。
''')

code(r'''
all_direct_review = final_evidence[
    final_evidence["evidence_tier_v5"].eq("direct_historical_evidence")
].copy()
all_direct_review["manual_same_policy_object"] = pd.NA
all_direct_review["manual_procedure_comparable"] = pd.NA
all_direct_review["manual_direction_valid"] = pd.NA
all_direct_review["manual_notes"] = pd.NA

print("All Direct evidence review rows:", len(all_direct_review))
display(all_direct_review[[
    "query_date", "query_title", "party", "evidence_date",
    "evidence_title", "tier_reason_v5", "stance_label",
]].sort_values(["query_date", "query_title", "party"]))
''')

md(r'''
## 8. 为Related、Similar和无材料查询保留跨季度样本

Direct已经全部检查，因此这里主要抽取非Direct状态：

- 每个季度5个division；
- 覆盖2025 Q1至2026 Q2；
- 优先包含Related、Similar和No external material；
- 重复运行会得到相同样本。
''')

code(r'''
query_audit["motion_date"] = pd.to_datetime(query_audit["motion_date"])
query_audit["time_quarter_v5"] = query_audit[
    "motion_date"
].dt.to_period("Q").astype(str)

division_pool = query_audit.groupby("division_key", as_index=False).agg(
    motion_date=("motion_date", "first"),
    motion_title=("motion_title", "first"),
    policy_domain_primary=("policy_domain_primary", "first"),
    motion_family=("motion_family", "first"),
    time_quarter=("time_quarter_v5", "first"),
    party_queries=("query_id", "size"),
    direct_queries=(
        "product_evidence_status_v5",
        lambda values: int((values == "direct_evidence_available").sum()),
    ),
    related_queries=(
        "product_evidence_status_v5",
        lambda values: int((values == "related_policy_material_only").sum()),
    ),
    similar_queries=(
        "product_evidence_status_v5",
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
    # Direct另有完整审核，这里优先选择其余三种状态
    selected_keys = []
    preferred_statuses = [
        "related_policy_material_only",
        "similar_votes_only",
        "no_external_material",
    ]
    for status in preferred_statuses:
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
    & ~final_evidence["evidence_tier_v5"].eq("direct_historical_evidence")
].copy()

manual_evidence["manual_policy_relevance"] = pd.NA
manual_evidence["manual_tier_correct"] = pd.NA
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
print("Manual context-review divisions:", manual_divisions["division_key"].nunique())
print("Manual context-review party queries:", len(manual_queries))
print("Manual context-review evidence rows:", len(manual_evidence))
''')

md(r'''
## 9. 结构检查

结构检查确保：

- Direct全部通过新防火墙；
- 低信息量Amendment不会进入Direct；
- Related和Similar不能影响概率；
- 没有时间泄漏和同division泄漏；
- 全部Direct都进入人工审核文件；
- 非Direct抽样覆盖完整六个季度。
''')

code(r'''
direct_rows = final_evidence[
    final_evidence["evidence_tier_v5"].eq("direct_historical_evidence")
].copy()
non_direct_rows = final_evidence[
    ~final_evidence["evidence_tier_v5"].eq("direct_historical_evidence")
].copy()

STRUCTURAL_GATES = {
    "all_queries_preserved_gate": set(query_audit["query_id"]) == set(queries["query_id"]),
    "maximum_seven_results_gate": bool(
        final_evidence.groupby("query_id").size().max() <= MAX_TOTAL_EVIDENCE
    ),
    "all_direct_pass_firewall_gate": bool(
        direct_rows.apply(
            lambda row: direct_firewall(row)[0] == "direct_historical_evidence",
            axis=1,
        ).all()
    ),
    "direct_only_historical_vote_gate": bool(
        direct_rows["source_type"].eq("historical_vote").all()
    ),
    "all_direct_exported_for_review_gate": set(
        all_direct_review["evidence_chunk_id"]
    ) == set(direct_rows["evidence_chunk_id"]),
    "non_direct_never_directional_gate": bool(
        non_direct_rows["direction_allowed_v5"].eq(False).all()
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
    "thirty_context_division_sample_gate": manual_divisions["division_key"].nunique() == 30,
    "five_divisions_per_quarter_gate": bool(
        quarter_sample_counts["divisions"].eq(DIVISIONS_PER_QUARTER).all()
    ),
    "all_six_quarters_gate": set(quarter_sample_counts["time_quarter"]) == set(EXPECTED_QUARTERS),
    "prediction_not_overridden_gate": bool(
        query_audit["prediction_overridden_v5"].eq(False).all()
    ),
    "rag_not_second_predictor_gate": bool(
        query_audit["rag_produces_separate_prediction_v5"].eq(False).all()
    ),
    "no_target_columns_read_gate": not any(
        column.startswith("target_")
        for frame in [queries, evidence, v3_metadata]
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
## 10. 保存输出和摘要

下一步只需要人工检查两个文件：

1. `all_direct_evidence_review_v5.csv`：全部Direct证据；
2. `manual_context_evidence_v5.csv`：跨六季度Related和Similar样本。

通过人工检查前，不应开始LLM解释调用。
''')

code(r'''
def json_safe(value):
    # 将Numpy和Pandas类型转换为JSON原生类型
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


final_evidence.to_csv(OUTPUT_DIR / "evidence_tiers_v5.csv", index=False)
query_audit.to_csv(OUTPUT_DIR / "query_evidence_status_v5.csv", index=False)
agent_packets.to_csv(OUTPUT_DIR / "agent_evidence_packets_v5.csv", index=False)
rejected_v5.to_csv(OUTPUT_DIR / "firewall_rejections_v5.csv", index=False)
all_direct_review.to_csv(OUTPUT_DIR / "all_direct_evidence_review_v5.csv", index=False)
manual_divisions.to_csv(OUTPUT_DIR / "manual_context_divisions_v5.csv", index=False)
manual_queries.to_csv(OUTPUT_DIR / "manual_context_queries_v5.csv", index=False)
manual_evidence.to_csv(OUTPUT_DIR / "manual_context_evidence_v5.csv", index=False)
quarter_sample_counts.to_csv(
    OUTPUT_DIR / "manual_context_quarter_coverage_v5.csv", index=False
)

status_counts = query_audit["product_evidence_status_v5"].value_counts()
tier_counts_output = final_evidence["evidence_tier_v5"].value_counts()
direct_change_counts = evidence_v5.loc[original_direct_mask, "evidence_tier_v5"].value_counts()

manifest = {
    "notebook_version": "11e-direct-evidence-firewall-v5",
    "api_calls": 0,
    "llm_calls": 0,
    "test_labels_read": False,
    "prediction_model_changed": False,
    "retrieval_rerun": False,
    "queries": len(query_audit),
    "unique_divisions": int(query_audit["division_key"].nunique()),
    "final_evidence_rows": len(final_evidence),
    "direct_review_rows": len(all_direct_review),
    "evidence_tier_counts": tier_counts_output.to_dict(),
    "query_status_counts": status_counts.to_dict(),
    "original_direct_reclassification": direct_change_counts.to_dict(),
    "manual_context_divisions": int(manual_divisions["division_key"].nunique()),
    "manual_context_quarters": EXPECTED_QUARTERS,
    "structural_gates": STRUCTURAL_GATES,
    "ready_for_llm_explanation": False,
}
with (OUTPUT_DIR / "run_manifest_v5.json").open("w", encoding="utf-8") as handle:
    json.dump(json_safe(manifest), handle, ensure_ascii=False, indent=2)

summary_lines = [
    "=== DIRECT EVIDENCE FIREWALL SUMMARY FOR REVIEW ===",
    "Notebook version: 11e-direct-evidence-firewall-v5",
    "API calls: 0",
    "LLM calls: 0",
    "Test labels read: False",
    "Prediction model changed: False",
    "Retrieval rerun: False",
    f"Queries: {len(query_audit)}",
    f"Unique divisions: {query_audit['division_key'].nunique()}",
    f"11D original Direct rows: {int(original_direct_mask.sum())}",
    f"11E remaining Direct rows: {len(all_direct_review)}",
    f"Final evidence rows: {len(final_evidence)}",
    f"Queries with Direct evidence rate: {(query_audit['product_evidence_status_v5'] == 'direct_evidence_available').mean():.3f}",
    f"Queries with Related-policy-only rate: {(query_audit['product_evidence_status_v5'] == 'related_policy_material_only').mean():.3f}",
    f"Queries with Similar-votes-only rate: {(query_audit['product_evidence_status_v5'] == 'similar_votes_only').mean():.3f}",
    f"Queries with no external material rate: {(query_audit['product_evidence_status_v5'] == 'no_external_material').mean():.3f}",
    f"All-Direct manual review rows: {len(all_direct_review)}",
    f"Manual context-review divisions: {manual_divisions['division_key'].nunique()}",
    f"Manual context-review evidence rows: {len(manual_evidence)}",
    f"Manual review quarters: {EXPECTED_QUARTERS}",
    f"Structural gates: {json_safe(STRUCTURAL_GATES)}",
    "Ready for LLM explanation: False",
    "Next step: Codex reviews every remaining Direct row and the balanced context sample.",
    f"Output directory: {OUTPUT_DIR}",
    "=== END DIRECT EVIDENCE FIREWALL SUMMARY ===",
]
summary_text = "\n".join(summary_lines)
(OUTPUT_DIR / "direct_evidence_firewall_summary_v5.txt").write_text(
    summary_text, encoding="utf-8"
)

print(summary_text)
print("\nOriginal Direct reclassification:")
display(direct_change_counts.rename("rows").to_frame())
print("\nEvidence tier counts:")
display(tier_counts_output.rename("rows").to_frame())
print("\nQuery product status:")
display(status_counts.rename("queries").to_frame())
print("\nManual context quarter coverage:")
display(quarter_sample_counts)
''')

md(r'''
## 如何判断是否可以进入最终Agent解释

运行完成后不要只看结构检查。还要人工检查：

### Direct evidence

- 检查全部剩余行；
- 至少90%的行必须同时满足同一政策对象、程序可比和方向有效；
- 任意年度Finance Bill误匹配都应视为失败。

### Related和Similar

- 固定跨季度样本的政策相关率至少80%；
- 可以宽于Direct，但标题和限制说明必须准确；
- 完全不同的法规不能因为同属就业、税务或移民领域就被展示。

只有这两个层级都通过，才适合进行少量LLM产品解释测试。
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
