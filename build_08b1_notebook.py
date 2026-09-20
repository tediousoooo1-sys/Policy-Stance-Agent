import json
from pathlib import Path


OUTPUT = Path(__file__).with_name(
    "08b1_refined_evidence_gate_diagnostic_v2_1.ipynb"
)


def lines(text):
    text = text.strip("\n")
    return [line + "\n" for line in text.splitlines()]


cells = []


def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    })


def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    })


add_markdown(r'''
# 08b v2.1 — Refined Evidence Gate Diagnostic

## 为什么需要这一版

08b v2成功拦截了全部12个错误，但只保留了18个原正确答案中的4个。它不是“判断更聪明”，而是“几乎不回答”。

v2.1不会放弃安全门控，而是修复三条过度严格的规则：

1. 历史投票已经有归一化立场，不再让 LLM 二次翻译方向；
2. Clause 编号只在同一份 Bill 内比较；
3. Manifesto 明确匹配同一政策对象和动作时，可以用于具体政策，但不能仅凭宽泛价值观推断。

本 Notebook 仍然只做开发诊断：API调用为0，不使用 Test，也不把旧80条重新包装成独立验证。
''')

add_markdown(r'''
## 产品规则如何转成代码

规则不是“模型觉得相似就通过”，而是可以逐条验收：

| Product rule | Pass condition | Otherwise |
|---|---|---|
| Same Bill clause comparison | Same Bill and same Clause number | Context only |
| Cross-Bill policy comparison | Same substantive policy object | Clause number ignored |
| Historical vote direction | Use normalized historical stance | Do not ask LLM to reinterpret |
| Manifesto evidence | Same object + same action + sufficient anchor coverage | Context only |
| Bill reference | Never provides party direction | Background only |
| Annual Bill history | Same annual version required | Context only |
''')

add_code(r'''
# 导入本地离线分析所需的库；这里不会导入或调用 OpenAI
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import f1_score

pd.set_option("display.max_columns", 180)
pd.set_option("display.max_colwidth", 180)

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "processed"
INPUT_08_DIR = PROCESSED_DIR / "locked_validation_v1"
INPUT_08B_DIR = PROCESSED_DIR / "locked_validation_failure_audit_v2"
OUTPUT_DIR = PROCESSED_DIR / "locked_validation_failure_audit_v2_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_PATH = INPUT_08B_DIR / "query_rescore_diagnostic_v2.csv"
EVIDENCE_PATH = INPUT_08B_DIR / "evidence_gate_diagnostic_v2.csv"
PUBLIC_QUERY_PATH = INPUT_08_DIR / "locked_queries_public_v1.csv"
LOCK_MANIFEST_PATH = INPUT_08_DIR / "lock_manifest_v1.json"

ELECTION_DATE = pd.Timestamp("2024-07-05")
PARTIES = ["conservative", "green", "labour", "liberal-democrat"]

print("Notebook version: 08b-refined-gate-v2.1")
print("API calls: 0")
print("LLM calls: 0")
print("Test split used: False")
print("Output directory:", OUTPUT_DIR)
''')

add_markdown(r'''
## 1. 读取同一批锁定结果

v2.1只能读取08已经产生的80条结果和证据判断。它不会重新抽样，也不会产生新的 LLM 回答。

这样做的目的是比较门控规则，而不是让模型反复尝试直到碰巧答对。
''')

add_code(r'''
# 统一处理空值、布尔值和 JSON 类型
def clean_text(value):
    """把缺失值和多余空格转换成安全字符串。"""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def native_bool(value):
    """把 Python、NumPy 和 CSV 中常见的布尔形式转为 bool。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean_text(value).lower() in {"true", "1", "yes"}


def json_default(value):
    """把 NumPy 和 Pandas 类型转换成 JSON 可以保存的原生类型。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


for path in [
    QUERY_PATH, EVIDENCE_PATH, PUBLIC_QUERY_PATH, LOCK_MANIFEST_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)

queries = pd.read_csv(QUERY_PATH)
evidence = pd.read_csv(EVIDENCE_PATH)
public_queries = pd.read_csv(PUBLIC_QUERY_PATH)
lock_manifest = json.loads(LOCK_MANIFEST_PATH.read_text(encoding="utf-8"))

for frame in [queries, evidence, public_queries]:
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].fillna("")

for column in ["query_date", "evidence_date"]:
    if column in evidence.columns:
        evidence[column] = pd.to_datetime(evidence[column], errors="coerce")
if "query_date" in queries.columns:
    queries["query_date"] = pd.to_datetime(queries["query_date"], errors="coerce")

# 补充政策对象是否直接锚定在 Motion 原文中的信息
query_contract_meta = public_queries[[
    "query_id", "contract_source_anchored", "source_quote",
    "legislation_name_clean", "policy_action",
]].drop_duplicates("query_id")

evidence = evidence.merge(
    query_contract_meta,
    on="query_id",
    how="left",
    validate="many_to_one",
)

queries["answered"] = queries["final_stance"].isin(["support", "oppose"])
queries["correct_if_answered"] = np.where(
    queries["answered"],
    queries["final_stance"].eq(queries["target_policy_stance"]),
    np.nan,
)

STRUCTURAL_GATES = {
    "eighty_queries_gate": len(queries) == 80,
    "unique_query_gate": queries["query_id"].nunique() == 80,
    "four_parties_gate": set(queries["party"]) == set(PARTIES),
    "evidence_query_subset_gate": set(evidence["query_id"]).issubset(
        set(queries["query_id"])
    ),
    "locked_hash_gate": bool(lock_manifest.get("locked_queries_sha256")),
    "test_split_gate": lock_manifest.get("test_split_used") is False,
}

print("Structural gates:", STRUCTURAL_GATES)
if not all(STRUCTURAL_GATES.values()):
    raise ValueError("08或08b输入文件发生变化，停止诊断。")

print("Queries:", len(queries))
print("Evidence judgments:", len(evidence))
print("Locked query SHA-256:", lock_manifest["locked_queries_sha256"])
''')

add_markdown(r'''
## 2. 提取可以核对的政策合同

这里不尝试“理解全部政治含义”，而是先提取少量可靠信息：

- Amendment 和 Clause 编号；
- Bill 名称关键词；
- 政策动作，例如 increase、decrease、freeze；
- 证据讨论的是整个 Bill、具体条款，还是一般政策。

这些信息会决定证据是直接证据还是只能作为背景。
''')

add_code(r'''
# 定义文本合同提取规则
WORD_PATTERN = re.compile(r"[a-z0-9]+")
AMENDMENT_PATTERN = re.compile(
    r"(?:lords?\s+)?amendment(?:\s+no\.?|\s+number)?\s*(\d+[a-z]?)",
    flags=re.IGNORECASE,
)
CLAUSE_PATTERN = re.compile(
    r"(?:new\s+)?clause(?:\s+no\.?|\s+number)?\s*(\d+[a-z]?)",
    flags=re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
ANNUAL_BILL_PATTERN = re.compile(
    r"\b(finance bill|budget resolution|supply bill|appropriation bill|"
    r"taxation charges|national insurance contributions bill)\b",
    flags=re.IGNORECASE,
)

TOKEN_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into",
    "bill", "new", "clause", "amendment", "lords", "stage", "reading",
    "second", "third", "approve", "advance", "proposed", "motion",
    "order", "regulations", "act", "number", "no",
}

BILL_STOPWORDS = TOKEN_STOPWORDS | {
    "commencement", "programme", "report", "remaining",
}


def token_set(text, stopwords=TOKEN_STOPWORDS):
    """提取实质词，去除常见程序词。"""
    return {
        token for token in WORD_PATTERN.findall(clean_text(text).lower())
        if len(token) > 2 and token not in stopwords
    }


def extract_identifiers(text):
    """提取 Amendment、Clause 和年份编号。"""
    value = clean_text(text)
    return {
        "amendments": set(x.lower() for x in AMENDMENT_PATTERN.findall(value)),
        "clauses": set(x.lower() for x in CLAUSE_PATTERN.findall(value)),
        "years": set(YEAR_PATTERN.findall(value)),
    }


def bill_signature(text):
    """为 Bill 名称生成关键词集合，用于判断是否为同一份 Bill。"""
    lowered = clean_text(text).lower()
    if "bill" not in lowered:
        return set()
    return token_set(lowered, stopwords=BILL_STOPWORDS)


def same_bill(query_title, evidence_title):
    """只有 Bill 名称关键词高度重合时，才视为同一份 Bill。"""
    left = bill_signature(query_title)
    right = bill_signature(evidence_title)
    if not left or not right:
        return False
    similarity = len(left & right) / len(left | right)
    return similarity >= 0.75


def action_signature(text):
    """把自然语言动作归入少数可比较类别。"""
    lowered = clean_text(text).lower()
    patterns = [
        ("reject", r"\b(disagree|reject|oppose|decline|not approve)\b"),
        ("remove", r"\b(remove|delete|repeal|abolish|scrap)\b"),
        ("increase", r"\b(increase|raise|higher|expand|extend)\b"),
        ("decrease", r"\b(decrease|reduce|lower|cut)\b"),
        ("freeze", r"\b(freeze|frozen|maintain|retain)\b"),
        ("introduce", r"\b(introduce|insert|add|create|establish)\b"),
        ("approve", r"\b(approve|pass|advance|adopt|agree)\b"),
        ("review", r"\b(review|report|assess|consult|publish|monitor)\b"),
        ("restrict", r"\b(ban|prohibit|restrict|limit)\b"),
    ]
    return {
        label for label, pattern in patterns
        if re.search(pattern, lowered, flags=re.IGNORECASE)
    }


ACTION_CONFLICTS = {
    frozenset({"increase", "decrease"}),
    frozenset({"increase", "freeze"}),
    frozenset({"decrease", "freeze"}),
    frozenset({"approve", "reject"}),
    frozenset({"introduce", "remove"}),
}


def actions_compatible(query_text, evidence_text):
    """检查两段文字是否包含互相冲突的动作。"""
    query_actions = action_signature(query_text)
    evidence_actions = action_signature(evidence_text)
    for query_action in query_actions:
        for evidence_action in evidence_actions:
            if frozenset({query_action, evidence_action}) in ACTION_CONFLICTS:
                return False
    return True


def scope_level(text):
    """区分具体条款、整个 Bill、法定文书和一般政策。"""
    lowered = clean_text(text).lower()
    if re.search(r"\b((?:new\s+)?clause|amendment)\b", lowered):
        return "specific_provision"
    if re.search(r"\b(bill|second reading|third reading)\b", lowered):
        return "whole_bill"
    if re.search(r"\b(order|regulations|statutory instrument)\b", lowered):
        return "instrument"
    return "general_policy"


def party_role(party, date):
    """按2024年大选日期返回政党制度角色。"""
    date = pd.Timestamp(date) if not pd.isna(date) else pd.NaT
    if pd.isna(date):
        return "unknown"
    if date < ELECTION_DATE:
        mapping = {
            "conservative": "governing_party",
            "labour": "main_opposition",
            "green": "smaller_opposition",
            "liberal-democrat": "smaller_opposition",
        }
    else:
        mapping = {
            "labour": "governing_party",
            "conservative": "main_opposition",
            "green": "smaller_opposition",
            "liberal-democrat": "smaller_opposition",
        }
    return mapping.get(clean_text(party).lower(), "unknown")


def historical_policy_text(row):
    """从历史证据中优先取得结构化政策对象。"""
    for column in [
        "historical_substantive_policy_object",
        "historical_contract_policy_object",
        "historical_final_policy_object",
        "historical_vote_proposition",
    ]:
        value = clean_text(row.get(column))
        if value:
            return value
    return " ".join([
        clean_text(row.get("evidence_title")),
        clean_text(row.get("evidence_text")),
    ])


def effective_direction(row):
    """历史投票直接使用归一化标签；其他来源保留LLM方向判断。"""
    if clean_text(row.get("source_type")) == "historical_vote":
        stance = clean_text(row.get("historical_stance_toward_object"))
        if stance == "support":
            return "supports"
        if stance == "oppose":
            return "opposes"
        return ""
    direction = clean_text(row.get("directional_value"))
    return direction if direction in {"supports", "opposes"} else ""
''')

add_markdown(r'''
## 3. v2.1 准入规则

### 历史投票

- 必须有完整历史政策对象和归一化立场；
- 政策对象关系必须是 exact；
- Lords Amendment 编号必须一致；
- Clause 编号只在同一 Bill 内要求一致；
- 年度 Bill 不能跨年份直接互推；
- 整个 Bill 和某一个 Clause 不能互相代替。

### Manifesto

- 不允许为整个 Bill 或法定文书直接定向；
- 对象关系必须 exact；
- 关键词锚定覆盖率至少0.30；
- Manifesto 和查询必须共享明确政策动作；
- 大选前 Labour Manifesto 对执政后的具体行为只作为背景。
''')

add_code(r'''
# 应用精细化证据门控
def apply_refined_gate(row):
    """返回是否允许定向、原因和最终使用的方向。"""
    source_type = clean_text(row.get("source_type"))
    query_title = clean_text(row.get("query_title"))
    query_object = clean_text(row.get("query_policy_object"))
    query_procedure = clean_text(row.get("query_parliamentary_procedure"))
    query_text = " ".join([query_object, query_procedure])

    if not native_bool(row.get("accepted_directional_direct")):
        return False, "not_accepted_by_original_gate", ""

    if source_type == "bill_reference":
        return False, "bill_reference_background_only", ""

    contract_status = clean_text(row.get("query_contract_status"))
    source_anchored = native_bool(row.get("contract_source_anchored"))
    if contract_status == "invalid":
        return False, "invalid_query_contract", ""
    if contract_status == "review" and not source_anchored:
        return False, "unanchored_query_contract_review", ""

    relation = clean_text(row.get("object_relation"))
    evidence_title = clean_text(row.get("evidence_title"))
    evidence_text = (
        historical_policy_text(row)
        if source_type == "historical_vote"
        else " ".join([
            evidence_title,
            clean_text(row.get("evidence_text")),
        ])
    )

    if not actions_compatible(query_text, evidence_text):
        return False, "policy_action_conflict", ""

    query_scope = scope_level(" ".join([query_title, query_object]))
    evidence_scope = scope_level(" ".join([evidence_title, evidence_text]))
    if {query_scope, evidence_scope} == {"specific_provision", "whole_bill"}:
        return False, "provision_whole_bill_scope_mismatch", ""

    query_ids = extract_identifiers(" ".join([query_title, query_object]))
    evidence_ids = extract_identifiers(" ".join([evidence_title, evidence_text]))

    # Amendment编号属于当前议会对象，存在编号时必须严格一致
    if query_ids["amendments"]:
        if query_ids["amendments"] != evidence_ids["amendments"]:
            return False, "amendment_identifier_mismatch", ""

    # Clause编号只在同一份Bill内比较；不同Bill可以有相同政策但不同编号
    same_bill_flag = same_bill(query_title, evidence_title)
    if (
        same_bill_flag
        and query_ids["clauses"]
        and query_ids["clauses"] != evidence_ids["clauses"]
    ):
        return False, "same_bill_clause_identifier_mismatch", ""

    query_is_annual = bool(ANNUAL_BILL_PATTERN.search(
        " ".join([query_title, query_object])
    ))
    evidence_is_annual = bool(ANNUAL_BILL_PATTERN.search(
        " ".join([evidence_title, evidence_text])
    ))
    if query_is_annual and evidence_is_annual:
        query_date = pd.Timestamp(row.get("query_date"))
        evidence_date = pd.Timestamp(row.get("evidence_date"))
        if (
            not pd.isna(query_date)
            and not pd.isna(evidence_date)
            and query_date.year != evidence_date.year
        ):
            return False, "cross_year_annual_bill_history", ""

    if source_type == "historical_vote":
        if not native_bool(row.get("historical_contract_complete")):
            return False, "historical_contract_incomplete", ""
        if relation != "exact":
            return False, "historical_object_not_exact", ""
        direction = effective_direction(row)
        if not direction:
            return False, "historical_normalized_stance_missing", ""

        # 政权更替前的整个Bill或法定文书投票，不直接代表更替后的立场
        query_role = party_role(row.get("query_party"), row.get("query_date"))
        evidence_role = party_role(
            row.get("evidence_party"), row.get("evidence_date")
        )
        role_changed = (
            query_role != "unknown"
            and evidence_role != "unknown"
            and query_role != evidence_role
        )
        if role_changed and query_scope in {"whole_bill", "instrument"}:
            return False, "institutional_role_transition", ""

        reason = (
            "historical_exact_cross_bill_policy_match"
            if query_ids["clauses"] and not same_bill_flag
            else "historical_exact_match"
        )
        return True, reason, direction

    if source_type == "manifesto":
        if contract_status == "review":
            return False, "manifesto_blocked_for_query_contract_review", ""
        if query_scope in {"whole_bill", "instrument"}:
            return False, "manifesto_whole_instrument_context_only", ""
        if relation != "exact":
            return False, "manifesto_object_not_exact", ""

        anchor_coverage = pd.to_numeric(
            pd.Series([row.get("anchor_coverage")]), errors="coerce"
        ).iloc[0]
        if pd.isna(anchor_coverage) or anchor_coverage < 0.30:
            return False, "manifesto_anchor_coverage_too_low", ""

        query_actions = action_signature(query_text)
        evidence_actions = action_signature(evidence_text)
        shared_actions = query_actions & evidence_actions
        substantive_actions = {
            "reject", "remove", "increase", "decrease",
            "freeze", "review", "restrict",
        }
        if not (shared_actions & substantive_actions):
            return False, "manifesto_action_not_explicitly_shared", ""

        if (
            clean_text(row.get("query_party")) == "labour"
            and pd.Timestamp(row.get("query_date")) >= ELECTION_DATE
        ):
            return False, "pre_government_manifesto_context_only", ""

        direction = effective_direction(row)
        if not direction:
            return False, "manifesto_direction_missing", ""
        return True, "manifesto_explicit_object_action_match", direction

    return False, "unsupported_source_type", ""


refined_output = evidence.apply(
    apply_refined_gate,
    axis=1,
    result_type="expand",
)
refined_output.columns = [
    "accepted_directional_direct_v2_1",
    "gate_reason_v2_1",
    "effective_direction_v2_1",
]
evidence_v2_1 = pd.concat([evidence, refined_output], axis=1)

print("Refined gate reasons")
display(
    evidence_v2_1["gate_reason_v2_1"]
    .value_counts(dropna=False)
    .to_frame("evidence_rows")
)

print("Accepted evidence by source")
display(pd.crosstab(
    evidence_v2_1["source_type"],
    evidence_v2_1["accepted_directional_direct_v2_1"],
))
''')

add_markdown(r'''
## 4. 重新汇总，但不重新调用模型

每条查询只根据通过v2.1规则的证据重新汇总：

- 全部支持 → support；
- 全部反对 → oppose；
- 没有直接证据或方向冲突 → insufficient_evidence。

历史投票方向来自已经归一化的 `historical_stance_toward_object`，不再使用 LLM 的二次解释。
''')

add_code(r'''
# 按查询汇总通过门控的方向
rescore_rows = []
for query_id, group in evidence_v2_1.groupby("query_id"):
    accepted = group[group["accepted_directional_direct_v2_1"]].copy()
    directions = set(
        accepted["effective_direction_v2_1"]
        .map(clean_text)
        .loc[lambda series: series.isin(["supports", "opposes"])]
    )
    if directions == {"supports"}:
        stance = "support"
        reason = "all_accepted_evidence_supports"
    elif directions == {"opposes"}:
        stance = "oppose"
        reason = "all_accepted_evidence_opposes"
    elif not directions:
        stance = "insufficient_evidence"
        reason = "no_accepted_directional_evidence"
    else:
        stance = "insufficient_evidence"
        reason = "conflicting_directional_evidence"

    rescore_rows.append({
        "query_id": query_id,
        "final_stance_v2_1": stance,
        "decision_reason_v2_1": reason,
        "accepted_direct_evidence_v2_1": len(accepted),
        "accepted_supporting_v2_1": int(
            accepted["effective_direction_v2_1"].eq("supports").sum()
        ),
        "accepted_opposing_v2_1": int(
            accepted["effective_direction_v2_1"].eq("opposes").sum()
        ),
    })

query_rescore = pd.DataFrame(rescore_rows)
diagnostic = queries.merge(
    query_rescore,
    on="query_id",
    how="left",
    validate="one_to_one",
)

diagnostic["final_stance_v2_1"] = diagnostic[
    "final_stance_v2_1"
].fillna("insufficient_evidence")
diagnostic["decision_reason_v2_1"] = diagnostic[
    "decision_reason_v2_1"
].fillna("no_evidence_judgment_rows")
for column in [
    "accepted_direct_evidence_v2_1",
    "accepted_supporting_v2_1",
    "accepted_opposing_v2_1",
]:
    diagnostic[column] = diagnostic[column].fillna(0).astype(int)

diagnostic["answered_v2_1"] = diagnostic["final_stance_v2_1"].isin(
    ["support", "oppose"]
)
diagnostic["correct_if_answered_v2_1"] = np.where(
    diagnostic["answered_v2_1"],
    diagnostic["final_stance_v2_1"].eq(
        diagnostic["target_policy_stance"]
    ),
    np.nan,
)
''')

add_markdown(r'''
## 5. 同时衡量安全性与可用性

这一版不再只计算“错误拦截率”，因为错误也可能被确定性历史标签纠正。

- `Wrong answer resolution rate`：原错误被改正或拒答的比例；
- `Wrong answer correction rate`：原错误被直接改成正确答案的比例；
- `Correct answer retention rate`：原正确答案仍然正确回答的比例；
- `Coverage`：系统愿意回答的比例。

只有安全性和可用性同时达到最低标准，规则才可以进入新的独立验证。
''')

add_code(r'''
# 计算原始结果、08b v2和v2.1之间的诊断比较
original_wrong = diagnostic[
    diagnostic["answered"] & diagnostic["correct_if_answered"].eq(False)
].copy()
original_correct = diagnostic[
    diagnostic["answered"] & diagnostic["correct_if_answered"].eq(True)
].copy()
answered_v2_1 = diagnostic[diagnostic["answered_v2_1"]].copy()

wrong_resolved_mask = (
    ~original_wrong["answered_v2_1"]
    | original_wrong["correct_if_answered_v2_1"].eq(True)
)
wrong_corrected_mask = (
    original_wrong["answered_v2_1"]
    & original_wrong["correct_if_answered_v2_1"].eq(True)
)
correct_retained_mask = (
    original_correct["answered_v2_1"]
    & original_correct["correct_if_answered_v2_1"].eq(True)
)

metrics_v2_1 = {
    "queries": len(diagnostic),
    "answered": int(diagnostic["answered_v2_1"].sum()),
    "coverage": diagnostic["answered_v2_1"].mean(),
    "selective_accuracy": (
        answered_v2_1["correct_if_answered_v2_1"].mean()
        if len(answered_v2_1) else np.nan
    ),
    "answered_macro_f1": (
        f1_score(
            answered_v2_1["target_policy_stance"],
            answered_v2_1["final_stance_v2_1"],
            labels=["oppose", "support"],
            average="macro",
            zero_division=0,
        ) if len(answered_v2_1) else np.nan
    ),
    "wrong_answer_resolution_rate": wrong_resolved_mask.mean(),
    "wrong_answer_correction_rate": wrong_corrected_mask.mean(),
    "correct_answer_retention_rate": correct_retained_mask.mean(),
    "remaining_wrong_answers": int((
        diagnostic["answered_v2_1"]
        & diagnostic["correct_if_answered_v2_1"].eq(False)
    ).sum()),
}

original_metrics = {
    "queries": len(diagnostic),
    "answered": int(diagnostic["answered"].sum()),
    "coverage": diagnostic["answered"].mean(),
    "selective_accuracy": diagnostic.loc[
        diagnostic["answered"], "correct_if_answered"
    ].mean(),
}

comparison = pd.DataFrame([
    {"version": "08_original_locked", **original_metrics},
    {
        "version": "08b_v2_over_strict",
        "queries": len(diagnostic),
        "answered": int(diagnostic["answered_v2"].sum()),
        "coverage": diagnostic["answered_v2"].mean(),
        "selective_accuracy": diagnostic.loc[
            diagnostic["answered_v2"], "correct_if_answered_v2"
        ].mean(),
    },
    {"version": "08b_v2_1_refined", **metrics_v2_1},
])

party_rows = []
for party, group in diagnostic.groupby("party"):
    party_answered = group[group["answered_v2_1"]]
    party_rows.append({
        "party": party,
        "queries": len(group),
        "answered": len(party_answered),
        "coverage": group["answered_v2_1"].mean(),
        "selective_accuracy": (
            party_answered["correct_if_answered_v2_1"].mean()
            if len(party_answered) else np.nan
        ),
        "wrong_answers": int((
            group["answered_v2_1"]
            & group["correct_if_answered_v2_1"].eq(False)
        ).sum()),
    })
party_metrics = pd.DataFrame(party_rows)

print("Version comparison")
display(comparison.round(3))
print("Party metrics")
display(party_metrics.round(3))

print("Answered confusion matrix")
if len(answered_v2_1):
    display(pd.crosstab(
        answered_v2_1["target_policy_stance"],
        answered_v2_1["final_stance_v2_1"],
        margins=True,
    ))
''')

add_markdown(r'''
## 6. 输出PRD式验收表

这一表把技术规则翻译成产品经理可以直接检查的格式：需求、实现方式、通过数量和失败数量。

注意：这里的 Pass 只表示规则按预期执行，不表示模型最终效果已经通过独立验证。
''')

add_code(r'''
# 汇总每条产品规则在证据级别的执行情况
rule_rows = [
    {
        "requirement_id": "R1",
        "product_rule": "Bill reference is background only",
        "implementation": "Never allow bill_reference to set party direction",
        "blocked_rows": int(
            evidence_v2_1["gate_reason_v2_1"]
            .eq("bill_reference_background_only").sum()
        ),
    },
    {
        "requirement_id": "R2",
        "product_rule": "Historical direction uses normalized stance",
        "implementation": "Map support/oppose without LLM reinterpretation",
        "blocked_rows": int(
            evidence_v2_1["gate_reason_v2_1"]
            .eq("historical_normalized_stance_missing").sum()
        ),
    },
    {
        "requirement_id": "R3",
        "product_rule": "Clause number must match only within the same Bill",
        "implementation": "Compare Clause IDs only when Bill similarity >= 0.75",
        "blocked_rows": int(
            evidence_v2_1["gate_reason_v2_1"]
            .eq("same_bill_clause_identifier_mismatch").sum()
        ),
    },
    {
        "requirement_id": "R4",
        "product_rule": "Different annual Bills are not direct evidence",
        "implementation": "Block cross-year annual Bill history",
        "blocked_rows": int(
            evidence_v2_1["gate_reason_v2_1"]
            .eq("cross_year_annual_bill_history").sum()
        ),
    },
    {
        "requirement_id": "R5",
        "product_rule": "Manifesto must match object and action",
        "implementation": "Exact object, anchor >= 0.30 and shared action",
        "blocked_rows": int(
            evidence_v2_1["gate_reason_v2_1"].isin([
                "manifesto_object_not_exact",
                "manifesto_anchor_coverage_too_low",
                "manifesto_action_not_explicitly_shared",
            ]).sum()
        ),
    },
    {
        "requirement_id": "R6",
        "product_rule": "Role transition limits institutional history",
        "implementation": "Block pre-election Bill/instrument evidence after role change",
        "blocked_rows": int(
            evidence_v2_1["gate_reason_v2_1"]
            .eq("institutional_role_transition").sum()
        ),
    },
]

requirements_audit = pd.DataFrame(rule_rows)
display(requirements_audit)
''')

add_markdown(r'''
## 7. 保存结果并决定能否进入08c

建议的开发门槛是：

- 至少解决75%的原错误；
- 至少保留50%的原正确答案；
- Coverage至少10%；
- 每个政党至少保留一个回答；
- `bill_reference` 不提供任何方向；
- 不调用 API，不使用 Test。

这些门槛只决定规则是否值得冻结，不代表最终产品验收标准。
''')

add_code(r'''
# 保存诊断结果和运行清单
minimum_party_answered = int(party_metrics["answered"].min())
bill_reference_direction_count = int((
    evidence_v2_1["source_type"].eq("bill_reference")
    & evidence_v2_1["accepted_directional_direct_v2_1"]
).sum())

DIAGNOSTIC_GATES = {
    "structural_gates_preserved": all(STRUCTURAL_GATES.values()),
    "wrong_answer_resolution_gate": bool(
        metrics_v2_1["wrong_answer_resolution_rate"] >= 0.75
    ),
    "correct_answer_retention_gate": bool(
        metrics_v2_1["correct_answer_retention_rate"] >= 0.50
    ),
    "minimum_coverage_gate": bool(metrics_v2_1["coverage"] >= 0.10),
    "all_parties_answered_gate": minimum_party_answered >= 1,
    "bill_reference_background_gate": bill_reference_direction_count == 0,
    "no_api_calls_gate": True,
    "test_split_gate": True,
}

evidence_v2_1.to_csv(
    OUTPUT_DIR / "evidence_gate_diagnostic_v2_1.csv", index=False
)
diagnostic.to_csv(
    OUTPUT_DIR / "query_rescore_diagnostic_v2_1.csv", index=False
)
comparison.to_csv(
    OUTPUT_DIR / "metric_comparison_diagnostic_v2_1.csv", index=False
)
party_metrics.to_csv(
    OUTPUT_DIR / "party_metrics_diagnostic_v2_1.csv", index=False
)
requirements_audit.to_csv(
    OUTPUT_DIR / "product_requirements_audit_v2_1.csv", index=False
)

# 只把v2.1仍然答错、证据冲突或查询合同异常的记录放入人工队列
manual_review_queue = diagnostic[
    (
        diagnostic["answered_v2_1"]
        & diagnostic["correct_if_answered_v2_1"].eq(False)
    )
    | diagnostic["decision_reason_v2_1"].eq(
        "conflicting_directional_evidence"
    )
    | diagnostic["query_contract_status"].ne("ok")
].copy()
manual_review_queue.to_csv(
    OUTPUT_DIR / "manual_review_queue_v2_1.csv", index=False
)

run_manifest = {
    "notebook_version": "08b-refined-gate-v2.1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "input_locked_queries_sha256": lock_manifest["locked_queries_sha256"],
    "input_locked_labels_sha256": lock_manifest["locked_labels_sha256"],
    "api_calls": 0,
    "llm_calls": 0,
    "test_split_used": False,
    "diagnostic_only": True,
    "original_locked_validation_remains_failed": True,
    "structural_gates": STRUCTURAL_GATES,
    "diagnostic_gates": DIAGNOSTIC_GATES,
}

with (OUTPUT_DIR / "run_manifest_v2_1.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(
        run_manifest,
        handle,
        ensure_ascii=False,
        indent=2,
        default=json_default,
    )

print("Saved output directory:", OUTPUT_DIR)
print("Manual review queue rows:", len(manual_review_queue))
print("Diagnostic gates:", DIAGNOSTIC_GATES)
''')

add_code(r'''
# 输出可以直接复制回聊天框的摘要
print("=== REFINED EVIDENCE GATE SUMMARY FOR REVIEW ===")
print("Notebook version: 08b-refined-gate-v2.1")
print("API calls: 0")
print("LLM calls: 0")
print("Original locked validation remains failed: True")
print("Original answered:", original_metrics["answered"])
print("Original selective accuracy:", round(original_metrics["selective_accuracy"], 3))
print("Original wrong answers:", len(original_wrong))
print("V2.1 answered:", metrics_v2_1["answered"])
print("V2.1 coverage:", round(metrics_v2_1["coverage"], 3))
print("V2.1 selective accuracy:", round(metrics_v2_1["selective_accuracy"], 3))
print("V2.1 Macro-F1:", round(metrics_v2_1["answered_macro_f1"], 3))
print("Wrong answer resolution rate:", round(metrics_v2_1["wrong_answer_resolution_rate"], 3))
print("Wrong answer correction rate:", round(metrics_v2_1["wrong_answer_correction_rate"], 3))
print("Correct answer retention rate:", round(metrics_v2_1["correct_answer_retention_rate"], 3))
print("Remaining wrong answers:", metrics_v2_1["remaining_wrong_answers"])
print("Minimum party answered:", minimum_party_answered)
print("Manual review queue rows:", len(manual_review_queue))
print("Diagnostic gates:", DIAGNOSTIC_GATES)
print("Ready to freeze rules for untouched 08c:", all(DIAGNOSTIC_GATES.values()))
print("RUN_FINAL_TEST: False")
print("Test result: NOT RUN")
print("Output directory:", OUTPUT_DIR)
print("=== END REFINED EVIDENCE GATE SUMMARY ===")
''')


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print(OUTPUT)
