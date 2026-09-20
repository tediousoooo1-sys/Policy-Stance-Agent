import json
from pathlib import Path


OUTPUT = Path(__file__).with_name(
    "08b_locked_validation_failure_analysis_and_gate_repair.ipynb"
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
# 08b — Locked Validation Failure Analysis and Gate Repair

## 这一步要解决什么问题

08 的80条独立验证没有通过。现在不能继续在同一批数据上调整后，又把新成绩称为“独立验证成绩”。

因此，08b只做三件事：

1. 解释08为什么失败；
2. 用确定性规则检查哪些证据本来不应被当成直接立场证据；
3. 为下一批全新验证数据冻结规则。

这一步不调用 API、不重新训练模型、不使用 Test，也不会覆盖08原始结果。
''')

add_markdown(r'''
## 最重要的来源区别

`bill_reference` 是 Parliament Bill 的背景资料。它可以帮助理解 Bill 是什么，但永远不能证明某个政党支持或反对。

`historical_vote` 是过去发生的政党投票。它可能提供方向，但只有在“政策对象、动作、条款、年份和适用政治时期”足够一致时，才能作为直接证据。

例如，2023 Finance Bill 和 2024 Finance Bill 名称相似，但内容和执政党都可能不同。旧 Finance Bill 投票不能仅凭标题相同，直接推断新 Finance Bill 的立场。
''')

add_code(r'''
# 导入本地分析所需的库；本 Notebook 不导入 OpenAI，也不会调用任何 API
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import accuracy_score, f1_score

pd.set_option("display.max_columns", 160)
pd.set_option("display.max_colwidth", 180)

BASE_DIR = Path.cwd()
INPUT_DIR = BASE_DIR / "processed" / "locked_validation_v1"
OUTPUT_DIR = BASE_DIR / "processed" / "locked_validation_failure_audit_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORED_PATH = INPUT_DIR / "locked_scored_results_v1.csv"
JUDGMENTS_PATH = INPUT_DIR / "locked_evidence_judgments_v1.csv"
QUERIES_PATH = INPUT_DIR / "locked_queries_public_v1.csv"
LABELS_PATH = INPUT_DIR / "locked_labels_private_v1.csv"
LOCK_MANIFEST_PATH = INPUT_DIR / "lock_manifest_v1.json"

ELECTION_DATE = pd.Timestamp("2024-07-05")
PARTIES = ["conservative", "green", "labour", "liberal-democrat"]

print("Notebook version: 08b-locked-failure-audit-v2")
print("API calls: 0")
print("LLM calls: 0")
print("Test split used: False")
print("Output directory:", OUTPUT_DIR)
''')

add_markdown(r'''
## 1. 读取并锁定08原始结果

这里读取08已经生成的结果，不重新抽样，也不修改真实标签。

如果查询数量、文件或锁定清单不一致，Notebook会停止，避免误把其他数据混进来。
''')

add_code(r'''
# 统一处理 CSV 中的布尔值、空值和日期
def clean_text(value):
    """把缺失值和多余空格转换成安全字符串。"""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def native_bool(value):
    """把 Python、NumPy 和 CSV 中常见的布尔形式统一为 bool。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean_text(value).lower() in {"true", "1", "yes"}


def json_default(value):
    """把 NumPy 和 Pandas 类型转换为 JSON 可以保存的原生类型。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


required_paths = [
    SCORED_PATH, JUDGMENTS_PATH, QUERIES_PATH,
    LABELS_PATH, LOCK_MANIFEST_PATH,
]
for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

scored = pd.read_csv(SCORED_PATH)
judgments = pd.read_csv(JUDGMENTS_PATH)
queries = pd.read_csv(QUERIES_PATH)
labels = pd.read_csv(LABELS_PATH)
lock_manifest = json.loads(LOCK_MANIFEST_PATH.read_text(encoding="utf-8"))

for column in ["query_date"]:
    if column in scored.columns:
        scored[column] = pd.to_datetime(scored[column], errors="coerce")
    if column in judgments.columns:
        judgments[column] = pd.to_datetime(judgments[column], errors="coerce")

if "evidence_date" in judgments.columns:
    judgments["evidence_date"] = pd.to_datetime(
        judgments["evidence_date"], errors="coerce"
    )

scored["answered"] = scored["final_stance"].isin(["support", "oppose"])
scored["correct_if_answered"] = np.where(
    scored["answered"],
    scored["final_stance"].eq(scored["target_policy_stance"]),
    np.nan,
)

STRUCTURAL_GATES = {
    "eighty_queries_gate": len(scored) == 80,
    "unique_query_gate": scored["query_id"].nunique() == 80,
    "four_parties_gate": set(scored["party"]) == set(PARTIES),
    "judgment_query_subset_gate": set(judgments["query_id"]).issubset(
        set(scored["query_id"])
    ),
    "locked_query_hash_present_gate": bool(
        lock_manifest.get("locked_queries_sha256")
    ),
    "locked_label_hash_present_gate": bool(
        lock_manifest.get("locked_labels_sha256")
    ),
    "test_split_gate": lock_manifest.get("test_split_used") is False,
}

print("Locked query SHA-256:", lock_manifest["locked_queries_sha256"])
print("Locked label SHA-256:", lock_manifest["locked_labels_sha256"])
print("Structural gates:", STRUCTURAL_GATES)

if not all(STRUCTURAL_GATES.values()):
    raise ValueError("08原始锁定文件不完整或已改变，请先检查输入文件。")

print("Scored queries:", len(scored))
print("Evidence judgments:", len(judgments))
''')

add_markdown(r'''
## 2. 先复现失败，而不是马上改规则

这一步查看原系统到底在哪些地方出错：

- `Coverage`：80条中有多少条给出了支持或反对，而不是拒答；
- `Selective accuracy`：只在已经回答的记录中计算正确率；
- `Oppose recall among answered`：真实为反对且系统选择回答时，有多少被正确识别；
- `Party metrics`：防止总体平均数掩盖某个政党的失败。
''')

add_code(r'''
# 复现08的核心指标，并增加真实支持/反对的混淆检查
answered = scored[scored["answered"]].copy()

original_metrics = {
    "queries": len(scored),
    "answered": len(answered),
    "coverage": scored["answered"].mean(),
    "selective_accuracy": answered["correct_if_answered"].mean(),
    "answered_macro_f1": f1_score(
        answered["target_policy_stance"],
        answered["final_stance"],
        labels=["oppose", "support"],
        average="macro",
        zero_division=0,
    ),
}

confusion = pd.crosstab(
    answered["target_policy_stance"],
    answered["final_stance"],
    margins=True,
)

party_original = scored.groupby("party").apply(
    lambda group: pd.Series({
        "queries": len(group),
        "answered": int(group["answered"].sum()),
        "coverage": group["answered"].mean(),
        "selective_accuracy": (
            group.loc[group["answered"], "correct_if_answered"].mean()
            if group["answered"].any() else np.nan
        ),
    }),
).reset_index()

print("Original locked-validation metrics")
display(pd.DataFrame([original_metrics]).round(3))
print("Answered confusion matrix")
display(confusion)
print("Original party metrics")
display(party_original.round(3))
''')

add_markdown(r'''
## 3. 把“相似文字”拆成可以检查的合同

TF-IDF 或 embedding 只能说明两段文字在词义上接近，不能自动证明它们是同一项政策。

这里额外抽取五类信息：

1. Amendment 编号；
2. Clause 编号；
3. 文件年份；
4. 动作，例如 increase、freeze、reject；
5. 证据范围，例如整个 Bill 或某一条 Clause。

这些规则不是重新预测政党立场，而是判断一条证据有没有资格影响预测。
''')

add_code(r'''
# 定义文本合同提取函数
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

GENERIC_TITLE_PATTERN = re.compile(
    r"^(delegated legislation|finance bill|financial statement|"
    r"business of the house|programme motion|remaining stages)$",
    flags=re.IGNORECASE,
)

ANNUAL_INSTRUMENT_PATTERN = re.compile(
    r"\b(finance bill|budget resolution|supply bill|appropriation bill|"
    r"taxation charges|national insurance contributions bill)\b",
    flags=re.IGNORECASE,
)

META_NEGATION_PATTERN = re.compile(
    r"\b(disagree(?:s|d|ing)? with|reject(?:s|ed|ing)?|"
    r"oppose(?:s|d|ing)?|decline(?:s|d|ing)? to approve|not approve)\b",
    flags=re.IGNORECASE,
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on",
    "with", "that", "this", "bill", "motion", "clause", "amendment",
    "new", "lords", "order", "regulations", "act", "approve", "stage",
}


def token_set(text):
    """生成去除常见程序词后的实质词集合。"""
    return {
        token for token in WORD_PATTERN.findall(clean_text(text).lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def extract_identifiers(text):
    """提取 Amendment、Clause 和年份等可核对编号。"""
    value = clean_text(text)
    return {
        "amendments": set(match.lower() for match in AMENDMENT_PATTERN.findall(value)),
        "clauses": set(match.lower() for match in CLAUSE_PATTERN.findall(value)),
        "years": set(YEAR_PATTERN.findall(value)),
    }


def action_signature(text):
    """把多种自然语言表达归入少数可比较动作。"""
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


def scope_level(text):
    """区分具体条款、整个法案和一般政策。"""
    lowered = clean_text(text).lower()
    if re.search(r"\b((?:new\s+)?clause|amendment)\b", lowered):
        return "specific_provision"
    if re.search(r"\b(bill|second reading|third reading)\b", lowered):
        return "whole_bill"
    if re.search(r"\b(order|regulations|statutory instrument)\b", lowered):
        return "instrument"
    return "general_policy"


def party_role(party, date):
    """根据日期返回四个政党的制度角色。"""
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
''')

add_markdown(r'''
## 4. 查询本身的质量检查

如果 Motion title、政策对象和程序信息明显冲突，后面的检索再强也无法修复错误问题定义。

这里不会自动改写政策对象，只会标记需要人工查看的查询。普通的 “Delegated Legislation” 等泛化标题不会因为缺少关键词而被误判。
''')

add_code(r'''
# 检查标题与政策对象是否存在明显冲突
def query_contract_status(row):
    """返回查询合同状态和可解释原因。"""
    title = clean_text(row.get("motion_title"))
    policy_object = clean_text(row.get("query_policy_object"))

    if not policy_object:
        return "invalid", "missing_policy_object"
    if GENERIC_TITLE_PATTERN.match(title):
        return "ok", "generic_title_allowed"

    title_tokens = token_set(title)
    object_tokens = token_set(policy_object)
    overlap = title_tokens & object_tokens

    # 标题和对象都包含多个实质词却完全没有交集时，只标记为需要复核
    if len(title_tokens) >= 2 and len(object_tokens) >= 3 and not overlap:
        return "review", "title_object_zero_overlap"

    return "ok", "title_object_consistent"


contract_rows = scored.apply(query_contract_status, axis=1, result_type="expand")
contract_rows.columns = ["query_contract_status", "query_contract_reason"]
scored = pd.concat([scored, contract_rows], axis=1)

print("Query contract status")
display(scored["query_contract_status"].value_counts(dropna=False).to_frame("queries"))

query_review = scored[
    scored["query_contract_status"].ne("ok")
][[
    "query_id", "party", "motion_title", "query_policy_object",
    "query_contract_status", "query_contract_reason",
]]
display(query_review)
''')

add_markdown(r'''
## 5. 更严格的证据准入规则

新规则采用“安全过滤器”设计：

- `bill_reference` 永远是背景；
- LLM原本没有判为直接证据的内容，不会被新规则升级；
- Amendment 或 Clause 编号不一致时降级；
- 同名年度 Bill 跨年份时降级；
- 整个 Bill 与单一 Clause 的范围不一致时降级；
- increase、decrease、freeze 等动作冲突时降级；
- 政权更替前的宽泛历史投票不能直接代表政权更替后的立场；
- Manifesto 必须是明确的同一政策对象，宽泛承诺只能作为背景。

“降级”表示仍可以显示给用户作为背景，但不允许决定最终支持或反对。
''')

add_code(r'''
# 定义互相冲突的动作组
ACTION_CONFLICTS = {
    frozenset({"increase", "decrease"}),
    frozenset({"increase", "freeze"}),
    frozenset({"decrease", "freeze"}),
    frozenset({"approve", "reject"}),
    frozenset({"introduce", "remove"}),
}


def evidence_policy_text(row):
    """优先使用历史政策对象；没有时才使用证据正文。"""
    if clean_text(row.get("source_type")) == "historical_vote":
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


def identifiers_compatible(query_text, evidence_text):
    """检查具体 Amendment、Clause 和年度编号是否冲突。"""
    query_ids = extract_identifiers(query_text)
    evidence_ids = extract_identifiers(evidence_text)

    for key in ["amendments", "clauses"]:
        query_values = query_ids[key]
        evidence_values = evidence_ids[key]
        if query_values and query_values != evidence_values:
            return False, f"{key}_identifier_mismatch"

    query_is_annual = bool(ANNUAL_INSTRUMENT_PATTERN.search(query_text))
    evidence_is_annual = bool(ANNUAL_INSTRUMENT_PATTERN.search(evidence_text))
    if query_is_annual and evidence_is_annual:
        query_years = query_ids["years"]
        evidence_years = evidence_ids["years"]
        if query_years and evidence_years and query_years != evidence_years:
            return False, "annual_bill_year_mismatch"

    return True, "identifiers_compatible"


def actions_compatible(query_text, evidence_text):
    """检查政策动作是否出现明显冲突。"""
    query_actions = action_signature(query_text)
    evidence_actions = action_signature(evidence_text)
    for query_action in query_actions:
        for evidence_action in evidence_actions:
            if frozenset({query_action, evidence_action}) in ACTION_CONFLICTS:
                return False, "policy_action_conflict"
    return True, "actions_compatible"


def apply_v2_gate(row):
    """判断一条证据是否可以直接影响最终立场。"""
    source_type = clean_text(row.get("source_type"))
    query_text = " ".join([
        clean_text(row.get("query_title")),
        clean_text(row.get("query_policy_object")),
        clean_text(row.get("query_parliamentary_procedure")),
    ])
    evidence_text = evidence_policy_text(row)

    # 08原来没有接受的证据，08b不会把它升级，避免扩大规则范围
    if not native_bool(row.get("accepted_directional_direct")):
        return False, "not_accepted_by_original_gate"

    # Bill reference始终只解释背景，不能提供政党方向
    if source_type == "bill_reference":
        return False, "bill_reference_background_only"

    query_status = clean_text(row.get("query_contract_status"))
    if query_status != "ok":
        return False, "query_contract_requires_review"

    # 双重否定必须先转换成基础政策对象和动作，未转换前不冒险推断
    if META_NEGATION_PATTERN.search(clean_text(row.get("query_policy_object"))):
        return False, "meta_negation_requires_normalization"

    identifier_ok, identifier_reason = identifiers_compatible(
        query_text, evidence_text
    )
    if not identifier_ok:
        return False, identifier_reason

    query_scope = scope_level(query_text)
    evidence_scope = scope_level(evidence_text)
    scope_pair = {query_scope, evidence_scope}
    if scope_pair == {"specific_provision", "whole_bill"}:
        return False, "provision_whole_bill_scope_mismatch"

    action_ok, action_reason = actions_compatible(query_text, evidence_text)
    if not action_ok:
        return False, action_reason

    object_relation = clean_text(row.get("object_relation"))
    direct_policy_match = native_bool(row.get("direct_policy_match"))
    same_domain = native_bool(row.get("same_domain"))

    if source_type == "manifesto":
        # Manifesto中的一般承诺不能直接证明一个具体条款或整个Bill的投票
        if query_scope in {"specific_provision", "whole_bill", "instrument"}:
            return False, "manifesto_procedural_context_only"
        if object_relation != "exact" or not direct_policy_match:
            return False, "manifesto_not_exact_enough"
        # Labour执政后的具体行为可能不同于大选前承诺，先降级为背景
        if (
            clean_text(row.get("query_party")) == "labour"
            and pd.Timestamp(row.get("query_date")) >= ELECTION_DATE
        ):
            return False, "pre_government_manifesto_context_only"
        return True, "manifesto_exact_policy_match"

    if source_type == "historical_vote":
        if not native_bool(row.get("historical_contract_complete")):
            return False, "historical_contract_incomplete"
        if object_relation != "exact" and not same_domain:
            return False, "historical_match_too_broad"

        query_role = party_role(row.get("query_party"), row.get("query_date"))
        evidence_role = party_role(
            row.get("evidence_party"), row.get("evidence_date")
        )
        role_changed = (
            query_role != "unknown"
            and evidence_role != "unknown"
            and query_role != evidence_role
        )
        if role_changed and object_relation != "exact":
            return False, "party_role_transition_broad_match"

        # 同名年度法案即使没有写出年份，也不能仅凭宽泛标题直接互推
        if (
            ANNUAL_INSTRUMENT_PATTERN.search(query_text)
            and ANNUAL_INSTRUMENT_PATTERN.search(evidence_text)
            and pd.Timestamp(row.get("query_date")).year
            != pd.Timestamp(row.get("evidence_date")).year
        ):
            return False, "cross_year_annual_bill_history"

        return True, "historical_specific_match"

    return False, "unsupported_source_type"
''')

add_code(r'''
# 把查询合同信息合并到每条证据，然后应用08b规则
contract_info = scored[[
    "query_id", "query_contract_status", "query_contract_reason",
]].drop_duplicates("query_id")

evidence_v2 = judgments.merge(
    contract_info,
    on="query_id",
    how="left",
    validate="many_to_one",
)

gate_output = evidence_v2.apply(apply_v2_gate, axis=1, result_type="expand")
gate_output.columns = ["accepted_directional_direct_v2", "gate_reason_v2"]
evidence_v2 = pd.concat([evidence_v2, gate_output], axis=1)

print("V2 gate reasons")
display(
    evidence_v2["gate_reason_v2"]
    .value_counts(dropna=False)
    .to_frame("evidence_rows")
)

print("Accepted evidence by source")
display(pd.crosstab(
    evidence_v2["source_type"],
    evidence_v2["accepted_directional_direct_v2"],
))
''')

add_markdown(r'''
## 6. 离线重判

这一格不会再次询问 LLM，而是重新汇总已经存在的证据判断：

- 没有合格直接证据 → `insufficient_evidence`；
- 合格证据都支持 → `support`；
- 合格证据都反对 → `oppose`；
- 支持与反对证据冲突 → `insufficient_evidence`。

这里得到的是“如果采用新安全规则，旧80条会怎样”的诊断结果，不是新的独立验证结果。
''')

add_code(r'''
# 按查询汇总通过新门控的证据方向
def rescore_query(group):
    """把通过门控的证据方向汇总成查询级结论。"""
    accepted = group[group["accepted_directional_direct_v2"]].copy()
    directions = set(
        accepted["directional_value"]
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

    return pd.Series({
        "final_stance_v2": stance,
        "decision_reason_v2": reason,
        "accepted_direct_evidence_v2": len(accepted),
        "accepted_supporting_v2": int(
            accepted["directional_value"].eq("supports").sum()
        ),
        "accepted_opposing_v2": int(
            accepted["directional_value"].eq("opposes").sum()
        ),
    })


query_rescore = (
    evidence_v2.groupby("query_id", group_keys=False)
    .apply(rescore_query)
    .reset_index()
)

diagnostic = scored.merge(
    query_rescore,
    on="query_id",
    how="left",
    validate="one_to_one",
)

# 没有证据判断行的查询也必须明确拒答
diagnostic["final_stance_v2"] = diagnostic[
    "final_stance_v2"
].fillna("insufficient_evidence")
diagnostic["decision_reason_v2"] = diagnostic[
    "decision_reason_v2"
].fillna("no_evidence_judgment_rows")
for column in [
    "accepted_direct_evidence_v2",
    "accepted_supporting_v2",
    "accepted_opposing_v2",
]:
    diagnostic[column] = diagnostic[column].fillna(0).astype(int)

diagnostic["answered_v2"] = diagnostic["final_stance_v2"].isin(
    ["support", "oppose"]
)
diagnostic["correct_if_answered_v2"] = np.where(
    diagnostic["answered_v2"],
    diagnostic["final_stance_v2"].eq(
        diagnostic["target_policy_stance"]
    ),
    np.nan,
)
''')

add_markdown(r'''
## 7. 新规则究竟改善了什么

比单看 Accuracy 更重要的是同时查看两项：

- `Incorrect answer block rate`：原来答错的记录，有多少被新规则安全拦截；
- `Correct answer retention rate`：原来答对的记录，有多少仍被保留。

如果只追求第一项，系统可以全部拒答；如果只追求第二项，错误证据又会继续放行。因此两项必须一起看。
''')

add_code(r'''
# 计算诊断重判指标
answered_v2 = diagnostic[diagnostic["answered_v2"]].copy()
original_wrong = diagnostic[
    diagnostic["answered"] & diagnostic["correct_if_answered"].eq(False)
]
original_correct = diagnostic[
    diagnostic["answered"] & diagnostic["correct_if_answered"].eq(True)
]

incorrect_answer_block_rate = (
    (~original_wrong["answered_v2"]).mean()
    if len(original_wrong) else np.nan
)
correct_answer_retention_rate = (
    (
        original_correct["answered_v2"]
        & original_correct["correct_if_answered_v2"].eq(True)
    ).mean()
    if len(original_correct) else np.nan
)

v2_metrics = {
    "queries": len(diagnostic),
    "answered": int(diagnostic["answered_v2"].sum()),
    "coverage": diagnostic["answered_v2"].mean(),
    "selective_accuracy": (
        answered_v2["correct_if_answered_v2"].mean()
        if len(answered_v2) else np.nan
    ),
    "answered_macro_f1": (
        f1_score(
            answered_v2["target_policy_stance"],
            answered_v2["final_stance_v2"],
            labels=["oppose", "support"],
            average="macro",
            zero_division=0,
        ) if len(answered_v2) else np.nan
    ),
    "incorrect_answer_block_rate": incorrect_answer_block_rate,
    "correct_answer_retention_rate": correct_answer_retention_rate,
    "remaining_wrong_answers": int(
        (
            diagnostic["answered_v2"]
            & diagnostic["correct_if_answered_v2"].eq(False)
        ).sum()
    ),
}

comparison = pd.DataFrame([
    {"version": "08_original_locked", **original_metrics},
    {"version": "08b_diagnostic_rescore", **v2_metrics},
])

party_v2 = diagnostic.groupby("party").apply(
    lambda group: pd.Series({
        "queries": len(group),
        "answered": int(group["answered_v2"].sum()),
        "coverage": group["answered_v2"].mean(),
        "selective_accuracy": (
            group.loc[
                group["answered_v2"], "correct_if_answered_v2"
            ].mean()
            if group["answered_v2"].any() else np.nan
        ),
        "wrong_answers": int((
            group["answered_v2"]
            & group["correct_if_answered_v2"].eq(False)
        ).sum()),
    }),
).reset_index()

print("Original vs diagnostic rescore")
display(comparison.round(3))
print("V2 diagnostic party metrics")
display(party_v2.round(3))

print("V2 answered confusion matrix")
if len(answered_v2):
    display(pd.crosstab(
        answered_v2["target_policy_stance"],
        answered_v2["final_stance_v2"],
        margins=True,
    ))
else:
    print("新规则没有保留任何回答，需要放宽过严规则。")
''')

add_markdown(r'''
## 8. 给原错误分配可操作的原因

错误类型来自确定性规则，不由 LLM 自己解释自己。

一条查询可能同时存在多个问题，例如既是不同年份 Finance Bill，又跨越了政权更替。表中会保留全部原因，便于决定下一步只修改什么。
''')

add_code(r'''
# 汇总每个原始错误查询中，原本被接受但现在被拦截的原因
original_wrong_ids = set(original_wrong["query_id"])
wrong_evidence = evidence_v2[
    evidence_v2["query_id"].isin(original_wrong_ids)
    & evidence_v2["accepted_directional_direct"].map(native_bool)
].copy()


def join_unique(values):
    """把非空唯一值按稳定顺序连接起来。"""
    output = []
    for value in values:
        value = clean_text(value)
        if value and value not in output:
            output.append(value)
    return " | ".join(output)


failure_reasons = wrong_evidence.groupby("query_id").agg(
    v2_gate_reasons=("gate_reason_v2", join_unique),
    accepted_sources=("source_type", join_unique),
    accepted_evidence_titles=("evidence_title", join_unique),
).reset_index()

failure_taxonomy = original_wrong[[
    "query_id", "division_key", "party", "query_date", "motion_title",
    "query_policy_object", "motion_family", "era",
    "target_policy_stance", "final_stance",
]].merge(
    failure_reasons,
    on="query_id",
    how="left",
    validate="one_to_one",
).merge(
    diagnostic[[
        "query_id", "final_stance_v2", "answered_v2",
        "correct_if_answered_v2", "query_contract_status",
        "query_contract_reason",
    ]],
    on="query_id",
    how="left",
    validate="one_to_one",
)

print("Original wrong answers and V2 diagnosis")
display(failure_taxonomy)
''')

add_markdown(r'''
## 9. 保存诊断产物

输出文件不会覆盖08。文件名使用 `diagnostic`，提醒我们这80条已经参与规则修改，不能再次充当独立验证集。

通过08b不代表产品已经通过验证。它只表示错误原因能够被稳定识别，而且新规则没有退化成“全部拒答”。
''')

add_code(r'''
# 保存证据级、查询级和错误级诊断结果
evidence_v2.to_csv(
    OUTPUT_DIR / "evidence_gate_diagnostic_v2.csv", index=False
)
diagnostic.to_csv(
    OUTPUT_DIR / "query_rescore_diagnostic_v2.csv", index=False
)
failure_taxonomy.to_csv(
    OUTPUT_DIR / "failure_taxonomy_v2.csv", index=False
)
comparison.to_csv(
    OUTPUT_DIR / "metric_comparison_diagnostic_v2.csv", index=False
)
party_v2.to_csv(
    OUTPUT_DIR / "party_metrics_diagnostic_v2.csv", index=False
)

# 人工队列优先保留仍然答错、查询合同异常和证据冲突的记录
manual_review_queue = diagnostic[
    (
        diagnostic["answered_v2"]
        & diagnostic["correct_if_answered_v2"].eq(False)
    )
    | diagnostic["query_contract_status"].ne("ok")
    | diagnostic["decision_reason_v2"].eq("conflicting_directional_evidence")
].copy()
manual_review_queue.to_csv(
    OUTPUT_DIR / "manual_review_queue_v2.csv", index=False
)

DIAGNOSTIC_GATES = {
    "structural_gates_preserved": all(STRUCTURAL_GATES.values()),
    "all_original_wrong_cases_categorized": (
        failure_taxonomy["query_id"].nunique() == len(original_wrong)
    ),
    "incorrect_answer_block_gate": (
        bool(incorrect_answer_block_rate >= 0.60)
        if not pd.isna(incorrect_answer_block_rate) else False
    ),
    "correct_answer_retention_gate": (
        bool(correct_answer_retention_rate >= 0.40)
        if not pd.isna(correct_answer_retention_rate) else False
    ),
    "nonzero_coverage_gate": bool(v2_metrics["coverage"] > 0),
    "no_api_calls_gate": True,
    "test_split_gate": True,
}

run_manifest = {
    "notebook_version": "08b-locked-failure-audit-v2",
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

with (OUTPUT_DIR / "run_manifest_v2.json").open("w", encoding="utf-8") as handle:
    json.dump(
        run_manifest,
        handle,
        ensure_ascii=False,
        indent=2,
        default=json_default,
    )

print("Saved output directory:", OUTPUT_DIR)
print("Manual review rows:", len(manual_review_queue))
print("Diagnostic gates:", DIAGNOSTIC_GATES)
''')

add_markdown(r'''
## 10. 如何阅读最后摘要

理想结果不是“Accuracy突然变成100%”，而是：

- 大部分原错误被安全拦截；
- 仍保留一部分原来正确的回答；
- Conservative 的错误明显减少；
- `bill_reference` 没有被当成立场证据；
- 不同年份同名 Bill、不同 Amendment 编号和范围不一致证据被降级。

如果诊断门通过，下一步是冻结这些规则，再从尚未使用的 Validation division 中抽取一批全新样本做08c。只有08c才重新回答“面对没见过的数据是否有效”。
''')

add_code(r'''
# 输出便于复制回聊天框的简短摘要
print("=== LOCKED FAILURE AUDIT SUMMARY FOR REVIEW ===")
print("Notebook version: 08b-locked-failure-audit-v2")
print("API calls: 0")
print("LLM calls: 0")
print("Original locked validation remains failed: True")
print("Original answered:", original_metrics["answered"])
print("Original selective accuracy:", round(original_metrics["selective_accuracy"], 3))
print("Original wrong answers:", len(original_wrong))
print("Diagnostic answered:", v2_metrics["answered"])
print("Diagnostic coverage:", round(v2_metrics["coverage"], 3))
print("Diagnostic selective accuracy:", round(v2_metrics["selective_accuracy"], 3))
print("Incorrect answer block rate:", round(incorrect_answer_block_rate, 3))
print("Correct answer retention rate:", round(correct_answer_retention_rate, 3))
print("Remaining wrong answers:", v2_metrics["remaining_wrong_answers"])
print("Query contract review rows:", len(query_review))
print("Manual review queue rows:", len(manual_review_queue))
print("Diagnostic gates:", DIAGNOSTIC_GATES)
print("Ready to freeze rules for a new validation sample:", all(DIAGNOSTIC_GATES.values()))
print("RUN_FINAL_TEST: False")
print("Test result: NOT RUN")
print("Output directory:", OUTPUT_DIR)
print("Next step: freeze the rules, then build a new untouched 08c validation sample.")
print("=== END LOCKED FAILURE AUDIT SUMMARY ===")
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
