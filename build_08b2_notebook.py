import json
from pathlib import Path


OUTPUT = Path(__file__).with_name(
    "08b2_instrument_scope_patch_diagnostic_v2_2.ipynb"
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
# 08b v2.2 — Instrument Scope Priority Patch

## 这一步只修复什么

08b v2.1只剩一个错误：标题属于法定文书，但文书名称中包含单词 `Amendment`，代码便把它误认为普通议会修正案。

例如：

    Home Detention Curfew ... (Amendment) Order 2024

这里的 Amendment 是法定文书名称的一部分，不是议会正在表决的 Amendment 编号。

v2.2只修改范围判断顺序：

    Order / Regulations / Statutory Instrument
    → instrument

    普通 Clause / Parliamentary Amendment
    → specific_provision

其他v2.1规则完全不变。这样可以确认结果变化只来自这一处通用规则，而不是针对某条标签硬编码。
''')

add_code(r'''
# 导入离线分析库；本 Notebook 不调用任何 API
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
INPUT_DIR = PROCESSED_DIR / "locked_validation_failure_audit_v2_1"
LOCK_DIR = PROCESSED_DIR / "locked_validation_v1"
OUTPUT_DIR = PROCESSED_DIR / "locked_validation_failure_audit_v2_2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_PATH = INPUT_DIR / "query_rescore_diagnostic_v2_1.csv"
EVIDENCE_PATH = INPUT_DIR / "evidence_gate_diagnostic_v2_1.csv"
MANIFEST_PATH = INPUT_DIR / "run_manifest_v2_1.json"
LOCK_MANIFEST_PATH = LOCK_DIR / "lock_manifest_v1.json"

ELECTION_DATE = pd.Timestamp("2024-07-05")

print("Notebook version: 08b-instrument-scope-patch-v2.2")
print("API calls: 0")
print("LLM calls: 0")
print("Test split used: False")
print("Output directory:", OUTPUT_DIR)
''')

add_markdown(r'''
## 1. 读取v2.1并确认没有换数据

v2.2使用完全相同的80条查询、真实标签和证据判断。它不会重新检索，也不会重新调用 LLM。
''')

add_code(r'''
# 安全处理空值和 CSV 布尔值
def clean_text(value):
    """把缺失值和多余空格转换成安全字符串。"""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def native_bool(value):
    """把常见布尔值形式统一转换成 bool。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean_text(value).lower() in {"true", "1", "yes"}


def json_default(value):
    """把 NumPy 和 Pandas 类型转换成 JSON 原生类型。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


for path in [
    QUERY_PATH, EVIDENCE_PATH, MANIFEST_PATH, LOCK_MANIFEST_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)

queries = pd.read_csv(QUERY_PATH)
evidence = pd.read_csv(EVIDENCE_PATH)
v2_1_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
lock_manifest = json.loads(LOCK_MANIFEST_PATH.read_text(encoding="utf-8"))

for frame in [queries, evidence]:
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].fillna("")

for column in ["query_date", "evidence_date"]:
    if column in evidence.columns:
        evidence[column] = pd.to_datetime(evidence[column], errors="coerce")
if "query_date" in queries.columns:
    queries["query_date"] = pd.to_datetime(queries["query_date"], errors="coerce")

STRUCTURAL_GATES = {
    "eighty_queries_gate": len(queries) == 80,
    "unique_query_gate": queries["query_id"].nunique() == 80,
    "same_locked_query_hash_gate": (
        v2_1_manifest.get("input_locked_queries_sha256")
        == lock_manifest.get("locked_queries_sha256")
    ),
    "same_locked_label_hash_gate": (
        v2_1_manifest.get("input_locked_labels_sha256")
        == lock_manifest.get("locked_labels_sha256")
    ),
    "v2_1_was_diagnostic_only_gate": (
        v2_1_manifest.get("diagnostic_only") is True
    ),
    "test_split_gate": lock_manifest.get("test_split_used") is False,
}

print("Structural gates:", STRUCTURAL_GATES)
if not all(STRUCTURAL_GATES.values()):
    raise ValueError("v2.1输入或锁定数据已经改变，停止运行。")

print("Queries:", len(queries))
print("Evidence judgments:", len(evidence))
print("Locked query SHA-256:", lock_manifest["locked_queries_sha256"])
''')

add_markdown(r'''
## 2. 修正范围识别优先级

旧规则先寻找 `Amendment`，因此 `(Amendment) Order` 会被误判。

新规则先寻找法定文书关键词。只有不属于法定文书时，才继续判断 Clause 或 Amendment。

这是一条通用解析规则：代码不会查看目标标签，也不会写入具体 division_key。
''')

add_code(r'''
# 修正后的范围识别：法定文书优先于普通修正案
INSTRUMENT_PATTERN = re.compile(
    r"\b(order|regulations|statutory instrument)\b",
    flags=re.IGNORECASE,
)
PROVISION_PATTERN = re.compile(
    r"\b((?:new\s+)?clause|lords?\s+amendment|amendment\s+\d+[a-z]?)\b",
    flags=re.IGNORECASE,
)
BILL_STAGE_PATTERN = re.compile(
    r"\b(bill|second reading|third reading)\b",
    flags=re.IGNORECASE,
)


def corrected_scope_level(text):
    """按产品优先级识别法定文书、具体条款、整个Bill和一般政策。"""
    value = clean_text(text)
    if INSTRUMENT_PATTERN.search(value):
        return "instrument"
    if PROVISION_PATTERN.search(value):
        return "specific_provision"
    if BILL_STAGE_PATTERN.search(value):
        return "whole_bill"
    return "general_policy"


def party_role(party, date):
    """按2024年大选前后返回政党制度角色。"""
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


def scope_patch(row):
    """只对v2.1已接受的历史证据补充法定文书角色变化检查。"""
    originally_accepted = native_bool(
        row.get("accepted_directional_direct_v2_1")
    )
    if not originally_accepted:
        return False, clean_text(row.get("gate_reason_v2_1"))

    source_type = clean_text(row.get("source_type"))
    query_text = " ".join([
        clean_text(row.get("query_title")),
        clean_text(row.get("query_policy_object")),
    ])
    corrected_scope = corrected_scope_level(query_text)

    if source_type == "historical_vote" and corrected_scope == "instrument":
        query_role = party_role(row.get("query_party"), row.get("query_date"))
        evidence_role = party_role(
            row.get("evidence_party"), row.get("evidence_date")
        )
        role_changed = (
            query_role != "unknown"
            and evidence_role != "unknown"
            and query_role != evidence_role
        )
        if role_changed:
            return False, "instrument_institutional_role_transition"

    return True, clean_text(row.get("gate_reason_v2_1"))


patch_output = evidence.apply(scope_patch, axis=1, result_type="expand")
patch_output.columns = [
    "accepted_directional_direct_v2_2",
    "gate_reason_v2_2",
]
evidence_v2_2 = pd.concat([evidence, patch_output], axis=1)

evidence_v2_2["corrected_query_scope_v2_2"] = evidence_v2_2.apply(
    lambda row: corrected_scope_level(" ".join([
        clean_text(row.get("query_title")),
        clean_text(row.get("query_policy_object")),
    ])),
    axis=1,
)

# 显示这一个补丁实际改变了哪些证据，避免隐藏范围扩大
changed_evidence = evidence_v2_2[
    evidence_v2_2["accepted_directional_direct_v2_1"].map(native_bool)
    & ~evidence_v2_2["accepted_directional_direct_v2_2"]
].copy()

print("Evidence rows changed by v2.2:", len(changed_evidence))
display(changed_evidence[[
    "query_id", "query_party", "query_title", "source_type",
    "evidence_date", "evidence_title", "corrected_query_scope_v2_2",
    "gate_reason_v2_2",
]])
''')

add_markdown(r'''
## 3. 使用相同方向重新汇总

v2.2不会更改v2.1已经确定的证据方向，只会撤销错误放行的法定文书历史证据。

如果撤销后没有其他直接证据，系统返回 `insufficient_evidence`。
''')

add_code(r'''
# 按查询重新汇总通过v2.2门控的证据
rescore_rows = []
for query_id, group in evidence_v2_2.groupby("query_id"):
    accepted = group[group["accepted_directional_direct_v2_2"]].copy()
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
        "final_stance_v2_2": stance,
        "decision_reason_v2_2": reason,
        "accepted_direct_evidence_v2_2": len(accepted),
        "accepted_supporting_v2_2": int(
            accepted["effective_direction_v2_1"].eq("supports").sum()
        ),
        "accepted_opposing_v2_2": int(
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

diagnostic["final_stance_v2_2"] = diagnostic[
    "final_stance_v2_2"
].fillna("insufficient_evidence")
diagnostic["decision_reason_v2_2"] = diagnostic[
    "decision_reason_v2_2"
].fillna("no_evidence_judgment_rows")
for column in [
    "accepted_direct_evidence_v2_2",
    "accepted_supporting_v2_2",
    "accepted_opposing_v2_2",
]:
    diagnostic[column] = diagnostic[column].fillna(0).astype(int)

diagnostic["answered_v2_2"] = diagnostic["final_stance_v2_2"].isin(
    ["support", "oppose"]
)
diagnostic["correct_if_answered_v2_2"] = np.where(
    diagnostic["answered_v2_2"],
    diagnostic["final_stance_v2_2"].eq(
        diagnostic["target_policy_stance"]
    ),
    np.nan,
)
''')

add_markdown(r'''
## 4. 验收这一处补丁

补丁必须同时满足：

- 至少修改一条证据，否则说明代码没有生效；
- 只能撤销法定文书历史证据，不能影响 Manifesto 或普通 Clause；
- 原错误解决率至少75%；
- 原正确答案保留率至少50%；
- Coverage至少10%；
- 每个政党至少保留一个回答。

这些仍然是开发门槛。真正的泛化能力需要08c使用全新样本确认。
''')

add_code(r'''
# 计算v2.2诊断指标
original_wrong = diagnostic[
    diagnostic["answered"] & diagnostic["correct_if_answered"].eq(False)
].copy()
original_correct = diagnostic[
    diagnostic["answered"] & diagnostic["correct_if_answered"].eq(True)
].copy()
answered_v2_2 = diagnostic[diagnostic["answered_v2_2"]].copy()

wrong_resolved = (
    ~original_wrong["answered_v2_2"]
    | original_wrong["correct_if_answered_v2_2"].eq(True)
)
wrong_corrected = (
    original_wrong["answered_v2_2"]
    & original_wrong["correct_if_answered_v2_2"].eq(True)
)
correct_retained = (
    original_correct["answered_v2_2"]
    & original_correct["correct_if_answered_v2_2"].eq(True)
)

metrics_v2_2 = {
    "queries": len(diagnostic),
    "answered": int(diagnostic["answered_v2_2"].sum()),
    "coverage": diagnostic["answered_v2_2"].mean(),
    "selective_accuracy": (
        answered_v2_2["correct_if_answered_v2_2"].mean()
        if len(answered_v2_2) else np.nan
    ),
    "answered_macro_f1": (
        f1_score(
            answered_v2_2["target_policy_stance"],
            answered_v2_2["final_stance_v2_2"],
            labels=["oppose", "support"],
            average="macro",
            zero_division=0,
        ) if len(answered_v2_2) else np.nan
    ),
    "wrong_answer_resolution_rate": wrong_resolved.mean(),
    "wrong_answer_correction_rate": wrong_corrected.mean(),
    "correct_answer_retention_rate": correct_retained.mean(),
    "remaining_wrong_answers": int((
        diagnostic["answered_v2_2"]
        & diagnostic["correct_if_answered_v2_2"].eq(False)
    ).sum()),
}

party_rows = []
for party, group in diagnostic.groupby("party"):
    party_answered = group[group["answered_v2_2"]]
    party_rows.append({
        "party": party,
        "queries": len(group),
        "answered": len(party_answered),
        "coverage": group["answered_v2_2"].mean(),
        "selective_accuracy": (
            party_answered["correct_if_answered_v2_2"].mean()
            if len(party_answered) else np.nan
        ),
        "wrong_answers": int((
            group["answered_v2_2"]
            & group["correct_if_answered_v2_2"].eq(False)
        ).sum()),
    })
party_metrics = pd.DataFrame(party_rows)

patch_scope_valid = (
    len(changed_evidence) >= 1
    and changed_evidence["source_type"].eq("historical_vote").all()
    and changed_evidence["corrected_query_scope_v2_2"].eq("instrument").all()
)

DIAGNOSTIC_GATES = {
    "structural_gates_preserved": all(STRUCTURAL_GATES.values()),
    "instrument_scope_patch_activated_gate": len(changed_evidence) >= 1,
    "patch_scope_only_instruments_gate": bool(patch_scope_valid),
    "wrong_answer_resolution_gate": bool(
        metrics_v2_2["wrong_answer_resolution_rate"] >= 0.75
    ),
    "correct_answer_retention_gate": bool(
        metrics_v2_2["correct_answer_retention_rate"] >= 0.50
    ),
    "minimum_coverage_gate": bool(metrics_v2_2["coverage"] >= 0.10),
    "all_parties_answered_gate": int(party_metrics["answered"].min()) >= 1,
    "no_api_calls_gate": True,
    "test_split_gate": True,
}

print("V2.2 metrics")
display(pd.DataFrame([metrics_v2_2]).round(3))
print("Party metrics")
display(party_metrics.round(3))
print("Diagnostic gates:", DIAGNOSTIC_GATES)
''')

add_markdown(r'''
## 5. 保存冻结候选规则

如果全部门槛通过，v2.2成为08c要使用的冻结候选规则。

“冻结”表示08c运行期间不能再根据新样本修改这些规则。否则08c也会变成开发集，而不是独立确认集。
''')

add_code(r'''
# 保存v2.2证据、查询、指标和规则变更审计
evidence_v2_2.to_csv(
    OUTPUT_DIR / "evidence_gate_diagnostic_v2_2.csv", index=False
)
diagnostic.to_csv(
    OUTPUT_DIR / "query_rescore_diagnostic_v2_2.csv", index=False
)
changed_evidence.to_csv(
    OUTPUT_DIR / "instrument_scope_changed_evidence_v2_2.csv", index=False
)
pd.DataFrame([metrics_v2_2]).to_csv(
    OUTPUT_DIR / "metrics_diagnostic_v2_2.csv", index=False
)
party_metrics.to_csv(
    OUTPUT_DIR / "party_metrics_diagnostic_v2_2.csv", index=False
)

remaining_review = diagnostic[
    (
        diagnostic["answered_v2_2"]
        & diagnostic["correct_if_answered_v2_2"].eq(False)
    )
    | diagnostic["decision_reason_v2_2"].eq(
        "conflicting_directional_evidence"
    )
    | diagnostic["query_contract_status"].ne("ok")
].copy()
remaining_review.to_csv(
    OUTPUT_DIR / "manual_review_queue_v2_2.csv", index=False
)

run_manifest = {
    "notebook_version": "08b-instrument-scope-patch-v2.2",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "input_locked_queries_sha256": lock_manifest["locked_queries_sha256"],
    "input_locked_labels_sha256": lock_manifest["locked_labels_sha256"],
    "parent_rule_version": "08b-refined-gate-v2.1",
    "rule_change": (
        "Classify Order, Regulations and Statutory Instrument before "
        "parliamentary Amendment or Clause."
    ),
    "api_calls": 0,
    "llm_calls": 0,
    "test_split_used": False,
    "diagnostic_only": True,
    "original_locked_validation_remains_failed": True,
    "candidate_rules_frozen_for_08c": all(DIAGNOSTIC_GATES.values()),
    "structural_gates": STRUCTURAL_GATES,
    "diagnostic_gates": DIAGNOSTIC_GATES,
}

with (OUTPUT_DIR / "run_manifest_v2_2.json").open(
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
print("Remaining manual-review rows:", len(remaining_review))
''')

add_code(r'''
# 输出可以直接复制回聊天框的摘要
print("=== INSTRUMENT SCOPE PATCH SUMMARY FOR REVIEW ===")
print("Notebook version: 08b-instrument-scope-patch-v2.2")
print("API calls: 0")
print("LLM calls: 0")
print("Original locked validation remains failed: True")
print("Evidence rows changed by patch:", len(changed_evidence))
print("V2.2 answered:", metrics_v2_2["answered"])
print("V2.2 coverage:", round(metrics_v2_2["coverage"], 3))
print("V2.2 selective accuracy:", round(metrics_v2_2["selective_accuracy"], 3))
print("V2.2 Macro-F1:", round(metrics_v2_2["answered_macro_f1"], 3))
print("Wrong answer resolution rate:", round(metrics_v2_2["wrong_answer_resolution_rate"], 3))
print("Wrong answer correction rate:", round(metrics_v2_2["wrong_answer_correction_rate"], 3))
print("Correct answer retention rate:", round(metrics_v2_2["correct_answer_retention_rate"], 3))
print("Remaining wrong answers:", metrics_v2_2["remaining_wrong_answers"])
print("Minimum party answered:", int(party_metrics["answered"].min()))
print("Remaining manual-review rows:", len(remaining_review))
print("Diagnostic gates:", DIAGNOSTIC_GATES)
print("Ready to freeze rules for untouched 08c:", all(DIAGNOSTIC_GATES.values()))
print("RUN_FINAL_TEST: False")
print("Test result: NOT RUN")
print("Output directory:", OUTPUT_DIR)
print("=== END INSTRUMENT SCOPE PATCH SUMMARY ===")
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
