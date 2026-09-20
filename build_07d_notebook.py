import json
from pathlib import Path


OUTPUT = Path(__file__).with_name("07d_source_aware_gate_rescoring.ipynb")


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
# 07d — Source-aware Gate Rescoring

## 这一步要解决什么

07c 中，LLM 对四个直接案例的原始立场实际上全部判断正确，但代码门控把其中两个答案改成了 insufficient_evidence。

原因是旧门控只要看到 procedure_relation=different 就拒绝证据。这个规则过于简单：

- Manifesto 本身没有 Second Reading、amendment 等议会程序，不应该因为程序不同而被删除；
- 已经完成 polarity 归一化的历史投票，主要应比较政策对象，而不是要求法律程序词完全相同；
- 但“要求提交报告”与“支持底层政策”、Second Reading 与 Third Reading 仍然不能混为一谈。

07d 不调用 API。它直接读取 07c 已付费生成的14条结果，用更精确的来源感知规则重新评分。
''')

add_markdown(r'''
## procedure_relation 被拦截后发生了什么

模型先输出：

    proposed_stance = oppose

但旧代码只统计同时满足以下条件的证据：

    object_relation 合格
    evidence_role = directional_direct
    procedure_relation 不能是 different

如果 procedure_relation=different，这条证据不会进入 direct_evidence_count。计数变成0以后，最终输出被强制改为：

    final_stance = insufficient_evidence

因此“被拦截”不是 API 拒绝，也不是证据消失，而是安全代码覆盖了模型的原始答案。
''')

add_code(r'''
# 导入库并设置路径
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

pd.set_option("display.max_colwidth", 180)
pd.set_option("display.max_columns", 140)

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "processed"
INPUT_DIR = PROCESSED_DIR / "rag_agent_eval_v3"
OUTPUT_DIR = PROCESSED_DIR / "rag_agent_eval_v4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 07d 只重评分已有结果，永远不调用 API
API_CALLS = 0
LLM_CALLS = 0
RUN_FINAL_TEST = False

required_paths = {
    "cases": INPUT_DIR / "evaluation_cases_v3.csv",
    "evidence": INPUT_DIR / "evaluation_evidence_v3.csv",
    "results": INPUT_DIR / "agent_results_flat_v3.csv",
    "judgments": INPUT_DIR / "evidence_judgments_v3.csv",
}
missing = [str(path) for path in required_paths.values() if not path.exists()]
if missing:
    raise FileNotFoundError("缺少 07c 输出文件：\n" + "\n".join(missing))

cases = pd.read_csv(required_paths["cases"])
evidence = pd.read_csv(required_paths["evidence"])
results_v3 = pd.read_csv(required_paths["results"])
judgments = pd.read_csv(required_paths["judgments"])

for frame in [cases, evidence, results_v3, judgments]:
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].fillna("")

print("07c cases:", len(cases))
print("07c evidence rows:", len(evidence))
print("07c completed results:", results_v3["status"].eq("success").sum())
print("API calls in 07d:", API_CALLS)
''')

add_markdown(r'''
## 1. 新门控如何工作

先判断政策对象，再根据证据来源处理程序：

### Manifesto

Manifesto 没有议会程序。只要明确讨论相同政策对象并给出方向，就可以作为直接证据。

### Historical vote

历史证据必须有完整的历史政策对象和归一化立场。一般不再要求程序词完全相同，但保留两个安全例外：

1. 当前问题是 report、review、statement、assessment 或 consultation 时，支持底层政策不能证明支持这项报告要求；
2. 同一 Bill 的 Second Reading 不能自动证明 Third Reading 的最终立场。

### Bill reference

Bill reference 提供法律背景，不直接代表政党立场。
''')

add_code(r'''
# 文本清理与范围识别函数
def clean_text(value):
    """把缺失值和多余空格统一处理。"""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


REPORTING_PATTERN = re.compile(
    r"\b(report|reporting|review|statement|assessment|consultation|"
    r"publish|publication|lay before|monitoring)\b",
    flags=re.IGNORECASE,
)


def has_reporting_scope(text):
    """判断政策对象是否主要属于报告、审查或咨询要求。"""
    return bool(REPORTING_PATTERN.search(clean_text(text)))


def reading_stage(text):
    """从标题中提取 Bill 的议会审议阶段。"""
    lowered = clean_text(text).lower()
    if "first reading" in lowered or "bill introduction" in lowered:
        return "first_reading"
    if "second reading" in lowered:
        return "second_reading"
    if "third reading" in lowered:
        return "third_reading"
    return ""


def native_bool(value):
    """把 Pandas、NumPy 和字符串布尔值转成 Python bool。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean_text(value).lower() in {"true", "1", "yes"}


# 合并模型判断、证据元数据和查询元数据
audit = judgments.merge(
    evidence,
    on=["case_id", "evidence_id"],
    how="left",
    suffixes=("_model", "_evidence"),
)
audit = audit.merge(
    cases[[
        "case_id", "motion_title", "query_substantive_policy_object",
        "query_parliamentary_procedure", "query_contract_status",
    ]],
    on="case_id",
    how="left",
)

audit["query_reporting_scope"] = audit.apply(
    lambda row: has_reporting_scope(
        clean_text(row["query_substantive_policy_object"])
        + " " + clean_text(row["motion_title"])
    ),
    axis=1,
)
audit["evidence_reporting_scope"] = audit.apply(
    lambda row: has_reporting_scope(
        clean_text(row.get("historical_substantive_policy_object"))
        if row["source_type"] == "historical_vote"
        else clean_text(row.get("evidence_text"))
    ),
    axis=1,
)
audit["query_reading_stage"] = audit["motion_title"].map(reading_stage)
audit["evidence_reading_stage"] = audit["evidence_title"].map(reading_stage)
audit["reading_stage_mismatch"] = (
    audit["query_reading_stage"].ne("")
    & audit["evidence_reading_stage"].ne("")
    & audit["query_reading_stage"].ne(audit["evidence_reading_stage"])
)
audit["reporting_scope_mismatch"] = (
    audit["query_reporting_scope"]
    & ~audit["evidence_reporting_scope"]
)

display(audit[[
    "case_id", "evidence_id", "source_type", "evidence_title",
    "object_relation", "procedure_relation", "evidence_role",
    "query_reporting_scope", "evidence_reporting_scope",
    "reading_stage_mismatch", "reporting_scope_mismatch",
]])
''')

add_code(r'''
# 按来源判断一条模型方向证据能否被最终代码接受
def evaluate_gate(row):
    """返回是否接受直接证据、有效角色和拦截原因。"""
    model_direct = (
        row["evidence_role"] == "directional_direct"
        and row["directional_value"] in ["supports", "opposes"]
        and row["object_relation"]
        in ["exact", "substantively_compatible"]
    )

    if not model_direct:
        # 模型虽然可能把证据叫作 direct，但对象不匹配时必须降级
        if row["evidence_role"] == "directional_direct":
            return False, "relevant_context", "object_not_compatible"
        return False, row["evidence_role"], "model_not_direct"

    if row["source_type"] == "bill_reference":
        return False, "background_reference", "bill_reference_has_no_party_direction"

    if row["reporting_scope_mismatch"]:
        return False, "relevant_context", "reporting_scope_mismatch"

    if row["reading_stage_mismatch"]:
        return False, "relevant_context", "reading_stage_mismatch"

    if row["source_type"] == "manifesto":
        # Manifesto 没有议会程序，因此忽略 procedure_relation
        return True, "directional_direct", "manifesto_object_match"

    if row["source_type"] == "historical_vote":
        contract_complete = native_bool(row.get("historical_contract_complete"))
        if not contract_complete:
            return False, "relevant_context", "historical_contract_incomplete"
        # 历史对象与归一化立场完整时，不再要求程序词完全相同
        return True, "directional_direct", "normalized_historical_object_match"

    return False, "relevant_context", "unsupported_source_type"


gate_results = audit.apply(evaluate_gate, axis=1, result_type="expand")
gate_results.columns = [
    "accepted_directional_direct", "effective_evidence_role", "gate_reason"
]
audit = pd.concat([audit, gate_results], axis=1)

# 单独展示 07c 中四个错误最终决定受到的影响
display(audit[audit["case_id"].isin(["D01", "D03", "C05", "C06"])][[
    "case_id", "evidence_id", "source_type", "evidence_title",
    "evidence_role", "object_relation", "procedure_relation",
    "accepted_directional_direct", "effective_evidence_role", "gate_reason",
]])
''')

add_markdown(r'''
## 2. 用现有 LLM 输出重新生成最终答案

这一部分不改变 LLM 原始回答。代码只重新决定哪些证据有资格进入最终方向计数。

当直接证据全部支持时输出 support；全部反对时输出 oppose；没有合格证据或方向冲突时输出 insufficient_evidence。

公开概率只在最终方向与 support probability 一致时保留。拒答案例的公开概率为空，而不是伪装成50%的预测。
''')

add_code(r'''
# 按案例聚合通过新门控的方向证据
case_rows = []
for _, case in cases.iterrows():
    case_id = case["case_id"]
    case_audit = audit[audit["case_id"].eq(case_id)]
    accepted = case_audit[case_audit["accepted_directional_direct"]]
    directions = set(accepted["directional_value"])

    query_clear = case["query_contract_status"] in [
        "curated_clear", "source_anchored"
    ]
    if not query_clear or accepted.empty:
        final_stance_v4 = "insufficient_evidence"
    elif directions == {"supports"}:
        final_stance_v4 = "support"
    elif directions == {"opposes"}:
        final_stance_v4 = "oppose"
    else:
        final_stance_v4 = "insufficient_evidence"

    old = results_v3.loc[results_v3["case_id"].eq(case_id)].iloc[0]
    probability_raw = old["support_probability_raw"]
    probability_status_raw = clean_text(old["probability_status"])

    probability_consistent = (
        final_stance_v4 == "support"
        and probability_status_raw == "evidence_based"
        and pd.notna(probability_raw)
        and float(probability_raw) >= 0.5
    ) or (
        final_stance_v4 == "oppose"
        and probability_status_raw == "evidence_based"
        and pd.notna(probability_raw)
        and float(probability_raw) < 0.5
    )

    if final_stance_v4 == "insufficient_evidence":
        public_probability = np.nan
        public_probability_status = "not_available"
    elif probability_consistent:
        public_probability = float(probability_raw)
        public_probability_status = "uncalibrated_llm_estimate"
    else:
        public_probability = np.nan
        public_probability_status = "not_available_due_to_direction_conflict"

    case_rows.append({
        "case_id": case_id,
        "final_stance_v3": old["final_stance"],
        "final_stance_v4": final_stance_v4,
        "accepted_direct_evidence": int(len(accepted)),
        "support_probability_raw": probability_raw,
        "public_support_probability": public_probability,
        "public_probability_status": public_probability_status,
        "expected_output": case["expected_output"],
        "decision_correct_v4": final_stance_v4 == case["expected_output"],
        "overclaim_v4": (
            case["expected_evidence_class"] != "direct_evidence"
            and final_stance_v4 in ["support", "oppose"]
        ),
    })

rescored = cases.merge(
    pd.DataFrame(case_rows), on=["case_id", "expected_output"], how="left"
)

display(rescored[[
    "case_id", "party", "expected_evidence_class", "expected_output",
    "final_stance_v3", "final_stance_v4", "accepted_direct_evidence",
    "support_probability_raw", "public_support_probability",
    "public_probability_status", "decision_correct_v4", "overclaim_v4",
]])
''')

add_code(r'''
# 计算 07d 指标，并与 07c 做对照
expected_roles = evidence[[
    "case_id", "evidence_id", "expected_role_normalized"
]]
audit = audit.merge(
    expected_roles,
    on=["case_id", "evidence_id"],
    how="left",
    suffixes=("", "_gold"),
)
audit["effective_role_correct"] = (
    audit["effective_evidence_role"] == audit["expected_role_normalized"]
)
audit["expected_is_directional"] = audit[
    "expected_role_normalized"
].eq("directional_direct")
audit["effective_is_directional"] = audit[
    "effective_evidence_role"
].eq("directional_direct")
audit["directional_safety_correct"] = (
    audit["expected_is_directional"] == audit["effective_is_directional"]
)

direct_mask = rescored["expected_evidence_class"].eq("direct_evidence")
abstain_mask = ~direct_mask

metrics_v4 = {
    "cases_rescored": int(len(rescored)),
    "direct_stance_accuracy": float(
        rescored.loc[direct_mask, "decision_correct_v4"].mean()
    ),
    "abstention_accuracy": float(
        rescored.loc[abstain_mask, "decision_correct_v4"].mean()
    ),
    "overclaim_rate": float(
        rescored.loc[abstain_mask, "overclaim_v4"].mean()
    ),
    "effective_role_accuracy": float(audit["effective_role_correct"].mean()),
    "directional_safety_accuracy": float(
        audit["directional_safety_correct"].mean()
    ),
    "direct_probability_availability": float(
        rescored.loc[direct_mask, "public_support_probability"].notna().mean()
    ),
    "public_probability_contract_rate": float(
        (
            rescored["final_stance_v4"].eq("insufficient_evidence")
            == rescored["public_support_probability"].isna()
        ).mean()
    ),
    "api_calls": API_CALLS,
    "estimated_api_cost_usd": 0.0,
}

metrics_v3 = {
    "direct_stance_accuracy": float(
        results_v3.loc[
            results_v3["expected_evidence_class"].eq("direct_evidence"),
            "decision_correct",
        ].map(native_bool).mean()
    ),
    "abstention_accuracy": float(
        results_v3.loc[
            ~results_v3["expected_evidence_class"].eq("direct_evidence"),
            "decision_correct",
        ].map(native_bool).mean()
    ),
    "overclaim_rate": float(
        results_v3.loc[
            ~results_v3["expected_evidence_class"].eq("direct_evidence"),
            "overclaim",
        ].map(native_bool).mean()
    ),
}

comparison = pd.DataFrame([
    {"version": "07c", **metrics_v3},
    {
        "version": "07d",
        "direct_stance_accuracy": metrics_v4["direct_stance_accuracy"],
        "abstention_accuracy": metrics_v4["abstention_accuracy"],
        "overclaim_rate": metrics_v4["overclaim_rate"],
    },
])

acceptance_gates = {
    "all_14_rescored_gate": metrics_v4["cases_rescored"] == 14,
    "direct_accuracy_gate": metrics_v4["direct_stance_accuracy"] >= 0.75,
    "abstention_gate": metrics_v4["abstention_accuracy"] >= 0.90,
    "overclaim_gate": metrics_v4["overclaim_rate"] <= 0.10,
    "effective_role_accuracy_gate": metrics_v4["effective_role_accuracy"] >= 0.75,
    "directional_safety_gate": metrics_v4["directional_safety_accuracy"] >= 0.95,
    "public_probability_contract_gate": (
        metrics_v4["public_probability_contract_rate"] == 1.0
    ),
    "no_api_calls_gate": API_CALLS == 0 and LLM_CALLS == 0,
}

print("07c vs 07d decision comparison:")
display(comparison.round(3))
print("07d metrics:")
display(pd.DataFrame([metrics_v4]).round(3))
print("Acceptance gates:", acceptance_gates)
''')

add_code(r'''
# 保存重评分结果和运行记录
rescored.to_csv(
    OUTPUT_DIR / "agent_results_rescored_v4.csv", index=False
)
audit.to_csv(
    OUTPUT_DIR / "evidence_gate_audit_v4.csv", index=False
)
comparison.to_csv(
    OUTPUT_DIR / "v3_v4_decision_comparison.csv", index=False
)
pd.DataFrame([metrics_v4]).to_csv(
    OUTPUT_DIR / "evaluation_metrics_v4.csv", index=False
)

manifest = {
    "notebook_version": "07d-source-aware-gate-v4",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "input_version": "07c-historical-contract-agent-v3",
    "api_calls": int(API_CALLS),
    "llm_calls": int(LLM_CALLS),
    "cases_rescored": int(len(rescored)),
    "acceptance_gates": {
        key: bool(value) for key, value in acceptance_gates.items()
    },
    "test_split_used": False,
}
with (OUTPUT_DIR / "run_manifest_v4.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)

print("=== SOURCE-AWARE GATE RESCORING SUMMARY FOR REVIEW ===")
print("Notebook version: 07d-source-aware-gate-v4")
print("Input results: 07c-historical-contract-agent-v3")
print("API calls:", API_CALLS)
print("LLM calls:", LLM_CALLS)
print("Cases rescored:", metrics_v4["cases_rescored"])
print("Direct stance accuracy:", round(metrics_v4["direct_stance_accuracy"], 3))
print("Abstention accuracy:", round(metrics_v4["abstention_accuracy"], 3))
print("Overclaim rate:", round(metrics_v4["overclaim_rate"], 3))
print("Effective evidence role accuracy:", round(metrics_v4["effective_role_accuracy"], 3))
print("Directional safety accuracy:", round(metrics_v4["directional_safety_accuracy"], 3))
print("Direct probability availability:", round(metrics_v4["direct_probability_availability"], 3))
print("Public probability contract rate:", round(metrics_v4["public_probability_contract_rate"], 3))
print("Estimated API cost (USD):", metrics_v4["estimated_api_cost_usd"])
print("Acceptance gates:", acceptance_gates)

if all(bool(value) for value in acceptance_gates.values()):
    print("Next step: 07d passes; freeze the gate rules and proceed to 08 integration.")
else:
    print("Next step: inspect failed gate reasons only; do not run Test or new API calls.")

print("RUN_FINAL_TEST:", RUN_FINAL_TEST)
print("Test result: NOT RUN")
print("Output directory:", OUTPUT_DIR)
print("=== END SOURCE-AWARE GATE RESCORING SUMMARY ===")
''')


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "spatial",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.13",
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
