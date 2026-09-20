import json
from pathlib import Path


ROOT = Path(__file__).parent
BASE_08 = json.loads(
    (ROOT / "08_locked_validation_end_to_end_evaluation.ipynb")
    .read_text(encoding="utf-8")
)
BASE_GATE = json.loads(
    (ROOT / "08b1_refined_evidence_gate_diagnostic_v2_1.ipynb")
    .read_text(encoding="utf-8")
)
BASE_SCOPE = json.loads(
    (ROOT / "08b2_instrument_scope_patch_diagnostic_v2_2.ipynb")
    .read_text(encoding="utf-8")
)

OUTPUT = ROOT / "08c2_two_layer_agent_paid_evaluation.ipynb"


def lines(text):
    text = text.strip("\n")
    return [line + "\n" for line in text.splitlines()]


def source_of(notebook, index):
    return "".join(notebook["cells"][index]["source"])


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
# 08c2 — Two-Layer Agent Paid Locked Evaluation

## 这一步评估什么

08c2使用08c1已经锁定的60条全新查询。规则、查询、标签和备用模型阈值均已冻结。

最终产品输出分三层：

1. `evidence_backed_prediction`：通过v2.2安全门控，有直接证据与引用；
2. `model_based_estimate`：RAG不能回答，但制度型模型分数不在0.45–0.55不确定区间；
3. `insufficient_evidence`：两层都不可靠。

第二层不会伪造引用，也不会被计入 Evidence-backed coverage。

第一次运行保持 `RUN_PAID_EVALUATION=False`，只检查Prompt、哈希和费用。明确确认后才运行60次API。
''')

add_markdown(r'''
## 如何理解最终指标

- `Evidence-backed accuracy`：有直接证据的回答是否可靠；
- `Evidence-backed coverage`：多少问题真的找到了足够直接的证据；
- `Model-estimate accuracy`：RAG拒答后，备用模型的估计是否有用；
- `Overall actionable coverage`：两层合计能给出多少答案；
- `Overall accuracy`：所有实际输出的支持/反对中，有多少正确；
- `Fully insufficient rate`：两层都拒答的比例。

不能用第二层的数量冒充RAG覆盖率。
''')

add_code(r'''
# 导入API、结构化输出和评测所需的库
import getpass
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
from IPython.display import display
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import accuracy_score, f1_score

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 180)

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "processed"
INPUT_DIR = PROCESSED_DIR / "two_layer_validation_v1"
FROZEN_RULE_DIR = PROCESSED_DIR / "locked_validation_failure_audit_v2_2"

# 复用08的逐条缓存函数；本Notebook的输入与输出都保存在同一锁定目录
OUTPUT_DIR = INPUT_DIR

QUERY_PATH = INPUT_DIR / "two_layer_query_runtime_v1.csv"
RETRIEVAL_PATH = INPUT_DIR / "two_layer_retrieval_evidence_v1.csv"
PUBLIC_QUERY_PATH = INPUT_DIR / "two_layer_queries_public_v1.csv"
PRIVATE_LABEL_PATH = INPUT_DIR / "two_layer_labels_private_v1.csv"
LOCK_MANIFEST_PATH = INPUT_DIR / "two_layer_lock_manifest_v1.json"
PREPARATION_MANIFEST_PATH = INPUT_DIR / "preparation_manifest_v1.json"
FALLBACK_PATH = INPUT_DIR / "fallback_estimates_public_v1.csv"
FROZEN_RULE_MANIFEST_PATH = FROZEN_RULE_DIR / "run_manifest_v2_2.json"

PARTIES = ["conservative", "green", "labour", "liberal-democrat"]
ELECTION_DATE = pd.Timestamp("2024-07-05")
EXPECTED_DIVISIONS = 15
EXPECTED_QUERIES = 60

# 第一次运行必须保持False
RUN_PAID_EVALUATION = False
PAID_CONFIRMATION = ""
REQUIRED_CONFIRMATION = "我确认运行60条08c双层Agent付费评测"
RETRY_FAILED_CASES = False

MODEL = "gpt-4o-mini-2024-07-18"
MAX_PAID_CALLS = EXPECTED_QUERIES
MAX_OUTPUT_TOKENS = 1200
MAX_EVIDENCE_CHARS = 3000
INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 0.60

print("Notebook version: 08c2-two-layer-paid-evaluation-v1")
print("Paid evaluation enabled:", RUN_PAID_EVALUATION)
print("Maximum paid calls:", MAX_PAID_CALLS)
print("Output directory:", INPUT_DIR)
''')

add_markdown(r'''
## 1. 读取锁定文件并验证哈希

这里会重新计算查询和标签SHA-256，并验证冻结规则清单。任何不一致都会在API调用之前停止。
''')

add_code(r'''
# 统一处理空值、布尔值和JSON类型
def clean_text(value):
    """把缺失值和多余空格转换成安全字符串。"""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def native_bool(value):
    """把常见布尔形式统一转换成Python bool。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean_text(value).lower() in {"true", "1", "yes"}


def json_default(value):
    """把NumPy和Pandas类型转换成JSON原生类型。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dataframe_sha256(frame, sort_columns):
    """对排序后的CSV文本计算稳定SHA-256。"""
    ordered = frame.sort_values(sort_columns).reset_index(drop=True)
    payload = ordered.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


required_paths = [
    QUERY_PATH, RETRIEVAL_PATH, PUBLIC_QUERY_PATH, PRIVATE_LABEL_PATH,
    LOCK_MANIFEST_PATH, PREPARATION_MANIFEST_PATH, FALLBACK_PATH,
    FROZEN_RULE_MANIFEST_PATH,
]
for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(path)

queries = pd.read_csv(QUERY_PATH)
retrieval = pd.read_csv(RETRIEVAL_PATH)
locked_queries = pd.read_csv(PUBLIC_QUERY_PATH)
locked_labels = pd.read_csv(PRIVATE_LABEL_PATH)
fallback_estimates = pd.read_csv(FALLBACK_PATH)
lock_manifest = json.loads(LOCK_MANIFEST_PATH.read_text(encoding="utf-8"))
preparation_manifest = json.loads(
    PREPARATION_MANIFEST_PATH.read_text(encoding="utf-8")
)
frozen_rule_manifest = json.loads(
    FROZEN_RULE_MANIFEST_PATH.read_text(encoding="utf-8")
)

for frame in [
    queries, retrieval, locked_queries, locked_labels, fallback_estimates,
]:
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].fillna("")

for column in ["query_date", "evidence_date"]:
    if column in retrieval.columns:
        retrieval[column] = pd.to_datetime(retrieval[column], errors="coerce")
if "query_date" in queries.columns:
    queries["query_date"] = pd.to_datetime(queries["query_date"], errors="coerce")

current_rule_payload = json.dumps(
    frozen_rule_manifest,
    ensure_ascii=False,
    sort_keys=True,
).encode("utf-8")
current_rule_hash = hashlib.sha256(current_rule_payload).hexdigest()

STRUCTURAL_GATES = {
    "fifteen_divisions_gate": queries["division_key"].nunique() == 15,
    "sixty_queries_gate": len(queries) == 60,
    "fifteen_queries_per_party_gate": (
        queries["party"].value_counts().reindex(PARTIES).eq(15).all()
    ),
    "query_hash_gate": (
        dataframe_sha256(locked_queries, ["query_id"])
        == lock_manifest["locked_queries_sha256"]
    ),
    "label_hash_gate": (
        dataframe_sha256(locked_labels, ["query_id"])
        == lock_manifest["locked_labels_sha256"]
    ),
    "preparation_query_hash_gate": (
        preparation_manifest["locked_queries_sha256"]
        == lock_manifest["locked_queries_sha256"]
    ),
    "frozen_rule_hash_gate": (
        preparation_manifest["frozen_rule_manifest_sha256"]
        == current_rule_hash
    ),
    "frozen_rule_version_gate": (
        frozen_rule_manifest.get("notebook_version")
        == "08b-instrument-scope-patch-v2.2"
    ),
    "fallback_query_coverage_gate": (
        set(fallback_estimates["query_id"]) == set(queries["query_id"])
    ),
    "temporal_leakage_gate": (
        pd.to_datetime(retrieval["evidence_date"], errors="coerce")
        .lt(pd.to_datetime(retrieval["query_date"], errors="coerce"))
        .all()
    ),
    "same_division_gate": not (
        retrieval["evidence_division_key"].map(clean_text).ne("")
        & retrieval["evidence_division_key"].eq(
            retrieval["query_division_key"]
        )
    ).any(),
    "test_split_gate": (
        lock_manifest.get("test_split_used") is False
        and preparation_manifest.get("test_split_used") is False
    ),
}

print("Structural gates:", STRUCTURAL_GATES)
if not all(bool(value) for value in STRUCTURAL_GATES.values()):
    raise ValueError("锁定文件、冻结规则或泄漏检查失败，不应调用API。")
''')

add_markdown(r'''
## 2. LLM只审核证据，不决定最终产品层级

LLM逐条判断证据与当前政策对象的关系。最终是否允许成为直接证据，由冻结的v2.2代码门控决定。

真实标签不会进入Prompt。
''')

agent_contract = source_of(BASE_08, 15)
add_code(agent_contract)

add_markdown(r'''
## 3. 付费上限与明确确认

第一次运行到这里时：

    RUN_PAID_EVALUATION = False

确认60条Prompt和费用后，修改为：

    RUN_PAID_EVALUATION = True
    PAID_CONFIRMATION = "我确认运行60条08c双层Agent付费评测"

成功结果会逐条缓存。重新运行不会重复支付已经成功且哈希相同的查询。
''')

cost_code = source_of(BASE_08, 17)
cost_code = cost_code.replace("80条", "60条").replace(
    "Estimated maximum for 80 calls", "Estimated maximum for 60 calls"
)
add_code(cost_code)

api_code = source_of(BASE_08, 18)
api_code = api_code.replace(
    'locked_agent_results_v1.jsonl', 'two_layer_agent_results_v1.jsonl'
).replace("最多80次", "最多60次")
add_code(api_code)

add_markdown(r'''
## 4. 应用冻结的v2.2证据合同

以下代码来自已经通过开发门槛的v2.2规则。08c不能根据新标签修改这些条件。

特别注意：

- `bill_reference` 永远是背景；
- 历史投票方向直接使用归一化立场；
- Clause编号只在同一Bill内比较；
- 跨年份年度Bill不能直接互推；
- 法定文书优先于普通Amendment识别；
- Manifesto必须明确匹配对象与动作。
''')

gate_helpers = source_of(BASE_GATE, 6)
add_code(gate_helpers)

scope_source = source_of(BASE_SCOPE, 5)
scope_defs = scope_source.split("def scope_patch", 1)[0]
add_code(scope_defs)

add_code(r'''
# 判断查询标题与源文本锚定的政策对象是否存在明显冲突
GENERIC_TITLE_PATTERN = re.compile(
    r"^(delegated legislation|finance bill|financial statement|"
    r"business of the house|programme motion|remaining stages)$",
    flags=re.IGNORECASE,
)


def query_contract_status(title, policy_object):
    """返回查询合同状态和可解释原因。"""
    title = clean_text(title)
    policy_object = clean_text(policy_object)
    if not policy_object:
        return "invalid", "missing_policy_object"
    if GENERIC_TITLE_PATTERN.match(title):
        return "ok", "generic_title_allowed"
    title_tokens = token_set(title)
    object_tokens = token_set(policy_object)
    if (
        len(title_tokens) >= 2
        and len(object_tokens) >= 3
        and not (title_tokens & object_tokens)
    ):
        return "review", "title_object_zero_overlap"
    return "ok", "title_object_consistent"
''')

refined_gate_source = source_of(BASE_GATE, 8)
refined_gate_function = refined_gate_source.split(
    "refined_output =", 1
)[0].replace("scope_level(", "corrected_scope_level(")
add_code(refined_gate_function)

add_markdown(r'''
## 5. 合并LLM审核与确定性门控

LLM可能把相关背景误判为直接证据，因此代码会再次核对来源、编号、动作、年份和政党角色。

历史投票的方向不采用LLM二次翻译，直接读取已归一化的历史立场。
''')

add_code(r'''
# 读取缓存结果，并只保留当前Prompt和锁定哈希对应的最新记录
def load_records(path):
    """读取JSONL缓存。"""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


RESULTS_JSONL = INPUT_DIR / "two_layer_agent_results_v1.jsonl"
records = load_records(RESULTS_JSONL)
current_hashes = {
    query_id: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    for query_id, prompt in prompt_previews.items()
}
valid_records = [
    row for row in records
    if row.get("prompt_sha256") == current_hashes.get(row.get("query_id"))
    and row.get("lock_sha256") == lock_manifest["locked_queries_sha256"]
]

if not valid_records:
    rag_results = pd.DataFrame()
    evidence_audit = pd.DataFrame()
    print("尚无当前锁定集的API结果。")
else:
    raw_records = pd.DataFrame(valid_records)
    raw_records["_order"] = np.arange(len(raw_records))
    latest = raw_records.sort_values("_order").drop_duplicates(
        "query_id", keep="last"
    )

    query_contract_meta = locked_queries[[
        "query_id", "contract_source_anchored", "source_quote",
        "extraction_confidence",
    ]].drop_duplicates("query_id")

    result_rows = []
    audit_rows = []
    for _, record in latest.iterrows():
        query_id = record["query_id"]
        query_matches = queries[queries["query_id"].eq(query_id)]
        if query_matches.empty:
            continue
        query = query_matches.iloc[0]
        meta = query_contract_meta[
            query_contract_meta["query_id"].eq(query_id)
        ].iloc[0]
        output = record.get("output")
        output = output if isinstance(output, dict) else {}
        judgments = output.get("evidence_judgments", [])
        judgments = judgments if isinstance(judgments, list) else []

        expected_ids = retrieval.loc[
            retrieval["query_id"].eq(query_id), "evidence_id"
        ].tolist()
        returned_ids = [
            item.get("evidence_id") for item in judgments
            if isinstance(item, dict)
        ]
        citation_set_valid = (
            len(returned_ids) == len(set(returned_ids))
            and set(returned_ids) == set(expected_ids)
        )

        contract_status, contract_reason = query_contract_status(
            query["motion_title"], query["query_policy_object"]
        )
        accepted_rows = []
        for item in judgments:
            if not isinstance(item, dict):
                continue
            evidence_matches = retrieval[
                retrieval["query_id"].eq(query_id)
                & retrieval["evidence_id"].eq(item.get("evidence_id"))
            ]
            if evidence_matches.empty:
                continue
            combined = {**evidence_matches.iloc[0].to_dict(), **item}
            combined["query_contract_status"] = contract_status
            combined["query_contract_reason"] = contract_reason
            combined["contract_source_anchored"] = meta[
                "contract_source_anchored"
            ]
            combined["source_quote"] = meta["source_quote"]

            # 这是原07d门的最低候选条件；后面仍要通过完整v2.2规则
            combined["accepted_directional_direct"] = (
                clean_text(combined.get("evidence_role"))
                == "directional_direct"
                and clean_text(combined.get("directional_value"))
                in {"supports", "opposes"}
                and clean_text(combined.get("object_relation"))
                in {"exact", "substantively_compatible"}
            )

            accepted, gate_reason, direction = apply_refined_gate(
                pd.Series(combined)
            )
            combined["accepted_directional_direct_v2_2"] = accepted
            combined["gate_reason_v2_2"] = gate_reason
            combined["effective_direction_v2_2"] = direction
            audit_rows.append(combined)
            if accepted:
                accepted_rows.append(combined)

        accepted_directions = {
            row["effective_direction_v2_2"] for row in accepted_rows
            if row["effective_direction_v2_2"] in {"supports", "opposes"}
        }
        if not citation_set_valid or not accepted_directions:
            rag_stance = "insufficient_evidence"
            rag_reason = "no_accepted_directional_evidence"
        elif accepted_directions == {"supports"}:
            rag_stance = "support"
            rag_reason = "all_accepted_evidence_supports"
        elif accepted_directions == {"opposes"}:
            rag_stance = "oppose"
            rag_reason = "all_accepted_evidence_opposes"
        else:
            rag_stance = "insufficient_evidence"
            rag_reason = "conflicting_directional_evidence"

        accepted_ids = [row["evidence_id"] for row in accepted_rows]
        result_rows.append({
            "query_id": query_id,
            "division_key": query["division_key"],
            "party": query["party"],
            "query_date": query["query_date"],
            "motion_title": query["motion_title"],
            "policy_domain_primary": query["policy_domain_primary"],
            "motion_family": query["motion_family"],
            "era": query["era"],
            "extraction_confidence": meta["extraction_confidence"],
            "api_status": record.get("status"),
            "rag_stance": rag_stance,
            "rag_decision_reason": rag_reason,
            "accepted_direct_evidence": len(accepted_rows),
            "accepted_evidence_ids": accepted_ids,
            "citation_id_set_valid": citation_set_valid,
            "llm_proposed_stance": output.get("proposed_stance"),
            "llm_confidence": output.get("confidence"),
            "reasoning_summary": output.get("reasoning_summary"),
            "input_tokens": record.get("input_tokens", 0),
            "output_tokens": record.get("output_tokens", 0),
            "estimated_cost_usd": record.get("estimated_cost_usd", 0.0),
        })

    rag_results = pd.DataFrame(result_rows)
    evidence_audit = pd.DataFrame(audit_rows)
    print("Current completed result rows:", len(rag_results))
    if len(rag_results):
        display(rag_results["rag_stance"].value_counts().to_frame("queries"))
''')

add_markdown(r'''
## 6. 组合双层产品输出

优先级固定：

    RAG有可靠直接证据
    → evidence_backed_prediction

    RAG拒答，但制度模型不在不确定区间
    → model_based_estimate

    两者都不能可靠判断
    → insufficient_evidence

第二层只显示模型分数，不显示伪造的证据ID。
''')

add_code(r'''
if rag_results.empty:
    final_outputs = pd.DataFrame()
    print("没有API结果，因此尚未生成双层输出。")
else:
    final_outputs = rag_results.merge(
        fallback_estimates,
        on=["query_id", "division_key", "party", "motion_family", "policy_domain_primary"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_fallback"),
    )
    rag_answered = final_outputs["rag_stance"].isin(["support", "oppose"])
    fallback_answered = final_outputs["model_estimated_stance"].isin(
        ["support", "oppose"]
    )

    final_outputs["output_tier"] = np.select(
        [rag_answered, ~rag_answered & fallback_answered],
        ["evidence_backed_prediction", "model_based_estimate"],
        default="insufficient_evidence",
    )
    final_outputs["final_stance"] = np.select(
        [rag_answered, ~rag_answered & fallback_answered],
        [final_outputs["rag_stance"], final_outputs["model_estimated_stance"]],
        default="insufficient_evidence",
    )
    final_outputs["public_support_score"] = np.where(
        final_outputs["output_tier"].eq("model_based_estimate"),
        final_outputs["model_support_score"],
        np.nan,
    )
    final_outputs["public_evidence_ids"] = np.where(
        final_outputs["output_tier"].eq("evidence_backed_prediction"),
        final_outputs["accepted_evidence_ids"].astype(str),
        "[]",
    )
    final_outputs["public_disclaimer"] = np.select(
        [
            final_outputs["output_tier"].eq("evidence_backed_prediction"),
            final_outputs["output_tier"].eq("model_based_estimate"),
        ],
        [
            "Evidence-backed prediction using accepted direct evidence.",
            "Model-based estimate only; no accepted direct RAG evidence.",
        ],
        default="Insufficient direct evidence and uncertain model estimate.",
    )

    display(final_outputs["output_tier"].value_counts().to_frame("queries"))
''')

add_markdown(r'''
## 7. 最后才打开私有标签评分

标签只在所有输出层级确定后合并。分别评估 Evidence-backed、Model estimate 和总体表现，防止第二层数量掩盖RAG覆盖不足。
''')

add_code(r'''
if final_outputs.empty or len(final_outputs) != EXPECTED_QUERIES:
    metrics = {}
    acceptance_gates = {}
    print("结果尚未完成60条，因此不打开最终评分。")
else:
    scored = final_outputs.merge(
        locked_labels[["query_id", "target_policy_stance"]],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    scored["answered"] = scored["final_stance"].isin(["support", "oppose"])
    scored["correct"] = np.where(
        scored["answered"],
        scored["final_stance"].eq(scored["target_policy_stance"]),
        np.nan,
    )

    evidence_rows = scored[
        scored["output_tier"].eq("evidence_backed_prediction")
    ]
    model_rows = scored[
        scored["output_tier"].eq("model_based_estimate")
    ]
    actionable = scored[scored["answered"]]
    low_confidence_rows = scored[
        scored["extraction_confidence"].eq("low")
    ]
    low_confidence_actionable = low_confidence_rows[
        low_confidence_rows["answered"]
    ]

    def safe_accuracy(frame):
        if frame.empty:
            return np.nan
        return accuracy_score(frame["target_policy_stance"], frame["final_stance"])


    def safe_macro_f1(frame):
        if frame.empty:
            return np.nan
        return f1_score(
            frame["target_policy_stance"], frame["final_stance"],
            labels=["oppose", "support"], average="macro", zero_division=0,
        )


    # Train party prior仅作为简单基线
    scored["party_prior_stance"] = np.where(
        scored["train_party_support_rate"].ge(0.5), "support", "oppose"
    )
    party_prior_accuracy = accuracy_score(
        scored["target_policy_stance"], scored["party_prior_stance"]
    )

    party_rows = []
    for party, group in scored.groupby("party"):
        party_actionable = group[group["answered"]]
        party_rows.append({
            "party": party,
            "queries": len(group),
            "evidence_backed": int(
                group["output_tier"].eq("evidence_backed_prediction").sum()
            ),
            "model_estimates": int(
                group["output_tier"].eq("model_based_estimate").sum()
            ),
            "insufficient": int(
                group["output_tier"].eq("insufficient_evidence").sum()
            ),
            "actionable_coverage": group["answered"].mean(),
            "actionable_accuracy": safe_accuracy(party_actionable),
        })
    party_metrics = pd.DataFrame(party_rows)

    metrics = {
        "completed_calls": int(scored["api_status"].eq("success").sum()),
        "evidence_backed_rows": len(evidence_rows),
        "evidence_backed_coverage": len(evidence_rows) / len(scored),
        "evidence_backed_accuracy": safe_accuracy(evidence_rows),
        "evidence_backed_macro_f1": safe_macro_f1(evidence_rows),
        "model_estimate_rows": len(model_rows),
        "model_estimate_coverage": len(model_rows) / len(scored),
        "model_estimate_accuracy": safe_accuracy(model_rows),
        "model_estimate_macro_f1": safe_macro_f1(model_rows),
        "overall_actionable_rows": len(actionable),
        "overall_actionable_coverage": len(actionable) / len(scored),
        "overall_accuracy": safe_accuracy(actionable),
        "overall_macro_f1": safe_macro_f1(actionable),
        "fully_insufficient_rate": 1 - len(actionable) / len(scored),
        "low_confidence_queries": len(low_confidence_rows),
        "low_confidence_actionable_coverage": (
            low_confidence_rows["answered"].mean()
            if len(low_confidence_rows) else np.nan
        ),
        "low_confidence_actionable_accuracy": safe_accuracy(
            low_confidence_actionable
        ),
        "party_prior_accuracy_all": party_prior_accuracy,
        "worst_party_actionable_accuracy": float(
            party_metrics["actionable_accuracy"].min()
        ),
        "citation_validity_rate": float(
            scored["citation_id_set_valid"].map(native_bool).mean()
        ),
        "estimated_api_cost_usd": float(scored["estimated_cost_usd"].sum()),
    }

    def at_least(value, threshold):
        return bool(not pd.isna(value) and value >= threshold)


    acceptance_gates = {
        "all_60_completed_gate": metrics["completed_calls"] == 60,
        "minimum_evidence_coverage_gate": (
            metrics["evidence_backed_coverage"] >= 0.10
        ),
        "evidence_accuracy_gate": at_least(
            metrics["evidence_backed_accuracy"], 0.75
        ),
        "model_estimate_accuracy_gate": at_least(
            metrics["model_estimate_accuracy"], 0.60
        ),
        "overall_actionable_coverage_gate": (
            metrics["overall_actionable_coverage"] >= 0.65
        ),
        "overall_accuracy_gate": at_least(metrics["overall_accuracy"], 0.65),
        "overall_macro_f1_gate": at_least(metrics["overall_macro_f1"], 0.60),
        "worst_party_accuracy_gate": at_least(
            metrics["worst_party_actionable_accuracy"], 0.55
        ),
        "citation_validity_gate": metrics["citation_validity_rate"] == 1.0,
        "structural_gates_preserved": all(STRUCTURAL_GATES.values()),
        "test_split_gate": True,
    }

    display(pd.DataFrame([metrics]).round(3))
    display(party_metrics.round(3))
    display(pd.crosstab(
        scored["output_tier"], scored["target_policy_stance"], margins=True
    ))

    final_outputs.to_csv(INPUT_DIR / "two_layer_public_outputs_v1.csv", index=False)
    scored.to_csv(INPUT_DIR / "two_layer_scored_results_v1.csv", index=False)
    evidence_audit.to_csv(INPUT_DIR / "two_layer_evidence_audit_v1.csv", index=False)
    pd.DataFrame([metrics]).to_csv(
        INPUT_DIR / "two_layer_metrics_v1.csv", index=False
    )
    party_metrics.to_csv(
        INPUT_DIR / "two_layer_party_metrics_v1.csv", index=False
    )
''')

add_code(r'''
# 保存运行清单并输出可复制摘要
run_manifest = {
    "notebook_version": "08c2-two-layer-paid-evaluation-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": MODEL,
    "paid_evaluation_enabled": bool(RUN_PAID_EVALUATION),
    "locked_queries_sha256": lock_manifest["locked_queries_sha256"],
    "locked_labels_sha256": lock_manifest["locked_labels_sha256"],
    "frozen_rule_manifest_sha256": current_rule_hash,
    "structural_gates": STRUCTURAL_GATES,
    "acceptance_gates": acceptance_gates,
    "test_split_used": False,
}
with (INPUT_DIR / "two_layer_run_manifest_v1.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(
        run_manifest, handle, ensure_ascii=False, indent=2,
        default=json_default,
    )

print("=== TWO-LAYER AGENT EVALUATION SUMMARY FOR REVIEW ===")
print("Notebook version: 08c2-two-layer-paid-evaluation-v1")
print("Model:", MODEL)
print("Paid evaluation enabled:", RUN_PAID_EVALUATION)
print("Locked queries:", len(queries))
print("Structural gates:", STRUCTURAL_GATES)
if not metrics:
    print("API results: NOT RUN OR INCOMPLETE")
    print("Completed cached results:", len(rag_results))
    print("Next step: inspect prompts and cost, then explicitly enable 60 calls.")
else:
    print("Completed calls:", metrics["completed_calls"])
    print("Evidence-backed rows:", metrics["evidence_backed_rows"])
    print("Evidence-backed coverage:", round(metrics["evidence_backed_coverage"], 3))
    print("Evidence-backed accuracy:", round(metrics["evidence_backed_accuracy"], 3))
    print("Evidence-backed Macro-F1:", round(metrics["evidence_backed_macro_f1"], 3))
    print("Model-estimate rows:", metrics["model_estimate_rows"])
    print("Model-estimate accuracy:", round(metrics["model_estimate_accuracy"], 3))
    print("Model-estimate Macro-F1:", round(metrics["model_estimate_macro_f1"], 3))
    print("Overall actionable coverage:", round(metrics["overall_actionable_coverage"], 3))
    print("Overall accuracy:", round(metrics["overall_accuracy"], 3))
    print("Overall Macro-F1:", round(metrics["overall_macro_f1"], 3))
    print("Fully insufficient rate:", round(metrics["fully_insufficient_rate"], 3))
    print("Low-confidence object queries:", metrics["low_confidence_queries"])
    print("Low-confidence actionable coverage:", round(metrics["low_confidence_actionable_coverage"], 3))
    print("Low-confidence actionable accuracy:", round(metrics["low_confidence_actionable_accuracy"], 3))
    print("Party-prior accuracy on all queries:", round(metrics["party_prior_accuracy_all"], 3))
    print("Worst-party actionable accuracy:", round(metrics["worst_party_actionable_accuracy"], 3))
    print("Citation validity rate:", round(metrics["citation_validity_rate"], 3))
    print("Estimated API cost (USD):", round(metrics["estimated_api_cost_usd"], 6))
    print("Acceptance gates:", acceptance_gates)
    if all(acceptance_gates.values()):
        print("Next step: two-layer locked validation passes; prepare final Test protocol.")
    else:
        print("Next step: locked validation does not pass; do not tune on these 60 cases or run Test.")
print("RUN_FINAL_TEST: False")
print("Test result: NOT RUN")
print("Output directory:", INPUT_DIR)
print("=== END TWO-LAYER AGENT EVALUATION SUMMARY ===")
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
