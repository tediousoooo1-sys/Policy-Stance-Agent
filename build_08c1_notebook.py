import json
from pathlib import Path


BASE_NOTEBOOK = Path(__file__).with_name(
    "08_locked_validation_end_to_end_evaluation.ipynb"
)
OUTPUT = Path(__file__).with_name(
    "08c1_two_layer_agent_locked_preparation.ipynb"
)


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


base = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))


add_markdown(r'''
# 08c1 — Two-Layer Agent Locked Validation Preparation

## 这一步要验证什么

v2.2已经把“有直接证据时乱答”的风险降下来，但 Evidence-backed coverage 只有15%。

08c采用双层产品输出：

    第一层：Evidence-backed prediction
    有通过v2.2门控的直接证据时，输出支持或反对并展示证据。

    第二层：Model-based estimate
    没有直接证据时，使用只在Train训练的制度型模型给出估计，
    并明确标记这不是RAG直接证据。

    两层都不可靠：Insufficient evidence

排除全部旧题后，大选后只剩5个未见 division，因此本次锁定15个 division：
10个大选前高/中置信度对象，加上全部5个大选后对象。大选后2个低置信度对象会被单独标记。

08c1只完成全新样本锁定、免费检索、备用模型估计和费用预览。API调用为0，不使用Test。
''')

add_markdown(r'''
## 为什么第二层不使用RAG语气

RAG检索到“相关背景”不等于检索到“该政党会支持或反对”的直接证据。

因此第二层不会生成虚假的引用，也不会说“根据证据，该党一定支持”。它只根据以下投票前可以知道的结构信息进行估计：

- 当前政党；
- 当前是执政党、主要反对党还是小反对党；
- 政策对象是否由政府背书；
- Motion family；
- Policy domain；
- 月份和少量议会程序特征。

备用模型不读取当前投票结果，也不读取Validation标签。
''')

add_code(r'''
# 导入免费准备阶段需要的库；这里不导入OpenAI，也不会调用API
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option("display.max_colwidth", 180)
pd.set_option("display.max_columns", 180)

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "processed"
MODEL_DIR = PROCESSED_DIR / "model_v2"
RAG_DIR = PROCESSED_DIR / "rag_v3"
OBJECT_DIR = PROCESSED_DIR / "policy_object_v6"
FROZEN_RULE_DIR = PROCESSED_DIR / "locked_validation_failure_audit_v2_2"
OUTPUT_DIR = PROCESSED_DIR / "two_layer_validation_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_PATH = MODEL_DIR / "model_validation_v2.csv"
TRAIN_PATH = MODEL_DIR / "model_train_v2.csv"
CHUNKS_PATH = RAG_DIR / "rag_chunks_v3.csv"
OBJECTS_PATH = OBJECT_DIR / "structured_policy_objects_v6.csv"
FROZEN_RULE_MANIFEST_PATH = FROZEN_RULE_DIR / "run_manifest_v2_2.json"

PARTIES = ["conservative", "green", "labour", "liberal-democrat"]
RANDOM_STATE = 20240921
ELECTION_DATE = pd.Timestamp("2024-07-05")
LOCKED_DIVISIONS = 15
QUERY_COUNT = LOCKED_DIVISIONS * len(PARTIES)

# 第二层模型参数在查看新标签前固定
FALLBACK_C = 0.5
FALLBACK_LOWER_THRESHOLD = 0.45
FALLBACK_UPPER_THRESHOLD = 0.55

# 08c2将继续使用同一小模型；08c1只预估费用
MODEL = "gpt-4o-mini-2024-07-18"
MAX_OUTPUT_TOKENS = 1200
INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 0.60

print("Notebook version: 08c1-two-layer-locked-preparation-v1")
print("API calls: 0")
print("LLM calls: 0")
print("Final Test used: False")
print("Output directory:", OUTPUT_DIR)
''')

add_markdown(r'''
## 1. 读取Train、Validation、知识库与冻结规则

这里会检查v2.2是否真的被标记为可冻结。若冻结清单不存在或没有通过，08c1会停止。
''')

add_code(source_of(base, 3))

add_code(r'''
# 验证08b v2.2冻结候选清单
if not FROZEN_RULE_MANIFEST_PATH.exists():
    raise FileNotFoundError(FROZEN_RULE_MANIFEST_PATH)

frozen_rule_manifest = json.loads(
    FROZEN_RULE_MANIFEST_PATH.read_text(encoding="utf-8")
)
if not frozen_rule_manifest.get("candidate_rules_frozen_for_08c", False):
    raise ValueError("v2.2尚未通过冻结门槛，不应创建08c样本。")
if frozen_rule_manifest.get("test_split_used") is not False:
    raise ValueError("冻结规则清单显示使用过Test，停止运行。")

frozen_rule_payload = json.dumps(
    frozen_rule_manifest,
    ensure_ascii=False,
    sort_keys=True,
).encode("utf-8")
FROZEN_RULE_MANIFEST_SHA256 = hashlib.sha256(
    frozen_rule_payload
).hexdigest()

print("Frozen rule version:", frozen_rule_manifest["notebook_version"])
print("Frozen rule manifest SHA-256:", FROZEN_RULE_MANIFEST_SHA256)
''')

add_markdown(r'''
## 2. 排除所有曾参与开发或调试的 division

除了06–07阶段的案例，这里还会排除08使用过的20个 division。

这样08c面对的是全新议案，而不是再次考试已经看过的题目。
''')

exclusion_code = source_of(base, 5).replace(
    "exclusion_files = [",
    "exclusion_files = [\n"
    "    PROCESSED_DIR / \"locked_validation_v1\" / \"locked_divisions_v1.csv\",",
)
add_code(exclusion_code)

add_markdown(r'''
## 3. 锁定新的15个 division，共60条政党查询

抽样规则在查看标签评分前固定：

- 10个大选前 division；
- 全部5个尚未见过的大选后 division；
- 尽量轮流覆盖不同政策领域；
- 每个 division 对四个政党各建立一条查询；
- 大选前只使用高/中置信度政策对象；
- 大选后允许最多2个有原文锚定的低置信度政策对象，并单独审计；
- 保存查询和标签 SHA-256，后续不允许重新抽样。
''')

lock_code = source_of(base, 7)
replacements = {
    'locked_divisions_v1.csv': 'two_layer_divisions_v1.csv',
    'locked_queries_public_v1.csv': 'two_layer_queries_public_v1.csv',
    'locked_labels_private_v1.csv': 'two_layer_labels_private_v1.csv',
    'lock_manifest_v1.json': 'two_layer_lock_manifest_v1.json',
    '__locked_v1': '__two_layer_v1',
    'locked-validation-v1': 'two-layer-validation-v1',
    'Loaded existing locked evaluation set.': 'Loaded existing two-layer evaluation set.',
    'Created new locked evaluation set.': 'Created new two-layer evaluation set.',
}
for old, new in replacements.items():
    lock_code = lock_code.replace(old, new)

# 新的未见大选后数据只有5个division，其中2个对象为低置信度但有原文锚定。
# 大选前仍只使用高/中置信度；大选后保留全部5个，避免复用旧题或触碰Test。
lock_code = lock_code.replace(
    'division_table["extraction_confidence"].isin(["high", "medium"])',
    'division_table["extraction_confidence"].isin(["high", "medium", "low"])',
)
lock_code = lock_code.replace(
    '''    pre = round_robin_domain_sample(
        eligible[eligible["era"].eq("pre_2024_election")],
        10,
        RANDOM_STATE,
    )
    post = round_robin_domain_sample(
        eligible[eligible["era"].eq("post_2024_election")],
        10,
        RANDOM_STATE + 1,
    )''',
    '''    pre_pool = eligible[
        eligible["era"].eq("pre_2024_election")
        & eligible["extraction_confidence"].isin(["high", "medium"])
    ]
    post_pool = eligible[eligible["era"].eq("post_2024_election")]
    pre = round_robin_domain_sample(pre_pool, 10, RANDOM_STATE)
    post = round_robin_domain_sample(post_pool, 5, RANDOM_STATE + 1)''',
)
add_code(lock_code)

add_markdown(r'''
## 4. 免费、严格时间滚动的RAG检索

检索只允许使用查询日期之前的资料，并排除当前 division 自己。

08c继续使用已经审计过的 TF-IDF rolling index，不在新样本上更换 Dense 权重或检索参数。
''')

add_code(source_of(base, 9))
add_code(source_of(base, 10))

retrieval_code = source_of(base, 11)
retrieval_code = retrieval_code.replace(
    'locked_query_runtime_v1.csv', 'two_layer_query_runtime_v1.csv'
).replace(
    'locked_retrieval_evidence_v1.csv', 'two_layer_retrieval_evidence_v1.csv'
)
add_code(retrieval_code)

add_markdown(r'''
## 5. 免费结构、泄漏和样本锁定检查

任何结构门失败都会停止运行。特别检查：

- 新样本与全部开发样本没有 division 重叠；
- 所有证据严格早于查询日期；
- 没有检索当前 division；
- 没有重复 chunk；
- 每个政党恰好20条；
- Test完全未使用。
''')

structure_code = source_of(base, 13)
structure_code = structure_code.replace(
    'locked_divisions["division_key"].nunique() == 20',
    'locked_divisions["division_key"].nunique() == LOCKED_DIVISIONS',
).replace(
    'len(locked_queries) == 80',
    'len(locked_queries) == QUERY_COUNT',
).replace(
    'locked_queries["party"].value_counts().eq(20).all()',
    'locked_queries["party"].value_counts().eq(LOCKED_DIVISIONS).all()',
).replace(
    '''locked_divisions["era"].value_counts().reindex(
            ["pre_2024_election", "post_2024_election"], fill_value=0
        ).eq(10).all()''',
    '''locked_divisions["era"].value_counts().reindex(
            ["pre_2024_election", "post_2024_election"], fill_value=0
        ).eq(pd.Series({
            "pre_2024_election": 10,
            "post_2024_election": 5,
        })).all()''',
)
structure_code = structure_code.replace(
    '    "test_split_gate": True,',
    '''    "low_confidence_cap_gate": int(
        locked_divisions["extraction_confidence"].eq("low").sum()
    ) <= 2,
    "low_confidence_source_anchored_gate": locked_divisions.loc[
        locked_divisions["extraction_confidence"].eq("low"),
        "contract_source_anchored",
    ].map(native_bool).all(),
    "test_split_gate": True,''',
)
add_code(structure_code)

add_markdown(r'''
## 6. 训练第二层制度型估计模型

这不是RAG，也不提供引用。它只是一个 Logistic Regression：根据制度与议案结构输出支持分数。

为什么选择 Logistic Regression：

- 目标是支持/反对二分类；
- 它可以输出0到1之间的支持分数；
- 比复杂模型更容易解释；
- 类别特征经过 One-hot encoding 后，每个政党、角色和政策领域都有独立权重。

`C=0.5` 使用之前04c的固定正则化强度，不根据08c标签重新选择。0.45–0.55之间视为过于接近五五开，第二层也拒答。
''')

add_code(r'''
# 构造投票前可获得的制度特征
def government_backing_status(frame):
    """把政府背书信息转换成三个可解释类别。"""
    known = pd.to_numeric(
        frame["government_backing_known"], errors="coerce"
    ).fillna(0).astype(int)
    backed = frame["final_object_government_backed"].map(native_bool)
    return np.where(
        known.eq(0),
        "unknown",
        np.where(backed, "government_backed", "not_government_backed"),
    )


train_fallback = train[
    train["target_binary_support"].isin([0, 1])
].copy()

# 从Validation表取新锁定查询的结构特征，但不取标签
fallback_input_columns = [
    "division_key", "party", "party_role", "motion_family",
    "policy_domain_primary", "government_backing_known",
    "final_object_government_backed", "motion_month",
    "is_governing_party", "is_main_opposition", "is_opposition_day",
    "is_amendment", "is_new_clause", "is_financial_motion",
]
fallback_input = locked_queries[[
    "query_id", "division_key", "party",
]].merge(
    validation[fallback_input_columns].drop_duplicates(
        ["division_key", "party"]
    ),
    on=["division_key", "party"],
    how="left",
    validate="one_to_one",
)

train_fallback["government_backing_status"] = government_backing_status(
    train_fallback
)
fallback_input["government_backing_status"] = government_backing_status(
    fallback_input
)

for frame in [train_fallback, fallback_input]:
    frame["role_backing_interaction"] = (
        frame["party_role"].map(clean_text)
        + "__"
        + frame["government_backing_status"].map(clean_text)
    )

CATEGORICAL_FEATURES = [
    "party", "party_role", "government_backing_status",
    "role_backing_interaction", "motion_family", "policy_domain_primary",
]
NUMERIC_FEATURES = [
    "motion_month", "is_governing_party", "is_main_opposition",
    "is_opposition_day", "is_amendment", "is_new_clause",
    "is_financial_motion",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
        (
            "numeric",
            StandardScaler(with_mean=False),
            NUMERIC_FEATURES,
        ),
    ],
    remainder="drop",
)

fallback_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            C=FALLBACK_C,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
    ),
])

fallback_model.fit(
    train_fallback[CATEGORICAL_FEATURES + NUMERIC_FEATURES],
    train_fallback["target_binary_support"].astype(int),
)

fallback_input["model_support_score"] = fallback_model.predict_proba(
    fallback_input[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
)[:, 1]
fallback_input["model_estimated_stance"] = np.select(
    [
        fallback_input["model_support_score"].ge(
            FALLBACK_UPPER_THRESHOLD
        ),
        fallback_input["model_support_score"].le(
            FALLBACK_LOWER_THRESHOLD
        ),
    ],
    ["support", "oppose"],
    default="insufficient_evidence",
)
fallback_input["output_tier_if_rag_abstains"] = np.where(
    fallback_input["model_estimated_stance"].isin(["support", "oppose"]),
    "model_based_estimate",
    "insufficient_evidence",
)

# Party prior只作为比较基线，不作为产品最终解释
party_prior = (
    train_fallback.groupby("party")["target_binary_support"]
    .mean()
    .rename("train_party_support_rate")
)
fallback_input = fallback_input.merge(
    party_prior,
    left_on="party",
    right_index=True,
    how="left",
    validate="many_to_one",
)

fallback_public_columns = [
    "query_id", "division_key", "party", "party_role",
    "government_backing_status", "motion_family",
    "policy_domain_primary", "model_support_score",
    "model_estimated_stance", "output_tier_if_rag_abstains",
    "train_party_support_rate",
]
fallback_estimates = fallback_input[fallback_public_columns].copy()
fallback_estimates.to_csv(
    OUTPUT_DIR / "fallback_estimates_public_v1.csv", index=False
)

print("Fallback estimate distribution")
display(
    fallback_estimates["model_estimated_stance"]
    .value_counts(dropna=False)
    .to_frame("queries")
)
print("Fallback support-score summary")
display(fallback_estimates["model_support_score"].describe().to_frame())
''')

add_markdown(r'''
## 7. 费用预览与冻结清单

费用预览使用字符数近似Token数，只用于设置上限。真正费用会在08c2读取API返回的Token usage重新计算。

这一步不会请求API Key，也不会执行任何付费调用。
''')

add_code(r'''
# 根据查询和检索证据长度估计最多60条调用的费用
if retrieval.empty:
    evidence_chars_by_query = pd.Series(dtype=float)
else:
    evidence_chars_by_query = retrieval.groupby("query_id")[
        "evidence_text"
    ].apply(lambda values: sum(len(clean_text(value)) for value in values))

cost_preview = queries[["query_id", "motion_excerpt"]].copy()
cost_preview["query_chars"] = (
    cost_preview["motion_excerpt"].map(clean_text).str.len()
)
cost_preview["evidence_chars"] = (
    cost_preview["query_id"].map(evidence_chars_by_query).fillna(0)
)
cost_preview["estimated_input_tokens"] = (
    # 额外900 tokens近似系统Prompt和结构化元数据
    (cost_preview["query_chars"] + cost_preview["evidence_chars"]) / 4
    + 900
).round().astype(int)
cost_preview["maximum_output_tokens"] = MAX_OUTPUT_TOKENS
cost_preview["estimated_max_cost_usd"] = (
    cost_preview["estimated_input_tokens"]
    * INPUT_PRICE_PER_MILLION / 1_000_000
    + cost_preview["maximum_output_tokens"]
    * OUTPUT_PRICE_PER_MILLION / 1_000_000
)

estimated_max_cost = float(cost_preview["estimated_max_cost_usd"].sum())
cost_preview.to_csv(OUTPUT_DIR / "api_cost_preview_v1.csv", index=False)

preparation_manifest = {
    "notebook_version": "08c1-two-layer-locked-preparation-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "random_state": RANDOM_STATE,
    "frozen_rule_version": frozen_rule_manifest["notebook_version"],
    "frozen_rule_manifest_sha256": FROZEN_RULE_MANIFEST_SHA256,
    "locked_queries_sha256": lock_manifest["locked_queries_sha256"],
    "locked_labels_sha256": lock_manifest["locked_labels_sha256"],
    "fallback_model": "structured_logistic_regression",
    "fallback_C": FALLBACK_C,
    "fallback_lower_threshold": FALLBACK_LOWER_THRESHOLD,
    "fallback_upper_threshold": FALLBACK_UPPER_THRESHOLD,
    "api_calls": 0,
    "llm_calls": 0,
    "test_split_used": False,
    "structural_gates": STRUCTURAL_GATES,
    "estimated_max_api_cost_usd": estimated_max_cost,
}

with (OUTPUT_DIR / "preparation_manifest_v1.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(
        preparation_manifest,
        handle,
        ensure_ascii=False,
        indent=2,
        default=lambda value: value.item()
        if isinstance(value, np.generic) else str(value),
    )

print("Estimated maximum API cost (USD):", round(estimated_max_cost, 4))
print("API calls made: 0")
''')

add_code(r'''
# 输出可以直接复制回聊天框的免费准备摘要
print("=== TWO-LAYER LOCKED PREPARATION SUMMARY FOR REVIEW ===")
print("Notebook version: 08c1-two-layer-locked-preparation-v1")
print("API calls: 0")
print("LLM calls: 0")
print("Frozen rule version:", frozen_rule_manifest["notebook_version"])
print("Locked divisions:", locked_divisions["division_key"].nunique())
print("Locked party queries:", len(locked_queries))
print("Pre-election divisions:", int(locked_divisions["era"].eq("pre_2024_election").sum()))
print("Post-election divisions:", int(locked_divisions["era"].eq("post_2024_election").sum()))
print("Low-confidence object divisions:", int(locked_divisions["extraction_confidence"].eq("low").sum()))
print("Excluded prior-debug divisions:", len(excluded_divisions))
print("Overlap with prior-debug divisions:", len(overlap_with_debug))
print("Policy domains:", locked_divisions["policy_domain_primary"].nunique())
print("Motion families:", locked_divisions["motion_family"].nunique())
print("Mean evidence per query:", round(queries["retrieved_evidence"].mean(), 3))
print("Queries with no evidence:", int(queries["retrieved_evidence"].eq(0).sum()))
print("Fallback model-estimate rows:", int(fallback_estimates["model_estimated_stance"].isin(["support", "oppose"]).sum()))
print("Fallback uncertain rows:", int(fallback_estimates["model_estimated_stance"].eq("insufficient_evidence").sum()))
print("Locked queries SHA-256:", lock_manifest["locked_queries_sha256"])
print("Locked labels SHA-256:", lock_manifest["locked_labels_sha256"])
print("Frozen rule manifest SHA-256:", FROZEN_RULE_MANIFEST_SHA256)
print("Structural gates:", STRUCTURAL_GATES)
print("Estimated maximum API cost (USD):", round(estimated_max_cost, 4))
print("Paid evaluation: NOT RUN")
print("RUN_FINAL_TEST: False")
print("Test result: NOT RUN")
print("Output directory:", OUTPUT_DIR)
print("Next step: review the new lock, fallback distribution and cost before 08c2 paid evaluation.")
print("=== END TWO-LAYER LOCKED PREPARATION SUMMARY ===")
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
