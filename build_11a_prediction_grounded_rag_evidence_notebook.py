import json
from pathlib import Path


ROOT = Path.cwd()
OUTPUT = ROOT / "11a_prediction_grounded_rag_evidence.ipynb"


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
# 11A — Prediction-Grounded RAG Evidence Layer

## 这一层的职责

最终模型已经冻结并完成Test。11A不会重新训练，也不会修改概率，而是同时回答两个不同问题：

1. **数学解释**：支持率是怎样由角色模型和最近100次投票先验计算出来的；
2. **政策解释**：哪些历史投票、manifesto内容支持、反对或只能作为背景。

正确顺序是：

```text
冻结模型先输出预测与概率
→ RAG中立检索相关历史投票、manifesto和Bill背景
→ 同时保留支持与反对方向的历史证据
→ 检索完成后才比较证据与预测
→ 输出agreement、conflict、context或insufficient
```

预测结果可以帮助Agent解释最终输出，但不能用于检索阶段只寻找“证明预测正确”的材料。

要特别注意：RAG文本不能证明一个精确概率为什么是73%。73%来自冻结模型的数学公式；RAG负责说明现实资料是否与这个预测一致、冲突或证据不足。

**11A不调用LLM、不调用API、不重新读取Test标签。**
''')

md(r'''
## 为什么不能直接按预测方向搜索

如果模型预测“支持”，然后RAG只搜索支持材料，系统几乎一定能找到看起来合理的解释，但无法发现模型可能错了。

因此历史投票采用对称检索：

- 优先保留最相关的一条历史支持证据；
- 优先保留最相关的一条历史反对证据；
- 剩余位置再按相关性填充；
- 整个选择过程不知道当前模型预测是支持还是反对。

只有证据已经固定后，系统才标记它与预测一致还是冲突。

## RAG不是第二个预测模型

历史证据中的`support`或`oppose`只描述该党对**过去那条议案**的立场。即使检索结果全部是反对，也不能直接推出当前议案必然反对，因为两条议案可能在政策范围、法案阶段、年份或政府背景上不同。

因此11A遵守三条产品规则：

- 不按支持证据和反对证据的条数进行多数表决；
- 不让RAG自动覆盖冻结模型；
- 出现冲突时显示“模型—历史类比冲突”，交给用户判断，而不是宣布哪一方一定错误。
''')

md(r'''
## 政策对象缺失时如何处理

最终预测表中很多`final_policy_object`为空，所以不能只依赖该字段。

11A使用以下回退顺序：

1. 已人工确认的`final_policy_object`；
2. 06G从议案原文中提取且有原文引用的`canonical_policy_object`；
3. 若提取置信度较低，仍可用于扩大搜索，但查询被标记为`review_required`；
4. 最后才使用议案标题和原文片段补充语境。

低置信度对象不会被伪装成确定事实。
''')

code(r'''
# 导入本地检索所需库；这里没有OpenAI或其他付费API
import hashlib
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from joblib import load
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from xgboost import DMatrix, XGBClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 180)
sns.set_theme(style="whitegrid")

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "processed"
MODEL_DIR = PROCESSED_DIR / "model_v2"
RAG_DIR = PROCESSED_DIR / "rag_v3"
OBJECT_DIR = PROCESSED_DIR / "policy_object_v6"
FINAL_DIR = PROCESSED_DIR / "final_test_three_role_v1"
OUTPUT_DIR = PROCESSED_DIR / "final_rag_evidence_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = FINAL_DIR / "final_test_predictions_public_v1.csv"
TEST_PATH = MODEL_DIR / "model_test_v2.csv"
CHUNKS_PATH = RAG_DIR / "rag_chunks_v3.csv"
OBJECTS_PATH = OBJECT_DIR / "structured_policy_objects_v6.csv"
PREPROCESSOR_PATH = FINAL_DIR / "frozen_role_preprocessor_v1.joblib"
ROLE_MODEL_PATH = FINAL_DIR / "frozen_role_xgboost_v1.json"
PRIOR_PATH = FINAL_DIR / "frozen_recent_party_prior_v1.json"

EXPECTED_FREEZE_SHA256 = (
    "175a2caaa62756fc0a47545afed76f2b07d233409cee20bafcfeb14d6bddd2fc"
)
IN_SCOPE_PARTIES = ["labour", "conservative", "liberal-democrat"]

MAX_RESULTS = 8
MAX_MANIFESTO_RESULTS = 2
MAX_HISTORICAL_RESULTS = 4
MAX_BILL_RESULTS = 1
MAX_CHUNKS_PER_DOCUMENT = 2

print("Notebook version: 11a-prediction-grounded-rag-evidence-v1")
print("API calls: 0")
print("LLM calls: 0")
print("Test labels read: False")
print("Prediction override allowed: False")
print("Output directory:", OUTPUT_DIR)
''')

md(r'''
## 1. 读取冻结预测、无标签查询字段、政策对象和知识库

Test CSV只读取明确列出的文本与结构字段，不读取任何`target_*`字段。
''')

code(r'''
# 只读取RAG查询需要的Test字段，usecols从源头排除标签
QUERY_FEATURE_COLUMNS = [
    "row_id",
    "division_key",
    "motion_date",
    "party",
    "motion_title_clean",
    "motion_text_clean",
    "legislation_name_clean",
    "motion_type",
    "motion_family",
    "policy_domain_primary",
    "final_object_government_backed",
    "government_backing_known",
    "motion_month",
    "total_possible_members",
    "is_governing_party",
    "is_main_opposition",
    "is_opposition_day",
    "is_amendment",
    "is_new_clause",
    "is_financial_motion",
    "policy_domain_count",
]

predictions = pd.read_csv(PREDICTIONS_PATH, low_memory=False)
query_features = pd.read_csv(
    TEST_PATH,
    usecols=QUERY_FEATURE_COLUMNS,
    low_memory=False,
)
objects = pd.read_csv(OBJECTS_PATH, low_memory=False)
chunks = pd.read_csv(CHUNKS_PATH, low_memory=False)

predictions["motion_date"] = pd.to_datetime(
    predictions["motion_date"], errors="raise"
)
query_features["motion_date"] = pd.to_datetime(
    query_features["motion_date"], errors="raise"
)
objects["motion_date"] = pd.to_datetime(objects["motion_date"], errors="coerce")
chunks["source_date"] = pd.to_datetime(chunks["source_date"], errors="coerce")

assert not any(column.startswith("target_") for column in query_features.columns)
assert predictions["freeze_sha256"].eq(EXPECTED_FREEZE_SHA256).all()
assert set(predictions["party"].unique()) == set(IN_SCOPE_PARTIES)
assert chunks["chunk_id"].is_unique

queries = predictions.merge(
    query_features,
    on=["row_id", "division_key", "motion_date", "party", "motion_title_clean", "motion_family", "policy_domain_primary"],
    how="left",
    validate="one_to_one",
)

object_columns = [
    "division_key",
    "canonical_policy_object",
    "generic_policy_object",
    "final_policy_object",
    "source_quote",
    "source_quote_anchored",
    "extraction_confidence",
    "review_required",
    "review_priority",
]
division_objects = objects[object_columns].drop_duplicates("division_key")
queries = queries.merge(
    division_objects,
    on="division_key",
    how="left",
    validate="many_to_one",
    suffixes=("_prediction", "_object"),
)

chunks["retrieval_text"] = chunks["retrieval_text"].fillna("").astype(str)
chunks["evidence_text"] = chunks["evidence_text"].fillna("").astype(str)
chunks["title"] = chunks["title"].fillna("").astype(str)

print("Frozen prediction queries:", len(queries))
print("Unique divisions:", queries["division_key"].nunique())
print("Knowledge-base chunks:", len(chunks))
display(chunks["source_type"].value_counts().to_frame("chunks"))
''')

md(r'''
## 2. 决定每条查询使用哪个政策对象

该步骤只定义搜索问题，不读取模型是否预测正确。
''')

code(r'''
def safe_text(value):
    # 将缺失值安全转换为空字符串并压缩空白
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_bool(value):
    # CSV中的布尔值可能是字符串；不能直接使用bool("False")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return safe_text(value).lower() in {"true", "1", "yes", "y"}


def resolve_policy_object(row):
    # 优先使用人工最终对象，其次使用有原文锚定的结构化提取
    prediction_object = safe_text(row.get("final_policy_object_prediction"))
    object_final = safe_text(row.get("final_policy_object_object"))
    canonical = safe_text(row.get("canonical_policy_object"))
    generic = safe_text(row.get("generic_policy_object"))
    confidence = safe_text(row.get("extraction_confidence")).lower()
    anchored = parse_bool(row.get("source_quote_anchored"))

    if prediction_object:
        return pd.Series({
            "effective_policy_object": prediction_object,
            "policy_object_source": "frozen_prediction_object",
            "query_confidence": "high",
            "query_review_required": False,
        })
    if object_final:
        return pd.Series({
            "effective_policy_object": object_final,
            "policy_object_source": "reviewed_v6_object",
            "query_confidence": confidence or "medium",
            "query_review_required": parse_bool(row.get("review_required")),
        })
    if canonical and anchored:
        return pd.Series({
            "effective_policy_object": canonical,
            "policy_object_source": "source_anchored_v6_object",
            "query_confidence": confidence or "low",
            "query_review_required": parse_bool(row.get("review_required")),
        })
    return pd.Series({
        "effective_policy_object": generic or safe_text(row.get("motion_title_clean")),
        "policy_object_source": "title_or_generic_fallback",
        "query_confidence": "low",
        "query_review_required": True,
    })


queries = pd.concat(
    [queries, queries.apply(resolve_policy_object, axis=1)],
    axis=1,
)
queries["query_id"] = queries["row_id"] + "__rag_evidence_v1"

assert queries["effective_policy_object"].str.len().gt(0).all()
assert queries["query_id"].is_unique

display(queries["policy_object_source"].value_counts().to_frame("queries"))
display(pd.crosstab(
    queries["query_confidence"], queries["query_review_required"]
))
''')

md(r'''
## 3. 拆解每个概率为什么会出现

最终概率不是LLM写出来的，也不是RAG文本投票得到的。冻结公式是：

```text
最终支持概率 = 50% × 角色模型概率 + 50% × 最近100次该党支持率
```

- **角色模型概率**：XGBoost根据政府/反对党角色、政府是否背书、议案类型、政策领域和程序特征计算；
- **最近100次先验**：该党最近100次有效历史投票中的支持比例，并使用Beta(1,1)平滑；
- **最终概率**：上述两个数的简单平均。

下面读取已经冻结的模型，而不是重新训练。XGBoost的贡献值使用log-odds单位：正值把概率推向支持，负值把概率推向反对。这样可以解释模型内部哪几个特征影响最大。
''')

code(r'''
ROLE_CATEGORICAL_FEATURES = [
    "party_role",
    "final_object_government_backed",
    "government_backing_known",
    "motion_type",
    "motion_family",
    "policy_domain_primary",
    "role_backing_interaction",
]

ROLE_NUMERIC_FEATURES = [
    "motion_month",
    "total_possible_members",
    "is_governing_party",
    "is_main_opposition",
    "is_opposition_day",
    "is_amendment",
    "is_new_clause",
    "is_financial_motion",
    "policy_domain_count",
]


def add_role_features(frame):
    # 完全复用最终测试时的特征处理，不加入政党名称或RAG结果
    result = frame.copy()
    result["role_backing_interaction"] = (
        result["party_role"].fillna("unknown").astype(str)
        + "__"
        + result["final_object_government_backed"]
            .fillna("unknown").astype(str)
    )
    for column in ROLE_CATEGORICAL_FEATURES:
        result[column] = result[column].fillna("__missing__").astype(str)
    for column in ROLE_NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def readable_feature_name(name):
    # 去掉流水线前缀，让最终说明更适合产品界面
    name = str(name)
    name = re.sub(r"^(categorical|numeric)__", "", name)
    return name.replace("_", " ")


def top_driver_text(feature_names, row_values, positive=True, top_n=3):
    # 仅显示绝对影响较大的特征，避免把几十个零贡献特征塞给用户
    pairs = [
        (readable_feature_name(name), float(value))
        for name, value in zip(feature_names, row_values)
        if (value > 0 if positive else value < 0)
        and abs(float(value)) >= 1e-8
    ]
    pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return " | ".join(
        f"{name} ({value:+.3f} log-odds)"
        for name, value in pairs[:top_n]
    )


role_preprocessor = load(PREPROCESSOR_PATH)
role_model = XGBClassifier()
role_model.load_model(ROLE_MODEL_PATH)
prior_manifest = json.loads(PRIOR_PATH.read_text(encoding="utf-8"))

role_input = add_role_features(queries)
role_matrix = role_preprocessor.transform(role_input)
role_feature_names = role_preprocessor.get_feature_names_out()
role_booster = role_model.get_booster()

# 最后一列是base value，其余列是每个特征对log-odds的贡献
role_contributions = role_booster.predict(
    DMatrix(role_matrix), pred_contribs=True
)
recomputed_role_probability = role_booster.predict(DMatrix(role_matrix))

assert role_contributions.shape[1] == len(role_feature_names) + 1
assert np.allclose(
    recomputed_role_probability,
    queries["role_probability"].to_numpy(),
    atol=1e-6,
)

model_explanations = queries[["row_id", "party"]].copy()
model_explanations["role_base_log_odds"] = role_contributions[:, -1]
model_explanations["top_supporting_model_drivers"] = [
    top_driver_text(role_feature_names, row[:-1], positive=True)
    for row in role_contributions
]
model_explanations["top_opposing_model_drivers"] = [
    top_driver_text(role_feature_names, row[:-1], positive=False)
    for row in role_contributions
]
model_explanations["recent_prior_window"] = model_explanations["party"].map(
    prior_manifest["counts"]
).astype(int)
model_explanations["recent_prior_support_votes_estimated"] = (
    model_explanations["party"].map(prior_manifest["rates"]) * 102 - 1
).round().astype(int)
model_explanations["recent_prior_oppose_votes_estimated"] = (
    model_explanations["recent_prior_window"]
    - model_explanations["recent_prior_support_votes_estimated"]
)
model_explanations["probability_formula"] = queries.apply(
    lambda row: (
        f"0.50 × {row['role_probability']:.3f} role model + "
        f"0.50 × {row['recent_prior_probability']:.3f} recent prior "
        f"= {row['support_probability']:.3f} support probability"
    ),
    axis=1,
)

recomputed_blend = (
    0.50 * queries["role_probability"]
    + 0.50 * queries["recent_prior_probability"]
)
assert np.allclose(
    recomputed_blend,
    queries["support_probability"],
    atol=1e-12,
)

display(model_explanations.head(3))
''')

md(r'''
## 4. 规范化政策语言

同义词归一用于改善检索，例如把`VAT`和`value added tax`视为同一概念。程序性短语会被弱化，避免只因为两条记录都出现“second reading”就认为它们政策相似。
''')

code(r'''
SYNONYM_GROUPS = [
    (r"\b(?:value added tax|vat)\b", " vat value added tax "),
    (r"\b(?:private schools?|independent schools?)\b", " private school independent school "),
    (r"\b(?:two child limit|two-child limit|two child cap|two-child cap)\b", " two child limit two child cap "),
    (r"\b(?:general practitioners?|gps?)\b", " gp general practitioner primary care doctor "),
    (r"\b(?:national insurance contributions?|nics?)\b", " national insurance employer contribution nic "),
    (r"\b(?:railways?|railroads?)\b", " railway rail public transport "),
    (r"\b(?:vapes?|vaping)\b", " vape vaping electronic cigarette "),
    (r"\b(?:tobacco|cigarettes?|smoking)\b", " tobacco cigarette smoking "),
]

PROCEDURAL_PATTERNS = [
    r"\bfirst reading\b", r"\bsecond reading\b", r"\bthird reading\b",
    r"\bcommencement\b", r"\bquestion put\b", r"\bstanding order\b",
    r"\bthat the bill be now read\b", r"\bthat the clause stand part\b",
    r"\bnew clause\s*\d*\b", r"\bclause\s*\d+\b",
    r"\bamendment proposed\b", r"\bamendment\b",
    r"\bi beg to move\b", r"\bthe house\b",
]

CONTENT_STOP_WORDS = set(ENGLISH_STOP_WORDS).union({
    "bill", "act", "order", "motion", "reading", "clause", "amendment",
    "question", "house", "commons", "approve", "proposed", "stage",
    "page", "line", "section", "part", "paragraph", "schedule",
})


def normalize_text(value):
    # 统一同义词并删除只表达议会程序的固定短语
    text = safe_text(value).lower().replace("__", " ")
    for pattern in PROCEDURAL_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    for pattern, replacement in SYNONYM_GROUPS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9£%\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(value):
    # 提取实质政策词，排除普通英文停用词和程序词
    return {
        token for token in re.findall(r"[a-z][a-z\-]{2,}", normalize_text(value))
        if token not in CONTENT_STOP_WORDS
    }


def token_coverage(query_value, evidence_value):
    # 计算查询政策词被证据覆盖的比例
    query_tokens = content_tokens(query_value)
    evidence_tokens = content_tokens(evidence_value)
    if not query_tokens or not evidence_tokens:
        return 0.0
    return len(query_tokens & evidence_tokens) / len(query_tokens)


chunks["retrieval_text_normalized"] = (
    chunks["title"].map(normalize_text)
    + " "
    + chunks["retrieval_text"].map(normalize_text)
).str.strip()
assert chunks["retrieval_text_normalized"].str.len().gt(0).all()
''')

md(r'''
## 5. 建立严格时间滚动TF-IDF索引

每条证据的`source_date`必须早于当前议案日期。IDF词频也只使用当时已经存在的材料，避免未来资料影响早期查询。

相同可用语料集合会共享一个索引，以减少重复计算。
''')

code(r'''
def make_vectorizer():
    # 同时使用单词和双词短语，保留政策名称中的连字符
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        norm="l2",
        max_features=100000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{1,}\b",
    )


index_cache = {}
date_to_signature = {}

for query_date in sorted(queries["motion_date"].dropna().unique()):
    query_date = pd.Timestamp(query_date)
    available_indices = tuple(np.flatnonzero(
        (
            chunks["source_date"].notna()
            & chunks["source_date"].lt(query_date)
        ).to_numpy()
    ).tolist())
    signature = hashlib.sha256(
        np.asarray(available_indices, dtype=np.int32).tobytes()
    ).hexdigest()
    date_to_signature[query_date] = signature
    if signature not in index_cache:
        vectorizer = make_vectorizer()
        matrix = vectorizer.fit_transform(
            chunks.iloc[list(available_indices)]["retrieval_text_normalized"]
        )
        index_cache[signature] = {
            "indices": np.asarray(available_indices, dtype=int),
            "vectorizer": vectorizer,
            "matrix": matrix,
        }

rolling_index_audit = pd.DataFrame([
    {
        "signature": signature,
        "available_chunks": len(value["indices"]),
        "vocabulary_size": len(value["vectorizer"].vocabulary_),
    }
    for signature, value in index_cache.items()
])

print("Unique query dates:", queries["motion_date"].nunique())
print("Unique rolling indexes:", len(index_cache))
display(rolling_index_audit.describe().round(1))
''')

md(r'''
## 6. 中立打分与双向历史证据选择

查询文本不包含模型预测方向。历史证据先通过相关性门槛，再分别保留支持和反对方向，最后按分数补足。
''')

code(r'''
# 这些阈值沿用06E经过回归检查的语义检索设置
POLICY_ABS_MIN = 0.030
POLICY_RELATIVE_TO_TOP = 0.55
POLICY_ANCHOR_MIN = 0.08
POLICY_STRONG_BASE = 0.090
HISTORICAL_BASE_MIN = 0.050
HISTORICAL_RELATIVE_TO_TOP = 0.60
HISTORICAL_ANCHOR_MIN = 0.10
HISTORICAL_STRONG_BASE = 0.105
BILL_ABS_MIN = 0.080
BILL_ANCHOR_MIN = 0.12
DOMAIN_MATCH_BOOST = 0.012
ANCHOR_COVERAGE_WEIGHT = 0.120
DIRECT_MATCH_BOOST = 0.180


def build_query_text(row):
    # 查询只使用政策对象、标题、法案名、领域和正文，不使用预测方向
    title = normalize_text(row.get("motion_title_clean"))
    policy_object = normalize_text(row.get("effective_policy_object"))
    legislation = normalize_text(row.get("legislation_name_clean"))
    domain = normalize_text(
        safe_text(row.get("policy_domain_primary")).replace("_", " ")
    )
    motion_excerpt = normalize_text(
        safe_text(row.get("motion_text_clean"))[:1800]
    )
    return " ".join([
        title, title, title,
        policy_object, policy_object, policy_object, policy_object,
        legislation, legislation,
        domain,
        motion_excerpt,
    ]).strip()


def build_query_anchor(row):
    # 锚点使用较短的政策对象、标题和法案名
    return " ".join([
        safe_text(row.get("effective_policy_object")),
        safe_text(row.get("motion_title_clean")),
        safe_text(row.get("legislation_name_clean")),
    ])


def direct_policy_match(row, evidence_title):
    # 至少两个政策词的名称包含关系才算直接名称匹配
    evidence_value = normalize_text(evidence_title)
    if not evidence_value:
        return False
    query_values = [
        normalize_text(row.get("effective_policy_object")),
        normalize_text(row.get("legislation_name_clean")),
        normalize_text(row.get("motion_title_clean")),
    ]
    for query_value in query_values:
        shorter = min(
            len(content_tokens(query_value)),
            len(content_tokens(evidence_value)),
        )
        if shorter >= 2 and (
            query_value in evidence_value or evidence_value in query_value
        ):
            return True
    return False


def score_candidates(query_row):
    # 在查询日期以前的知识库中计算中立相关性分数
    query_date = pd.Timestamp(query_row["motion_date"])
    signature = date_to_signature[query_date]
    rolling = index_cache[signature]
    available = chunks.iloc[rolling["indices"]].copy()

    party_ok = available["party"].isin([query_row["party"], "all"])
    division_ok = available["division_key"].ne(
        query_row["division_key"]
    ).fillna(True)
    local_positions = np.flatnonzero((party_ok & division_ok).to_numpy())
    if not len(local_positions):
        return pd.DataFrame()

    query_vector = rolling["vectorizer"].transform([
        build_query_text(query_row)
    ])
    all_scores = (rolling["matrix"] @ query_vector.T).toarray().ravel()
    candidates = available.iloc[local_positions].copy()
    candidates["base_score"] = all_scores[local_positions]

    anchor = build_query_anchor(query_row)
    candidates["anchor_coverage"] = candidates.apply(
        lambda evidence: token_coverage(
            anchor,
            safe_text(evidence["title"])
            + " "
            + safe_text(evidence["retrieval_text_normalized"])[:1200],
        ),
        axis=1,
    )
    candidates["direct_policy_match"] = candidates["title"].map(
        lambda value: direct_policy_match(query_row, value)
    )
    candidates["same_domain"] = (
        candidates["policy_domain"].fillna("").astype(str)
        .eq(safe_text(query_row.get("policy_domain_primary")))
    )
    candidates["adjusted_score"] = (
        candidates["base_score"]
        + candidates["anchor_coverage"] * ANCHOR_COVERAGE_WEIGHT
        + candidates["direct_policy_match"].astype(float) * DIRECT_MATCH_BOOST
        + np.where(
            candidates["source_type"].eq("historical_vote")
            & candidates["same_domain"],
            DOMAIN_MATCH_BOOST,
            0.0,
        )
    )
    return candidates.sort_values(
        ["adjusted_score", "base_score"], ascending=False
    )


def add_candidate(candidate, selected, document_counts, channel):
    # 限制重复chunk和单一文档占满全部位置
    if any(item["chunk_id"] == candidate["chunk_id"] for item in selected):
        return False
    document_id = candidate["document_id"]
    if document_counts.get(document_id, 0) >= MAX_CHUNKS_PER_DOCUMENT:
        return False
    item = candidate.to_dict()
    item["evidence_channel"] = channel
    selected.append(item)
    document_counts[document_id] = document_counts.get(document_id, 0) + 1
    return True


def select_neutral_evidence(candidates):
    # 选择过程不知道当前预测方向，历史支持和反对被对称处理
    selected = []
    document_counts = {}

    manifesto = candidates[candidates["source_type"].eq("manifesto")].copy()
    if len(manifesto):
        threshold = max(
            POLICY_ABS_MIN,
            manifesto["base_score"].max() * POLICY_RELATIVE_TO_TOP,
        )
        manifesto = manifesto[
            manifesto["base_score"].ge(threshold)
            & (
                manifesto["anchor_coverage"].ge(POLICY_ANCHOR_MIN)
                | manifesto["base_score"].ge(POLICY_STRONG_BASE)
                | manifesto["direct_policy_match"]
            )
        ]
        for _, candidate in manifesto.head(MAX_MANIFESTO_RESULTS).iterrows():
            add_candidate(candidate, selected, document_counts, "manifesto")

    historical = candidates[
        candidates["source_type"].eq("historical_vote")
    ].copy()
    if len(historical):
        relative = historical["adjusted_score"].max() * HISTORICAL_RELATIVE_TO_TOP
        historical = historical[
            historical["adjusted_score"].ge(relative)
            & historical["base_score"].ge(HISTORICAL_BASE_MIN)
            & (
                historical["anchor_coverage"].ge(HISTORICAL_ANCHOR_MIN)
                | (
                    historical["base_score"].ge(HISTORICAL_STRONG_BASE)
                    & historical["anchor_coverage"].ge(0.04)
                )
                | historical["direct_policy_match"]
            )
        ]

        # 先各取最相关的支持和反对历史证据，不参考当前预测
        for stance in ["support", "oppose"]:
            stance_rows = historical[historical["stance_label"].eq(stance)]
            if len(stance_rows):
                add_candidate(
                    stance_rows.iloc[0], selected, document_counts,
                    f"historical_{stance}",
                )
        for _, candidate in historical.iterrows():
            if sum(
                item["source_type"] == "historical_vote" for item in selected
            ) >= MAX_HISTORICAL_RESULTS:
                break
            add_candidate(candidate, selected, document_counts, "historical_fill")

    bills = candidates[candidates["source_type"].eq("bill_reference")].copy()
    bills = bills[
        bills["direct_policy_match"]
        | (
            bills["base_score"].ge(BILL_ABS_MIN)
            & bills["anchor_coverage"].ge(BILL_ANCHOR_MIN)
        )
    ]
    for _, candidate in bills.head(MAX_BILL_RESULTS).iterrows():
        add_candidate(candidate, selected, document_counts, "bill_background")

    return sorted(
        selected,
        key=lambda item: (item["adjusted_score"], item["base_score"]),
        reverse=True,
    )[:MAX_RESULTS]
''')

md(r'''
## 7. 执行975条免费检索并在检索后比较预测方向

`relationship_to_prediction`只在证据已经选择完成后计算，因此不会影响检索排名。
''')

code(r'''
evidence_rows = []
query_rows = []

for query_number, (_, query) in enumerate(
    queries.sort_values(["motion_date", "division_key", "party"]).iterrows(),
    start=1,
):
    candidates = score_candidates(query)
    selected = select_neutral_evidence(candidates)

    supports_prediction = 0
    contradicts_prediction = 0
    for rank, item in enumerate(selected, start=1):
        stance = safe_text(item.get("stance_label"))
        directional_evidence_gate = bool(
            item["source_type"] == "historical_vote"
            and stance in {"support", "oppose"}
            and (
                bool(item["direct_policy_match"])
                or (
                    float(item["anchor_coverage"]) >= 0.18
                    and float(item["base_score"]) >= 0.075
                )
            )
        )
        if directional_evidence_gate:
            relationship = (
                "supports_prediction"
                if stance == query["predicted_stance"]
                else "contradicts_prediction"
            )
            supports_prediction += relationship == "supports_prediction"
            contradicts_prediction += relationship == "contradicts_prediction"
        elif item["source_type"] == "manifesto":
            relationship = "requires_direction_judgment"
        else:
            relationship = "context_only"

        if item["source_type"] == "bill_reference":
            evidence_role = "bill_background_only"
        elif item["source_type"] == "manifesto":
            evidence_role = "party_policy_context_direction_unresolved"
        elif directional_evidence_gate and bool(item["direct_policy_match"]):
            evidence_role = "strong_historical_analogue"
        elif directional_evidence_gate:
            evidence_role = "related_historical_analogue"
        else:
            evidence_role = "context_only"

        evidence_rows.append({
            "query_id": query["query_id"],
            "row_id": query["row_id"],
            "query_division_key": query["division_key"],
            "query_date": query["motion_date"],
            "query_party": query["party"],
            "predicted_stance": query["predicted_stance"],
            "support_probability": query["support_probability"],
            "effective_policy_object": query["effective_policy_object"],
            "query_confidence": query["query_confidence"],
            "rank": rank,
            "source_type": item["source_type"],
            "evidence_channel": item["evidence_channel"],
            "relationship_to_prediction": relationship,
            "evidence_role": evidence_role,
            "evidence_is_current_vote_ground_truth": False,
            "evidence_chunk_id": item["chunk_id"],
            "evidence_document_id": item["document_id"],
            "evidence_title": item["title"],
            "evidence_party": item["party"],
            "evidence_date": item["source_date"],
            "evidence_division_key": item["division_key"],
            "evidence_policy_domain": item["policy_domain"],
            "stance_label": stance,
            "source_url": item["source_url"],
            "page_number": item["page_number"],
            "page_column": item["page_column"],
            "base_score": item["base_score"],
            "adjusted_score": item["adjusted_score"],
            "anchor_coverage": item["anchor_coverage"],
            "direct_policy_match": item["direct_policy_match"],
            "directional_evidence_gate": directional_evidence_gate,
            "evidence_text": item["evidence_text"],
        })

    if supports_prediction and contradicts_prediction:
        evidence_status = "mixed_historical_analogues"
    elif supports_prediction:
        evidence_status = "aligned_historical_analogue_found"
    elif contradicts_prediction:
        evidence_status = "challenging_historical_analogue_found"
    elif any(item["source_type"] == "manifesto" for item in selected):
        evidence_status = "manifesto_or_context_requires_review"
    elif selected:
        evidence_status = "context_only"
    else:
        evidence_status = "insufficient_evidence"

    query_rows.append({
        "query_id": query["query_id"],
        "row_id": query["row_id"],
        "division_key": query["division_key"],
        "motion_date": query["motion_date"],
        "party": query["party"],
        "party_role": query["party_role"],
        "motion_title": query["motion_title_clean"],
        "effective_policy_object": query["effective_policy_object"],
        "policy_object_source": query["policy_object_source"],
        "query_confidence": query["query_confidence"],
        "query_review_required": query["query_review_required"],
        "predicted_stance": query["predicted_stance"],
        "support_probability": query["support_probability"],
        "evidence_results": len(selected),
        "supporting_history": supports_prediction,
        "contradicting_history": contradicts_prediction,
        "manifesto_results": sum(
            item["source_type"] == "manifesto" for item in selected
        ),
        "historical_results": sum(
            item["source_type"] == "historical_vote" for item in selected
        ),
        "bill_results": sum(
            item["source_type"] == "bill_reference" for item in selected
        ),
        "evidence_status": evidence_status,
        "rag_produces_separate_prediction": False,
        "prediction_overridden": False,
    })

    if query_number % 100 == 0:
        print(f"Completed retrieval: {query_number}/{len(queries)}")

evidence = pd.DataFrame(evidence_rows)
query_audit = pd.DataFrame(query_rows)
query_audit = query_audit.merge(
    model_explanations.drop(columns=["party"]),
    on="row_id",
    how="left",
    validate="one_to_one",
)

assert len(query_audit) == len(queries)
assert query_audit["prediction_overridden"].eq(False).all()
assert query_audit["rag_produces_separate_prediction"].eq(False).all()
if len(evidence):
    assert not evidence.duplicated(["query_id", "evidence_chunk_id"]).any()
    assert evidence.groupby("query_id").size().le(MAX_RESULTS).all()
    assert (
        pd.to_datetime(evidence["evidence_date"])
        < pd.to_datetime(evidence["query_date"])
    ).all()
    assert evidence["query_division_key"].ne(
        evidence["evidence_division_key"]
    ).fillna(True).all()
''')

md(r'''
## 8. 生成Agent可直接使用的解释与证据包

每条输出保留冻结概率，并附上最多8条证据ID。Agent后续可以阅读证据正文生成自然语言解释，但必须遵守：

- 先用`probability_formula`说明数字怎样计算；
- 再用模型驱动因素说明哪些结构特征推高或压低角色模型概率；
- 不修改`support_probability`；
- 不按支持/反对证据条数重新投票；
- 不把历史类比称为当前投票的真实标签；
- 冲突证据必须明确展示；
- 只有背景资料时不能声称“证据证明该党支持”；
- 证据不足时输出不确定，而不是编造理由。
''')

code(r'''
def joined_ids(group, relationship=None):
    # 按检索排名连接真实证据ID，不生成不存在的引用
    subset = group
    if relationship is not None:
        subset = subset[subset["relationship_to_prediction"].eq(relationship)]
    return " | ".join(subset.sort_values("rank")["evidence_chunk_id"].astype(str))


packet_rows = []
for _, query in query_audit.iterrows():
    subset = evidence[evidence["query_id"].eq(query["query_id"])]
    packet_rows.append({
        **query.to_dict(),
        "supporting_evidence_ids": joined_ids(
            subset, "supports_prediction"
        ),
        "opposing_evidence_ids": joined_ids(
            subset, "contradicts_prediction"
        ),
        "context_evidence_ids": " | ".join(
            subset[
                subset["relationship_to_prediction"].isin([
                    "context_only", "requires_direction_judgment"
                ])
            ].sort_values("rank")["evidence_chunk_id"].astype(str)
        ),
        "all_evidence_ids": joined_ids(subset),
        "agent_instruction": (
            "First explain the frozen probability formula and its main model "
            "drivers. Then discuss the listed evidence as historical analogues "
            "or background, not as ground truth for the current vote. Present "
            "conflicting evidence explicitly. Never vote by evidence count, "
            "change the frozen probability, or infer stance from bill background."
        ),
    })

agent_packets = pd.DataFrame(packet_rows)

queries.to_csv(OUTPUT_DIR / "final_rag_queries_v1.csv", index=False)
model_explanations.to_csv(
    OUTPUT_DIR / "final_model_probability_explanations_v1.csv", index=False
)
evidence.to_csv(OUTPUT_DIR / "final_rag_evidence_v1.csv", index=False)
query_audit.to_csv(OUTPUT_DIR / "final_rag_query_audit_v1.csv", index=False)
agent_packets.to_csv(
    OUTPUT_DIR / "final_agent_evidence_packets_v1.csv", index=False
)
rolling_index_audit.to_csv(
    OUTPUT_DIR / "rolling_index_audit_v1.csv", index=False
)

print("Agent evidence packets:", len(agent_packets))
print("Evidence rows:", len(evidence))
''')

md(r'''
## 9. 免费结构验收与摘要

这一步只评价检索安全性和覆盖情况，不评价预测准确率，也不运行LLM。
''')

code(r'''
source_counts = (
    evidence["source_type"].value_counts()
    if len(evidence)
    else pd.Series(dtype=int)
)
status_counts = query_audit["evidence_status"].value_counts()

party_summary = query_audit.groupby("party").agg(
    queries=("query_id", "size"),
    mean_evidence=("evidence_results", "mean"),
    queries_with_evidence=("evidence_results", lambda values: (values > 0).mean()),
    queries_with_supporting_history=("supporting_history", lambda values: (values > 0).mean()),
    queries_with_contradicting_history=("contradicting_history", lambda values: (values > 0).mean()),
    review_required_rate=("query_review_required", "mean"),
).reset_index()

structural_gates = {
    "all_predictions_have_packet_gate": bool(
        len(agent_packets) == len(predictions)
    ),
    "prediction_not_overridden_gate": bool(
        agent_packets["prediction_overridden"].eq(False).all()
    ),
    "rag_not_second_predictor_gate": bool(
        agent_packets["rag_produces_separate_prediction"].eq(False).all()
    ),
    "evidence_not_ground_truth_gate": bool(
        len(evidence) == 0
        or evidence["evidence_is_current_vote_ground_truth"].eq(False).all()
    ),
    "temporal_leakage_gate": bool(
        len(evidence) == 0
        or (
            pd.to_datetime(evidence["evidence_date"])
            < pd.to_datetime(evidence["query_date"])
        ).all()
    ),
    "same_division_gate": bool(
        len(evidence) == 0
        or evidence["query_division_key"].ne(
            evidence["evidence_division_key"]
        ).fillna(True).all()
    ),
    "duplicate_evidence_gate": bool(
        len(evidence) == 0
        or not evidence.duplicated([
            "query_id", "evidence_chunk_id"
        ]).any()
    ),
    "maximum_eight_results_gate": bool(
        len(evidence) == 0
        or evidence.groupby("query_id").size().le(MAX_RESULTS).all()
    ),
    "no_target_columns_read_gate": bool(
        not any(column.startswith("target_") for column in query_features.columns)
    ),
    "no_api_calls_gate": True,
    "no_llm_calls_gate": True,
}

manifest = {
    "notebook_version": "11a-prediction-grounded-rag-evidence-v1",
    "freeze_sha256": EXPECTED_FREEZE_SHA256,
    "queries": int(len(agent_packets)),
    "evidence_rows": int(len(evidence)),
    "structural_gates": structural_gates,
    "prediction_override_allowed": False,
    "rag_produces_separate_prediction": False,
    "evidence_count_voting_allowed": False,
    "retrieval_uses_predicted_stance": False,
    "stance_comparison_occurs_after_retrieval": True,
    "probability_formula_verified": True,
    "frozen_role_model_reloaded": True,
    "api_calls": 0,
    "llm_calls": 0,
}
(OUTPUT_DIR / "run_manifest_v1.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

display(party_summary.round(3))
display(status_counts.to_frame("queries"))
display(source_counts.to_frame("evidence_rows"))

summary_lines = [
    "=== PREDICTION-GROUNDED RAG EVIDENCE SUMMARY FOR REVIEW ===",
    "Notebook version: 11a-prediction-grounded-rag-evidence-v1",
    "API calls: 0",
    "LLM calls: 0",
    "Test labels read: False",
    f"Frozen prediction queries: {len(agent_packets)}",
    f"Unique divisions: {agent_packets['division_key'].nunique()}",
    f"Evidence rows: {len(evidence)}",
    "Probability formula verified: True",
    "Frozen role-model feature contributions exported: True",
    f"Mean evidence per query: {query_audit['evidence_results'].mean():.3f}",
    f"Queries with evidence rate: "
    f"{query_audit['evidence_results'].gt(0).mean():.3f}",
    f"Queries requiring policy-object review rate: "
    f"{query_audit['query_review_required'].mean():.3f}",
    f"Retrieval uses predicted stance: False",
    f"Prediction override allowed: False",
    "RAG produces a separate prediction: False",
    "Evidence-count voting allowed: False",
    f"Structural gates: {structural_gates}",
    "Evidence status:",
    status_counts.to_string(),
    "Evidence source counts:",
    source_counts.to_string(),
    "Party evidence coverage:",
    party_summary.round(3).to_string(index=False),
    f"Output directory: {OUTPUT_DIR}",
    "Next step: inspect a fixed evidence sample before any optional LLM explanation calls.",
    "=== END PREDICTION-GROUNDED RAG EVIDENCE SUMMARY ===",
]

summary_text = "\n".join(summary_lines)
print(summary_text)
(OUTPUT_DIR / "prediction_grounded_rag_summary_v1.txt").write_text(
    summary_text, encoding="utf-8"
)
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
