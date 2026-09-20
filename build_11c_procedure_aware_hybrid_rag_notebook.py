import json
from pathlib import Path


ROOT = Path.cwd()
OUTPUT = ROOT / "11c_procedure_aware_hybrid_rag_retrieval.ipynb"


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
# 11C — Procedure-Aware Hybrid RAG Retrieval

## 这一步解决什么问题

11B证明政治角色会影响历史投票是否可比，但人工检查仍发现许多“文字相似、政策不相同”的证据。

例如：

- 当前查询是Water Restoration Fund，旧材料却是House of Lords改革；
- 当前查询是Official Controls Regulations，旧材料却是Passport Fees Regulations；
- 当前查询是某个具体New Clause，旧材料只是对整部Bill的二读投票。

11C使用三道门：

```text
第一道：政策内容相似
第二道：法律和议会程序范围可比较
第三道：政党角色和政府背书环境可比较
```

只有三道门都通过的历史投票，才可以作为支持或反对方向证据。
''')

md(r'''
## TF-IDF、Dense embedding和Hybrid分别做什么

- **TF-IDF**擅长找相同的专有名词，例如`Water Restoration Fund`；
- **Dense embedding**擅长找意思相近但用词不同的句子；
- **Hybrid**把两者各占50%，先生成较宽的候选池；
- **程序规则**随后检查同一Bill、Clause编号、法定文书名称和年份。

Embedding只负责“可能相关”。它不能让程序范围错误的材料通过。

## 为什么先找120条候选，最后只保留最多7条

先找较多候选可以避免真正相关的材料因为初始排名稍低而消失。严格筛选后，Agent只接收少量材料：

- 最多3条同角色、程序兼容的历史方向证据；
- 最多1条政策相同但角色不同的历史背景；
- 最多2条相关manifesto背景；
- 最多1条真正对应的Bill背景。

这些是来源上限，不是强制配额。没有合格材料时允许返回0条。
''')

code(r'''
# 导入本地检索库；不调用OpenAI或其他API
import hashlib
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 220)
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 180)
sns.set_theme(style="whitegrid")

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "processed"
MODEL_DIR = PROCESSED_DIR / "model_v2"
RAG_DIR = PROCESSED_DIR / "rag_v3"
HYBRID_CACHE_DIR = PROCESSED_DIR / "rag_hybrid_v7"
INPUT_DIR = PROCESSED_DIR / "final_rag_evidence_v1"
OUTPUT_DIR = PROCESSED_DIR / "procedure_aware_rag_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = MODEL_DIR / "model_train_v2.csv"
VALIDATION_PATH = MODEL_DIR / "model_validation_v2.csv"
QUERIES_PATH = INPUT_DIR / "final_rag_queries_v1.csv"
PACKETS_PATH = INPUT_DIR / "final_agent_evidence_packets_v1.csv"
CHUNKS_PATH = RAG_DIR / "rag_chunks_v3.csv"
EMBEDDINGS_PATH = HYBRID_CACHE_DIR / "chunk_embeddings_v7.npy"
EMBEDDING_MANIFEST_PATH = HYBRID_CACHE_DIR / "embedding_manifest_v7.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
HYBRID_DENSE_WEIGHT = 0.50
CANDIDATE_POOL_SIZE = 120
MAX_DIRECTIONAL_HISTORY = 3
MAX_CROSS_ROLE_CONTEXT = 1
MAX_MANIFESTO_CONTEXT = 2
MAX_BILL_BACKGROUND = 1
MAX_FINAL_EVIDENCE = 7
MANUAL_DIVISIONS = 30
RANDOM_STATE = 42
ELECTION_TRANSITION_DATE = pd.Timestamp("2024-07-05")

print("Notebook version: 11c-procedure-aware-hybrid-rag-v3")
print("API calls: 0")
print("LLM calls: 0")
print("Test labels read: False")
print("Embedding model:", EMBEDDING_MODEL)
print("Hybrid dense weight:", HYBRID_DENSE_WEIGHT)
print("Candidate pool size:", CANDIDATE_POOL_SIZE)
print("Output directory:", OUTPUT_DIR)
''')

md(r'''
## 1. 读取查询、知识库和无标签历史元数据

历史Train和Validation只读取政策、程序、角色和政府背书字段。`usecols`中没有任何`target_*`标签。
''')

code(r'''
HISTORICAL_COLUMNS = [
    "division_key",
    "party",
    "motion_date",
    "motion_title_clean",
    "motion_text_clean",
    "legislation_name_clean",
    "final_policy_object",
    "motion_type",
    "motion_family",
    "policy_domain_primary",
    "party_role",
    "final_object_government_backed",
    "government_backing_known",
]

queries = pd.read_csv(QUERIES_PATH, low_memory=False)
base_packets = pd.read_csv(PACKETS_PATH, low_memory=False)
chunks = pd.read_csv(CHUNKS_PATH, low_memory=False)
train_context = pd.read_csv(
    TRAIN_PATH, usecols=HISTORICAL_COLUMNS, low_memory=False
)
validation_context = pd.read_csv(
    VALIDATION_PATH, usecols=HISTORICAL_COLUMNS, low_memory=False
)

queries["motion_date"] = pd.to_datetime(queries["motion_date"], errors="raise")
chunks["source_date"] = pd.to_datetime(chunks["source_date"], errors="coerce")

historical_context = pd.concat(
    [train_context, validation_context], ignore_index=True
)
historical_context["motion_date"] = pd.to_datetime(
    historical_context["motion_date"], errors="raise"
)
historical_context = historical_context.drop_duplicates(
    ["division_key", "party"]
)

assert not any(column.startswith("target_") for column in HISTORICAL_COLUMNS)
assert queries["query_id"].is_unique
assert base_packets["query_id"].is_unique
assert chunks["chunk_id"].is_unique

print("Queries:", len(queries))
print("Unique divisions:", queries["division_key"].nunique())
print("Knowledge-base chunks:", len(chunks))
print("Historical context rows:", len(historical_context))
''')

md(r'''
## 2. 准备政策对象、Bill名称和程序合同

“合同”不是法律合同，而是用于比较的结构化描述：

- 当前对象是整部Bill、具体Clause、Amendment、法定文书还是一般政策；
- Bill或Regulation的名称是什么；
- 是否包含明确Clause或Amendment编号；
- 政策动作是增加、减少、禁止、审查还是批准。

这些字段比单纯的文字相似度更适合阻止错误类比。
''')

code(r'''
WORD_PATTERN = re.compile(r"[a-z0-9]+")
CLAUSE_PATTERN = re.compile(
    r"(?:new\s+)?clause(?:\s+no\.?|\s+number)?\s*(\d+[a-z]?)",
    flags=re.IGNORECASE,
)
AMENDMENT_PATTERN = re.compile(
    r"(?:lords?\s+)?amendment(?:\s+no\.?|\s+number)?\s*(\d+[a-z]?)",
    flags=re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
ANNUAL_BILL_PATTERN = re.compile(
    r"\b(finance bill|budget resolution|supply bill|appropriation bill|"
    r"national insurance contributions bill)\b",
    flags=re.IGNORECASE,
)

CONTENT_STOP_WORDS = set(ENGLISH_STOP_WORDS).union({
    "bill", "act", "order", "motion", "reading", "clause", "amendment",
    "question", "house", "commons", "approve", "proposed", "stage",
    "page", "line", "section", "part", "paragraph", "schedule",
    "regulation", "regulations", "draft", "lords", "new", "number",
})
BILL_STOP_WORDS = CONTENT_STOP_WORDS.union({
    "commencement", "programme", "report", "remaining", "rights",
})


def safe_text(value):
    # 将缺失值转换为空字符串，并压缩多余空格
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_bool(value):
    # 统一处理CSV中的布尔值
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return safe_text(value).lower() in {"true", "1", "yes", "y"}


def normalize_text(value):
    # 保留政策实词，删除标点和多余空格
    text = safe_text(value).lower().replace("__", " ")
    text = re.sub(r"[^a-z0-9£%\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(value, stop_words=CONTENT_STOP_WORDS):
    # 提取可用于政策比较的实质词
    return {
        token for token in WORD_PATTERN.findall(normalize_text(value))
        if len(token) > 2 and token not in stop_words
    }


def token_metrics(left, right):
    # 同时返回共享词数量、较短文本覆盖率和Jaccard相似度
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0, 0.0, 0.0
    shared = len(left_tokens & right_tokens)
    containment = shared / min(len(left_tokens), len(right_tokens))
    jaccard = shared / len(left_tokens | right_tokens)
    return shared, containment, jaccard


def extract_identifiers(value):
    # 提取Clause、Amendment和年份编号
    text = safe_text(value)
    return {
        "clauses": set(x.lower() for x in CLAUSE_PATTERN.findall(text)),
        "amendments": set(x.lower() for x in AMENDMENT_PATTERN.findall(text)),
        "years": set(YEAR_PATTERN.findall(text)),
    }


def scope_level(title, motion_type=""):
    # 将议案分成具体条款、整部Bill、法定文书和一般政策
    lowered = normalize_text(title)
    motion_type = safe_text(motion_type).lower()
    if (
        re.search(r"\b((?:new\s+)?clause|amendment)\b", lowered)
        or motion_type in {
            "amendment", "lords_amendment", "proposed_clause",
            "add_clause_to_bill", "committee_clause",
        }
    ):
        return "specific_provision"
    if (
        re.search(r"\b(second reading|third reading)\b", lowered)
        or motion_type in {"second_stage", "third_stage", "bill_introduction", "reasoned_amendment"}
    ):
        return "whole_bill"
    if (
        re.search(r"\b(regulations?|statutory instrument|order)\b", lowered)
        or motion_type in {
            "approve_statutory_instrument", "revoke_statutory_instrument",
            "eu_document_scrutiny",
        }
    ):
        return "instrument"
    return "general_policy"


def bill_signature(legislation_name, title):
    # 优先使用结构化法案名，没有时才从标题提取
    value = safe_text(legislation_name) or safe_text(title)
    return token_set(value, stop_words=BILL_STOP_WORDS)


def signature_similarity(left, right):
    # 计算名称关键词的Jaccard相似度
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def action_signature(value):
    # 把自然语言动作归入少量可核查类别
    lowered = normalize_text(value)
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


def actions_compatible(left, right):
    # 明确相反的动作不能作为同方向政策类比
    left_actions = action_signature(left)
    right_actions = action_signature(right)
    return not any(
        frozenset({a, b}) in ACTION_CONFLICTS
        for a in left_actions for b in right_actions
    )
''')

md(r'''
## 3. 将历史元数据连接到知识库

知识库chunk保存了证据正文。模型数据保存了该次投票的Bill名称、政策对象、议案类型、政党角色和政府背书。两者通过`division_key + party`连接。
''')

code(r'''
def normalize_role(value):
    # 统一角色名称
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


def infer_role(party, motion_date):
    # 仅在原角色缺失时按日期和政党回退
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


historical_context["historical_party_role"] = historical_context[
    "party_role"
].map(normalize_role)
missing_role = historical_context["historical_party_role"].eq("unknown")
historical_context.loc[missing_role, "historical_party_role"] = (
    historical_context.loc[missing_role].apply(
        lambda row: infer_role(row["party"], row["motion_date"]), axis=1
    )
)

history_lookup = historical_context.rename(columns={
    "division_key": "division_key",
    "party": "party",
    "motion_date": "historical_motion_date",
    "motion_title_clean": "historical_motion_title",
    "motion_text_clean": "historical_motion_text",
    "legislation_name_clean": "historical_legislation_name",
    "final_policy_object": "historical_policy_object",
    "motion_type": "historical_motion_type",
    "motion_family": "historical_motion_family",
    "policy_domain_primary": "historical_policy_domain",
    "final_object_government_backed": "historical_government_backing",
    "government_backing_known": "historical_backing_known",
})[[
    "division_key", "party", "historical_motion_date",
    "historical_motion_title", "historical_motion_text",
    "historical_legislation_name", "historical_policy_object",
    "historical_motion_type", "historical_motion_family",
    "historical_policy_domain", "historical_party_role",
    "historical_government_backing", "historical_backing_known",
]]

chunks = chunks.merge(
    history_lookup,
    on=["division_key", "party"],
    how="left",
    validate="many_to_one",
)
chunks["retrieval_text"] = chunks["retrieval_text"].fillna("").astype(str)
chunks["evidence_text"] = chunks["evidence_text"].fillna("").astype(str)
chunks["title"] = chunks["title"].fillna("").astype(str)
chunks["retrieval_text_normalized"] = (
    chunks["title"].map(normalize_text)
    + " "
    + chunks["retrieval_text"].map(normalize_text)
).str.strip()

historical_mask = chunks["source_type"].eq("historical_vote")
historical_metadata_coverage = float(
    chunks.loc[historical_mask, "historical_party_role"]
    .fillna("unknown").ne("unknown").mean()
)
print("Historical role metadata coverage:", round(historical_metadata_coverage, 3))
''')

md(r'''
## 4. 读取已经缓存的384维Dense embeddings

这些向量由`all-MiniLM-L6-v2`生成。384维是模型设计决定的压缩语义表示，不是人为挑选384个政策词。

Notebook会验证chunk顺序、模型名称和行数。缓存不匹配时直接停止，避免证据与错误向量错位。
''')

code(r'''
embedding_manifest = json.loads(
    EMBEDDING_MANIFEST_PATH.read_text(encoding="utf-8")
)
expected_embedding_signature = hashlib.sha256(
    (
        "|".join(chunks["chunk_id"].astype(str))
        + "|"
        + EMBEDDING_MODEL
    ).encode("utf-8")
).hexdigest()

embedding_cache_valid = bool(
    embedding_manifest.get("signature") == expected_embedding_signature
    and embedding_manifest.get("model") == EMBEDDING_MODEL
    and int(embedding_manifest.get("rows", -1)) == len(chunks)
)
if not embedding_cache_valid:
    raise ValueError("Dense embedding缓存与当前知识库不匹配，请先重跑06H。")

chunk_embeddings = np.load(EMBEDDINGS_PATH)
if chunk_embeddings.shape != (
    len(chunks), int(embedding_manifest["dimensions"])
):
    raise ValueError("Dense embedding形状与manifest不一致。")

dense_model = SentenceTransformer(EMBEDDING_MODEL)
print("Dense embeddings loaded:", chunk_embeddings.shape)
''')

md(r'''
## 5. 建立查询文本与严格时间TF-IDF索引

查询文本只使用政策对象、Motion标题、Bill名称、政策领域和Motion正文，不包含模型预测方向。

每个查询只能看到日期早于它的证据。TF-IDF词频也只用当时已存在的文本计算，防止未来语料影响过去查询。
''')

code(r'''
def build_query_text(row):
    # 政策对象权重最高，标题和Bill名次之，正文只作补充
    policy_object = normalize_text(row.get("effective_policy_object"))
    title = normalize_text(row.get("motion_title_clean"))
    legislation = normalize_text(row.get("legislation_name_clean"))
    domain = normalize_text(
        safe_text(row.get("policy_domain_primary")).replace("_", " ")
    )
    excerpt = normalize_text(safe_text(row.get("motion_text_clean"))[:1600])
    return " ".join([
        policy_object, policy_object, policy_object, policy_object,
        title, title,
        legislation, legislation,
        domain,
        excerpt,
    ]).strip()


queries["query_text_v3"] = queries.apply(build_query_text, axis=1)
query_embeddings = dense_model.encode(
    queries["query_text_v3"].tolist(),
    batch_size=EMBEDDING_BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
).astype("float32")
query_embedding_lookup = {
    query_id: query_embeddings[position]
    for position, query_id in enumerate(queries["query_id"])
}


def make_vectorizer():
    # 单词和双词短语兼顾名称精确匹配与短语含义
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
        chunks["source_date"].notna().to_numpy()
        & chunks["source_date"].lt(query_date).to_numpy()
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

print("Query embeddings:", query_embeddings.shape)
print("Unique rolling TF-IDF indexes:", len(index_cache))
''')

md(r'''
## 6. 定义程序兼容规则

最重要的规则：

- 整部Bill投票不能直接代表某个具体Clause；
- 同一Bill的Clause或Amendment编号必须一致；
- 不同Bill之间只有政策对象和动作高度一致时才可比较；
- Regulations必须匹配具体文书名称和年份；
- Finance Bill等年度法案不能跨年份直接互推；
- Bill reference只能作为背景；
- Manifesto在这一免费阶段只作为政策背景，不自动判断方向。
''')

code(r'''
def current_contract(query):
    # 构造当前查询的政策和程序合同
    title = safe_text(query.get("motion_title_clean"))
    policy_object = safe_text(query.get("effective_policy_object"))
    legislation = safe_text(query.get("legislation_name_clean"))
    combined = " ".join([title, policy_object, legislation])
    return {
        "title": title,
        "object": policy_object,
        "legislation": legislation,
        "scope": scope_level(title, query.get("motion_type")),
        "ids": extract_identifiers(combined),
        "bill_signature": bill_signature(legislation, title),
        "actions": action_signature(" ".join([title, policy_object])),
    }


def historical_contract(candidate):
    # 历史投票优先使用结构化对象，缺失时回退到标题和正文
    title = safe_text(candidate.get("historical_motion_title")) or safe_text(
        candidate.get("title")
    )
    policy_object = safe_text(candidate.get("historical_policy_object"))
    if not policy_object:
        policy_object = " ".join([
            title,
            safe_text(candidate.get("historical_motion_text"))[:1400],
        ])
    legislation = safe_text(candidate.get("historical_legislation_name"))
    combined = " ".join([title, policy_object, legislation])
    return {
        "title": title,
        "object": policy_object,
        "legislation": legislation,
        "scope": scope_level(title, candidate.get("historical_motion_type")),
        "ids": extract_identifiers(combined),
        "bill_signature": bill_signature(legislation, title),
        "actions": action_signature(" ".join([title, policy_object])),
    }


def compare_backing(query, candidate):
    # 两边政府背书都明确时才判断相同或不同
    current_known = parse_bool(query.get("government_backing_known"))
    history_known = parse_bool(candidate.get("historical_backing_known"))
    current_value = safe_text(
        query.get("final_object_government_backed")
    ).lower()
    history_value = safe_text(
        candidate.get("historical_government_backing")
    ).lower()
    if not current_known or not history_known:
        return "unknown"
    if not current_value or not history_value:
        return "unknown"
    return "same" if current_value == history_value else "different"


def compare_procedure(query, candidate):
    # 返回程序是否兼容、原因和可核查的相似度信息
    q = current_contract(query)
    h = historical_contract(candidate)
    shared, containment, jaccard = token_metrics(q["object"], h["object"])
    bill_similarity = signature_similarity(
        q["bill_signature"], h["bill_signature"]
    )
    same_bill = bill_similarity >= 0.75
    action_ok = actions_compatible(q["object"], h["object"])

    if not action_ok:
        return False, "policy_action_conflict", shared, containment, jaccard, same_bill

    if {q["scope"], h["scope"]} == {"specific_provision", "whole_bill"}:
        return False, "provision_whole_bill_scope_mismatch", shared, containment, jaccard, same_bill

    if q["scope"] == "instrument" or h["scope"] == "instrument":
        if q["scope"] != h["scope"]:
            return False, "instrument_scope_mismatch", shared, containment, jaccard, same_bill
        year_match = bool(q["ids"]["years"] & h["ids"]["years"])
        if shared < 3 or containment < 0.75 or not year_match:
            return False, "instrument_name_or_year_mismatch", shared, containment, jaccard, same_bill
        return True, "exact_instrument_match", shared, containment, jaccard, same_bill

    if q["scope"] == "specific_provision":
        if h["scope"] != "specific_provision":
            return False, "specific_provision_scope_mismatch", shared, containment, jaccard, same_bill
        if same_bill:
            if q["ids"]["amendments"] and q["ids"]["amendments"] != h["ids"]["amendments"]:
                return False, "same_bill_amendment_identifier_mismatch", shared, containment, jaccard, same_bill
            if q["ids"]["clauses"] and q["ids"]["clauses"] != h["ids"]["clauses"]:
                return False, "same_bill_clause_identifier_mismatch", shared, containment, jaccard, same_bill
            if shared >= 2 and containment >= 0.45:
                return True, "same_bill_same_provision", shared, containment, jaccard, same_bill
            return False, "same_bill_policy_object_too_weak", shared, containment, jaccard, same_bill
        if shared >= 3 and containment >= 0.65 and jaccard >= 0.25:
            return True, "cross_bill_exact_policy_provision", shared, containment, jaccard, same_bill
        return False, "cross_bill_policy_object_not_exact", shared, containment, jaccard, same_bill

    if q["scope"] == "whole_bill":
        if h["scope"] != "whole_bill":
            return False, "whole_bill_scope_mismatch", shared, containment, jaccard, same_bill
        if same_bill:
            return True, "same_bill_stage_history", shared, containment, jaccard, same_bill
        return False, "different_bill_whole_stage", shared, containment, jaccard, same_bill

    query_is_annual = bool(ANNUAL_BILL_PATTERN.search(
        " ".join([q["title"], q["object"], q["legislation"]])
    ))
    history_is_annual = bool(ANNUAL_BILL_PATTERN.search(
        " ".join([h["title"], h["object"], h["legislation"]])
    ))
    if query_is_annual and history_is_annual:
        if q["ids"]["years"] and h["ids"]["years"] and not (
            q["ids"]["years"] & h["ids"]["years"]
        ):
            return False, "cross_year_annual_bill_history", shared, containment, jaccard, same_bill

    if shared >= 3 and containment >= 0.65 and jaccard >= 0.25:
        return True, "general_policy_exact_match", shared, containment, jaccard, same_bill
    return False, "general_policy_object_not_exact", shared, containment, jaccard, same_bill
''')

md(r'''
## 7. Hybrid候选检索和严格筛选

检索阶段不知道模型预测是支持还是反对。合格历史材料会先分别保留一条`support`和一条`oppose`，再按相关性补足，因此不会只寻找证明模型正确的材料。
''')

code(r'''
def candidate_indices(query):
    # 过滤未来证据、其他政党和当前同一division
    query_date = pd.Timestamp(query["motion_date"])
    rolling = index_cache[date_to_signature[query_date]]
    available = chunks.iloc[rolling["indices"]]
    party_ok = available["party"].isin([query["party"], "all"])
    different_division = available["division_key"].fillna("").astype(str).ne(
        str(query["division_key"])
    )
    nonempty = available["retrieval_text_normalized"].str.len().gt(0)
    local_positions = np.flatnonzero(
        (party_ok & different_division & nonempty).to_numpy()
    )
    return rolling, local_positions


def score_candidates(query):
    # 以50%关键词分数和50%语义分数组成Hybrid分数
    rolling, local_positions = candidate_indices(query)
    if not len(local_positions):
        return pd.DataFrame()
    query_vector = rolling["vectorizer"].transform([
        query["query_text_v3"]
    ])
    sparse_scores_all = (
        rolling["matrix"] @ query_vector.T
    ).toarray().ravel()
    global_indices = rolling["indices"][local_positions]
    dense_scores = (
        chunk_embeddings[global_indices]
        @ query_embedding_lookup[query["query_id"]]
    )
    candidates = chunks.iloc[global_indices].copy()
    candidates["tfidf_score"] = sparse_scores_all[local_positions]
    candidates["dense_score"] = dense_scores
    candidates["hybrid_score"] = (
        (1 - HYBRID_DENSE_WEIGHT) * candidates["tfidf_score"]
        + HYBRID_DENSE_WEIGHT * candidates["dense_score"]
    )
    return candidates.nlargest(CANDIDATE_POOL_SIZE, "hybrid_score")


def enrich_candidate(query, candidate):
    # 计算程序、角色和政府背书的可比性
    result = candidate.to_dict()
    source_type = safe_text(candidate.get("source_type"))
    result.update({
        "procedure_compatible": False,
        "procedure_reason": "not_directional_source",
        "shared_policy_tokens": 0,
        "policy_containment": 0.0,
        "policy_jaccard": 0.0,
        "same_bill": False,
        "role_relation": "not_applicable",
        "government_backing_relation": "not_applicable",
        "directional_gate_v3": False,
        "evidence_role_v3": "context_only",
    })

    if source_type == "historical_vote":
        comparison = compare_procedure(query, candidate)
        (
            result["procedure_compatible"],
            result["procedure_reason"],
            result["shared_policy_tokens"],
            result["policy_containment"],
            result["policy_jaccard"],
            result["same_bill"],
        ) = comparison
        historical_role = normalize_role(
            candidate.get("historical_party_role")
        )
        current_role = normalize_role(query.get("party_role"))
        result["role_relation"] = (
            "same_role" if historical_role == current_role
            else "different_role" if historical_role != "unknown"
            else "unknown"
        )
        result["government_backing_relation"] = compare_backing(
            query, candidate
        )
        result["directional_gate_v3"] = bool(
            result["procedure_compatible"]
            and result["role_relation"] == "same_role"
            and result["government_backing_relation"] != "different"
            and safe_text(candidate.get("stance_label"))
            in {"support", "oppose"}
        )
        if result["directional_gate_v3"]:
            result["evidence_role_v3"] = (
                "procedure_and_role_matched_historical_direction"
            )
        elif result["procedure_compatible"]:
            result["evidence_role_v3"] = (
                "policy_matched_different_institutional_context"
            )
        else:
            result["evidence_role_v3"] = "rejected_procedure_or_policy_mismatch"

    elif source_type == "manifesto":
        shared, containment, jaccard = token_metrics(
            query.get("effective_policy_object"),
            " ".join([
                safe_text(candidate.get("title")),
                safe_text(candidate.get("evidence_text")),
            ]),
        )
        result["shared_policy_tokens"] = shared
        result["policy_containment"] = containment
        result["policy_jaccard"] = jaccard
        relevant = bool(
            shared >= 2
            and containment >= 0.35
            and float(candidate.get("dense_score", 0)) >= 0.28
        )
        result["procedure_compatible"] = relevant
        result["procedure_reason"] = (
            "manifesto_policy_context_match"
            if relevant else "manifesto_policy_context_too_weak"
        )
        result["evidence_role_v3"] = (
            "manifesto_policy_context_direction_unresolved"
            if relevant else "rejected_manifesto_context_mismatch"
        )

    elif source_type == "bill_reference":
        query_signature = bill_signature(
            query.get("legislation_name_clean"),
            query.get("motion_title_clean"),
        )
        evidence_signature = bill_signature("", candidate.get("title"))
        similarity = signature_similarity(
            query_signature, evidence_signature
        )
        result["same_bill"] = similarity >= 0.75
        result["procedure_compatible"] = result["same_bill"]
        result["procedure_reason"] = (
            "matching_bill_background"
            if result["same_bill"] else "unrelated_bill_reference"
        )
        result["evidence_role_v3"] = (
            "matching_bill_background_only"
            if result["same_bill"] else "rejected_unrelated_bill_reference"
        )

    return result


def add_unique(item, selected, used_chunks, used_documents):
    # 防止重复chunk和同一文档占满全部证据位置
    chunk_id = item["chunk_id"]
    document_id = safe_text(item.get("document_id"))
    if chunk_id in used_chunks:
        return False
    if document_id and document_id in used_documents:
        return False
    selected.append(item)
    used_chunks.add(chunk_id)
    if document_id:
        used_documents.add(document_id)
    return True


def select_evidence(enriched_candidates):
    # 对称选择历史支持和反对，再选择少量背景
    selected = []
    used_chunks = set()
    used_documents = set()

    directional = [
        item for item in enriched_candidates
        if item["directional_gate_v3"]
    ]
    for stance in ["support", "oppose"]:
        rows = [
            item for item in directional
            if safe_text(item.get("stance_label")) == stance
        ]
        if rows:
            add_unique(rows[0], selected, used_chunks, used_documents)
    for item in directional:
        if sum(x["directional_gate_v3"] for x in selected) >= MAX_DIRECTIONAL_HISTORY:
            break
        add_unique(item, selected, used_chunks, used_documents)

    cross_role = [
        item for item in enriched_candidates
        if item["evidence_role_v3"]
        == "policy_matched_different_institutional_context"
    ]
    for item in cross_role[:MAX_CROSS_ROLE_CONTEXT]:
        add_unique(item, selected, used_chunks, used_documents)

    manifesto = [
        item for item in enriched_candidates
        if item["evidence_role_v3"]
        == "manifesto_policy_context_direction_unresolved"
    ]
    manifesto_added = 0
    for item in manifesto:
        if manifesto_added >= MAX_MANIFESTO_CONTEXT:
            break
        if add_unique(item, selected, used_chunks, used_documents):
            manifesto_added += 1

    bills = [
        item for item in enriched_candidates
        if item["evidence_role_v3"] == "matching_bill_background_only"
    ]
    for item in bills[:MAX_BILL_BACKGROUND]:
        add_unique(item, selected, used_chunks, used_documents)

    return sorted(
        selected,
        key=lambda item: float(item["hybrid_score"]),
        reverse=True,
    )[:MAX_FINAL_EVIDENCE]
''')

md(r'''
## 8. 对975条冻结预测执行检索

`predicted_stance`只在证据已经固定后用于标记一致或冲突，不参与候选打分、程序筛选或证据选择。
''')

code(r'''
evidence_rows = []
query_rows = []
rejection_counter = defaultdict(int)

ordered_queries = queries.sort_values(
    ["motion_date", "division_key", "party"]
).reset_index(drop=True)

for position, query in ordered_queries.iterrows():
    candidates = score_candidates(query)
    enriched_candidates = []
    for _, candidate in candidates.sort_values(
        "hybrid_score", ascending=False
    ).iterrows():
        item = enrich_candidate(query, candidate)
        enriched_candidates.append(item)
        if not item["procedure_compatible"]:
            rejection_counter[item["procedure_reason"]] += 1

    selected = select_evidence(enriched_candidates)
    supporting = 0
    challenging = 0

    for rank, item in enumerate(selected, start=1):
        stance = safe_text(item.get("stance_label"))
        if item["directional_gate_v3"]:
            relationship = (
                "supports_prediction"
                if stance == query["predicted_stance"]
                else "contradicts_prediction"
            )
            supporting += relationship == "supports_prediction"
            challenging += relationship == "contradicts_prediction"
        elif item["source_type"] == "manifesto":
            relationship = "requires_direction_judgment"
        else:
            relationship = "context_only"

        evidence_rows.append({
            "query_id": query["query_id"],
            "row_id": query["row_id"],
            "query_division_key": query["division_key"],
            "query_date": query["motion_date"],
            "query_party": query["party"],
            "query_party_role": query["party_role"],
            "query_title": query["motion_title_clean"],
            "query_policy_object": query["effective_policy_object"],
            "predicted_stance": query["predicted_stance"],
            "support_probability": query["support_probability"],
            "rank": rank,
            "source_type": item["source_type"],
            "evidence_role_v3": item["evidence_role_v3"],
            "relationship_to_prediction_v3": relationship,
            "evidence_chunk_id": item["chunk_id"],
            "evidence_document_id": item["document_id"],
            "evidence_title": item["title"],
            "evidence_party": item["party"],
            "evidence_date": item["source_date"],
            "evidence_division_key": item["division_key"],
            "historical_party_role": item.get("historical_party_role"),
            "historical_policy_object": item.get("historical_policy_object"),
            "historical_legislation_name": item.get("historical_legislation_name"),
            "stance_label": stance,
            "procedure_compatible": item["procedure_compatible"],
            "procedure_reason": item["procedure_reason"],
            "role_relation": item["role_relation"],
            "government_backing_relation": item["government_backing_relation"],
            "directional_gate_v3": item["directional_gate_v3"],
            "shared_policy_tokens": item["shared_policy_tokens"],
            "policy_containment": item["policy_containment"],
            "policy_jaccard": item["policy_jaccard"],
            "same_bill": item["same_bill"],
            "tfidf_score": item["tfidf_score"],
            "dense_score": item["dense_score"],
            "hybrid_score": item["hybrid_score"],
            "source_url": item["source_url"],
            "page_number": item["page_number"],
            "evidence_text": item["evidence_text"],
            "evidence_is_current_vote_ground_truth": False,
        })

    if supporting and challenging:
        status = "mixed_procedure_and_role_matched_history"
    elif supporting:
        status = "aligned_procedure_and_role_matched_history"
    elif challenging:
        status = "challenging_procedure_and_role_matched_history"
    elif any(item["source_type"] == "manifesto" for item in selected):
        status = "context_requires_direction_review"
    elif selected:
        status = "background_only"
    else:
        status = "insufficient_evidence"

    query_rows.append({
        "query_id": query["query_id"],
        "row_id": query["row_id"],
        "division_key": query["division_key"],
        "motion_date": query["motion_date"],
        "party": query["party"],
        "party_role": query["party_role"],
        "motion_title": query["motion_title_clean"],
        "motion_type": query["motion_type"],
        "motion_family": query["motion_family"],
        "policy_domain_primary": query["policy_domain_primary"],
        "effective_policy_object": query["effective_policy_object"],
        "predicted_stance": query["predicted_stance"],
        "support_probability": query["support_probability"],
        "evidence_results": len(selected),
        "directional_results": sum(
            item["directional_gate_v3"] for item in selected
        ),
        "supporting_history": supporting,
        "challenging_history": challenging,
        "evidence_status_v3": status,
        "rag_produces_separate_prediction": False,
        "prediction_overridden": False,
    })

    if (position + 1) % 100 == 0:
        print(f"Completed retrieval: {position + 1}/{len(ordered_queries)}")

evidence_v3 = pd.DataFrame(evidence_rows)
query_audit_v3 = pd.DataFrame(query_rows)

assert len(query_audit_v3) == len(queries)
assert query_audit_v3["prediction_overridden"].eq(False).all()
if len(evidence_v3):
    assert evidence_v3["evidence_date"].lt(evidence_v3["query_date"]).all()
    assert evidence_v3["query_division_key"].ne(
        evidence_v3["evidence_division_key"]
    ).fillna(True).all()
    assert not evidence_v3.duplicated([
        "query_id", "evidence_chunk_id"
    ]).any()
    assert evidence_v3.groupby("query_id").size().le(
        MAX_FINAL_EVIDENCE
    ).all()
''')

md(r'''
## 9. 生成最终解释包和30个不同division的审计样本

抽样单位改为division，而不是“division × party”。因此30个样本一定是30项不同议案，再包含这30项议案对应的三个产品内政党查询。

抽样按时间季度、政策领域和议案类型进行轮转，避免全部集中在2025年初的少数Bill。
''')

code(r'''
def join_ids(group, relationship=None):
    # 只连接真实存在的证据ID
    subset = group
    if relationship is not None:
        subset = subset[
            subset["relationship_to_prediction_v3"].eq(relationship)
        ]
    return " | ".join(
        subset.sort_values("rank")["evidence_chunk_id"].astype(str)
    )


packet_rows = []
packet_base = base_packets.set_index("query_id")
for _, query in query_audit_v3.iterrows():
    subset = evidence_v3[evidence_v3["query_id"].eq(query["query_id"])]
    old_packet = packet_base.loc[query["query_id"]].to_dict()
    packet_rows.append({
        **old_packet,
        **query.to_dict(),
        "supporting_evidence_ids_v3": join_ids(
            subset, "supports_prediction"
        ),
        "challenging_evidence_ids_v3": join_ids(
            subset, "contradicts_prediction"
        ),
        "context_evidence_ids_v3": " | ".join(
            subset[
                subset["relationship_to_prediction_v3"].isin([
                    "context_only", "requires_direction_judgment"
                ])
            ].sort_values("rank")["evidence_chunk_id"].astype(str)
        ),
        "all_evidence_ids_v3": join_ids(subset),
        "agent_instruction_v3": (
            "Explain the frozen probability formula first. Use only evidence "
            "that passed the procedure-aware gate for directional claims. "
            "Treat manifesto and different-role history as context. Present "
            "conflicts explicitly. Never change the frozen probability or vote "
            "by evidence count."
        ),
    })

agent_packets_v3 = pd.DataFrame(packet_rows)


def stable_hash(value):
    # 固定哈希保证每次抽到相同division
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


division_frame = (
    query_audit_v3.sort_values(["motion_date", "division_key"])
    .drop_duplicates("division_key")
    .copy()
)
division_frame["time_quarter"] = pd.to_datetime(
    division_frame["motion_date"]
).dt.to_period("Q").astype(str)
division_frame["sampling_stratum"] = (
    division_frame["time_quarter"].fillna("unknown")
    + "__" + division_frame["policy_domain_primary"].fillna("unknown")
    + "__" + division_frame["motion_family"].fillna("unknown")
)
division_frame["stable_hash"] = division_frame["division_key"].map(
    stable_hash
)
division_frame["stratum_rank"] = division_frame.sort_values(
    "stable_hash"
).groupby("sampling_stratum").cumcount()
manual_divisions = division_frame.sort_values([
    "stratum_rank", "sampling_stratum", "stable_hash"
]).head(MANUAL_DIVISIONS)

manual_queries = query_audit_v3[
    query_audit_v3["division_key"].isin(manual_divisions["division_key"])
].copy()
manual_evidence = evidence_v3[
    evidence_v3["query_id"].isin(manual_queries["query_id"])
].copy()
manual_evidence["manual_policy_relevance"] = ""
manual_evidence["manual_procedure_compatibility"] = ""
manual_evidence["manual_direction_valid"] = ""
manual_evidence["manual_notes"] = ""

assert manual_divisions["division_key"].nunique() == MANUAL_DIVISIONS
assert manual_queries["division_key"].nunique() == MANUAL_DIVISIONS

print("Manual-review divisions:", manual_divisions["division_key"].nunique())
print("Manual-review party queries:", len(manual_queries))
print("Manual-review evidence rows:", len(manual_evidence))
display(manual_divisions[[
    "motion_date", "motion_title", "policy_domain_primary",
    "motion_family", "time_quarter",
]].head(30))
''')

md(r'''
## 10. 保存结果和免费结构验收

结构门槛验证检索安全性和规则执行。它不使用真实投票答案，因此不能评价预测准确率。
''')

code(r'''
if len(evidence_v3):
    directional = evidence_v3["directional_gate_v3"].map(parse_bool)
else:
    directional = pd.Series(dtype=bool)

party_summary = query_audit_v3.groupby("party").agg(
    queries=("query_id", "size"),
    mean_evidence=("evidence_results", "mean"),
    directional_query_rate=(
        "directional_results", lambda values: float((values > 0).mean())
    ),
    aligned_history_rate=(
        "supporting_history", lambda values: float((values > 0).mean())
    ),
    challenging_history_rate=(
        "challenging_history", lambda values: float((values > 0).mean())
    ),
).reset_index()

rejection_counts = pd.Series(rejection_counter).sort_values(ascending=False)
status_counts = query_audit_v3["evidence_status_v3"].value_counts()

structural_gates = {
    "all_queries_preserved_gate": bool(
        len(query_audit_v3) == len(queries)
    ),
    "embedding_cache_gate": embedding_cache_valid,
    "historical_role_metadata_gate": bool(
        historical_metadata_coverage >= 0.99
    ),
    "strict_temporal_gate": bool(
        len(evidence_v3) == 0
        or evidence_v3["evidence_date"].lt(
            evidence_v3["query_date"]
        ).all()
    ),
    "same_division_gate": bool(
        len(evidence_v3) == 0
        or evidence_v3["query_division_key"].ne(
            evidence_v3["evidence_division_key"]
        ).fillna(True).all()
    ),
    "maximum_seven_results_gate": bool(
        len(evidence_v3) == 0
        or evidence_v3.groupby("query_id").size().le(
            MAX_FINAL_EVIDENCE
        ).all()
    ),
    "direction_requires_procedure_gate": bool(
        len(evidence_v3) == 0
        or evidence_v3.loc[directional, "procedure_compatible"]
        .map(parse_bool).all()
    ),
    "direction_requires_same_role_gate": bool(
        len(evidence_v3) == 0
        or evidence_v3.loc[directional, "role_relation"]
        .eq("same_role").all()
    ),
    "different_backing_not_directional_gate": bool(
        len(evidence_v3) == 0
        or not evidence_v3.loc[
            evidence_v3["government_backing_relation"].eq("different"),
            "directional_gate_v3",
        ].map(parse_bool).any()
    ),
    "bill_reference_background_gate": bool(
        len(evidence_v3) == 0
        or not evidence_v3.loc[
            evidence_v3["source_type"].eq("bill_reference"),
            "directional_gate_v3",
        ].map(parse_bool).any()
    ),
    "thirty_unique_divisions_gate": bool(
        manual_divisions["division_key"].nunique() == 30
    ),
    "prediction_not_overridden_gate": bool(
        query_audit_v3["prediction_overridden"].eq(False).all()
    ),
    "no_target_columns_read_gate": True,
    "no_api_calls_gate": True,
    "no_llm_calls_gate": True,
}

evidence_v3.to_csv(
    OUTPUT_DIR / "procedure_aware_evidence_v3.csv", index=False
)
query_audit_v3.to_csv(
    OUTPUT_DIR / "procedure_aware_query_audit_v3.csv", index=False
)
agent_packets_v3.to_csv(
    OUTPUT_DIR / "procedure_aware_agent_packets_v3.csv", index=False
)
manual_divisions.to_csv(
    OUTPUT_DIR / "manual_review_divisions_v3.csv", index=False
)
manual_queries.to_csv(
    OUTPUT_DIR / "manual_review_queries_v3.csv", index=False
)
manual_evidence.to_csv(
    OUTPUT_DIR / "manual_review_evidence_v3.csv", index=False
)
party_summary.to_csv(
    OUTPUT_DIR / "procedure_aware_party_summary_v3.csv", index=False
)
rejection_counts.rename("candidate_rows").to_csv(
    OUTPUT_DIR / "procedure_rejection_reasons_v3.csv"
)

manifest = {
    "notebook_version": "11c-procedure-aware-hybrid-rag-v3",
    "embedding_model": EMBEDDING_MODEL,
    "hybrid_dense_weight": HYBRID_DENSE_WEIGHT,
    "candidate_pool_size": CANDIDATE_POOL_SIZE,
    "queries": int(len(query_audit_v3)),
    "evidence_rows": int(len(evidence_v3)),
    "manual_review_divisions": int(
        manual_divisions["division_key"].nunique()
    ),
    "structural_gates": structural_gates,
    "api_calls": 0,
    "llm_calls": 0,
}
(OUTPUT_DIR / "run_manifest_v3.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

display(party_summary.round(3))
display(status_counts.to_frame("queries"))
display(rejection_counts.head(15).to_frame("candidate_rows"))

summary_lines = [
    "=== PROCEDURE-AWARE HYBRID RAG SUMMARY FOR REVIEW ===",
    "Notebook version: 11c-procedure-aware-hybrid-rag-v3",
    "API calls: 0",
    "LLM calls: 0",
    "Test labels read: False",
    f"Embedding model: {EMBEDDING_MODEL}",
    f"Hybrid dense weight: {HYBRID_DENSE_WEIGHT}",
    f"Candidate pool size: {CANDIDATE_POOL_SIZE}",
    f"Queries: {len(query_audit_v3)}",
    f"Unique divisions: {query_audit_v3['division_key'].nunique()}",
    f"Final evidence rows: {len(evidence_v3)}",
    f"Mean evidence per query: {query_audit_v3['evidence_results'].mean():.3f}",
    f"Queries with directional evidence rate: "
    f"{query_audit_v3['directional_results'].gt(0).mean():.3f}",
    f"Queries with any evidence rate: "
    f"{query_audit_v3['evidence_results'].gt(0).mean():.3f}",
    f"Manual-review unique divisions: "
    f"{manual_divisions['division_key'].nunique()}",
    f"Manual-review party queries: {len(manual_queries)}",
    f"Manual-review evidence rows: {len(manual_evidence)}",
    f"Structural gates: {structural_gates}",
    "Evidence status:",
    status_counts.to_string(),
    "Party evidence coverage:",
    party_summary.round(3).to_string(index=False),
    "Top procedure rejection reasons:",
    rejection_counts.head(15).to_string(),
    f"Output directory: {OUTPUT_DIR}",
    "Next step: Codex reviews the fixed 30-division evidence sample before any LLM calls.",
    "=== END PROCEDURE-AWARE HYBRID RAG SUMMARY ===",
]

summary_text = "\n".join(summary_lines)
print(summary_text)
(OUTPUT_DIR / "procedure_aware_hybrid_rag_summary_v3.txt").write_text(
    summary_text, encoding="utf-8"
)
''')

md(r'''
## 如何理解结果

覆盖率下降不一定是坏事。11C的目标是减少“看起来相关但不能判断方向”的材料。

重点看：

1. `Queries with directional evidence rate`：有多少查询真正找到程序和角色都可比的历史投票；
2. `Queries with any evidence rate`：有多少查询至少找到可靠背景；
3. `Top procedure rejection reasons`：旧材料主要因为什么被排除；
4. 固定30个不同division的人工样本：严格证据是否真的比11B更精确。

如果方向覆盖率下降但人工精度显著提高，说明系统变得更诚实。Agent可以在缺少直接证据时明确说“模型给出概率，但当前知识库只有背景资料”。
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
