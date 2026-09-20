from pathlib import Path

import nbformat as nbf


ROOT = Path.cwd()
OUTPUT = ROOT / '05_build_rag_knowledge_base_v2.ipynb'

nb = nbf.v4.new_notebook()
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python (spatial)',
        'language': 'python',
        'name': 'spatial',
    },
    'language_info': {'name': 'python', 'version': '3'},
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
    """# 05 — Build the RAG knowledge base

**Notebook version: 05-rag-v2-multichannel**

本 Notebook 只负责知识库与检索，不调用最终 LLM，也不预测 Test 标签。

知识库包含三类证据：

1. 四个政党的 2024 官方 manifesto；
2. Train 和 2024 Validation 中已经发生的历史议案与政党立场；
3. 可选的 Bill title、summary 和 stage 信息。

安全原则：

- Test 投票结果绝不进入知识库；
- 历史投票证据必须早于待预测 motion 日期；
- 检索时排除当前 division，防止同一议案泄漏；
- 每个 chunk 保存 party、source type、source date 和 URL；
- 未注明日期的手工资料默认不进入时间敏感检索。

当前环境默认使用可直接运行的 TF-IDF sparse retrieval。Sentence Transformer dense retrieval 是可选升级，不是05完成的必要条件。"""
))

cells.append(nbf.v4.new_code_cell(
    """# 导入知识库构建与检索需要的库。
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import json
import re
import shutil
import subprocess
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 180)
pd.set_option('display.max_colwidth', 160)

RANDOM_STATE = 42
TARGET = 'target_binary_support'
TEXT_COLUMN = 'model_text'

# True 时下载四个官方 manifesto；已经存在的文件不会重复下载。
DOWNLOAD_OFFICIAL_SOURCES = True

# 当前 spatial 环境未安装 sentence-transformers，因此默认关闭。
ENABLE_DENSE_EMBEDDINGS = False
DENSE_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# 未标注日期的手工文件默认不能用于时间敏感检索。
ALLOW_UNDATED_MANUAL_SOURCES = False

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 220
TOP_K = 8

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / 'processed' / 'model_v2'
RAG_SOURCE_DIR = BASE_DIR / 'rag_sources'
OFFICIAL_DIR = RAG_SOURCE_DIR / 'official_downloads'
MANUAL_DIR = RAG_SOURCE_DIR / 'manual'
OUTPUT_DIR = BASE_DIR / 'processed' / 'rag_v1'

for directory in [RAG_SOURCE_DIR, OFFICIAL_DIR, MANUAL_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

PARTIES = ['conservative', 'green', 'labour', 'liberal-democrat']
for party in PARTIES:
    (MANUAL_DIR / party).mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / 'model_train_v2.csv'
VALIDATION_PATH = DATA_DIR / 'model_validation_v2.csv'
TEST_PATH = DATA_DIR / 'model_test_v2.csv'
BILL_INFO_PATH = BASE_DIR / 'data' / 'raw' / 'bill_info.csv'

print('RAG source directory:', RAG_SOURCE_DIR)
print('Output directory:', OUTPUT_DIR)
print('Dense embeddings enabled:', ENABLE_DENSE_EMBEDDINGS)"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 1. Official source registry

来源表只登记官方政党页面或官方 PDF。`published_at` 用于时间过滤；这些 manifesto 均早于 2024 年大选后的开发查询和 2025–2026 Test。

如自动下载失败，可以手工下载文件并放入：

```text
reorganised/rag_sources/manual/<party>/YYYY-MM-DD__document-title.pdf
```

手工文件名必须以日期开头，才能进入时间敏感检索。"""
))

cells.append(nbf.v4.new_code_cell(
    """# 四个来源均为政党官方网页或官方文件。
official_sources = pd.DataFrame([
    {
        'document_id': 'manifesto_2024_labour',
        'party': 'labour',
        'title': 'Change — Labour Party Manifesto 2024',
        'source_type': 'manifesto',
        'published_at': '2024-06-13',
        'url': 'https://labour.org.uk/wp-content/uploads/2024/06/Labour-Party-manifesto-2024.pdf',
        'format': 'pdf',
        'local_filename': '2024-06-13__labour_manifesto_2024.pdf',
    },
    {
        'document_id': 'manifesto_2024_conservative',
        'party': 'conservative',
        'title': 'Conservative and Unionist Party Manifesto 2024',
        'source_type': 'manifesto',
        'published_at': '2024-06-11',
        'url': 'https://public.conservatives.com/publicweb/GE2024/Accessible-Manifesto/Accessible-PDF-Conservative-Manifesto-2024.pdf',
        'format': 'pdf',
        'local_filename': '2024-06-11__conservative_manifesto_2024.pdf',
    },
    {
        'document_id': 'manifesto_2024_liberal_democrat',
        'party': 'liberal-democrat',
        'title': 'For a Fair Deal — Liberal Democrat Manifesto 2024',
        'source_type': 'manifesto',
        'published_at': '2024-06-10',
        'url': 'https://www.libdems.org.uk/fileadmin/groups/2_Federal_Party/Documents/PolicyPapers/Manifesto_2024/For_a_Fair_Deal_-_Liberal_Democrat_Manifesto_2024_-_Clear_Print.pdf',
        'format': 'pdf',
        'local_filename': '2024-06-10__liberal_democrat_manifesto_2024.pdf',
    },
    {
        'document_id': 'manifesto_2024_green',
        'party': 'green',
        'title': 'Green Party 2024 General Election Manifesto',
        'source_type': 'manifesto',
        'published_at': '2024-06-12',
        'url': 'https://greenparty.org.uk/app/uploads/2024/06/Green-Party-2024-General-Election-Manifesto-Long-version-with-cover.pdf',
        'format': 'pdf',
        'local_filename': '2024-06-12__green_manifesto_2024.pdf',
    },
])

official_sources['published_at'] = pd.to_datetime(
    official_sources['published_at']
)
official_sources['local_path'] = official_sources['local_filename'].map(
    lambda name: str(OFFICIAL_DIR / name)
)
official_sources.to_csv(
    OUTPUT_DIR / 'official_policy_source_registry.csv', index=False
)
display(official_sources)"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 2. Download and extract official documents

PDF 使用系统中的 `pdftotext` 提取。HTML 会删除脚本、导航和样式标签，只保留可见正文。下载与提取状态都会保存在审计表中。"""
))

cells.append(nbf.v4.new_code_cell(
    """def local_source_is_valid(path, expected_format):
    # 防止网站重定向后把 HTML 首页误存为 PDF。
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    prefix = path.read_bytes()[:20].lstrip().lower()
    if expected_format == 'pdf':
        return prefix.startswith(b'%pdf')
    if expected_format == 'html':
        return b'<html' in prefix or b'<!doctype html' in prefix
    return True


def download_source(url, destination, expected_format):
    # 使用明确的 User-Agent，并验证实际下载格式。
    destination = Path(destination)
    if local_source_is_valid(destination, expected_format):
        return 'existing'

    response = requests.get(
        url,
        timeout=60,
        headers={'User-Agent': 'Academic RAG project/1.0'},
    )
    response.raise_for_status()
    destination.write_bytes(response.content)
    if not local_source_is_valid(destination, expected_format):
        raise ValueError(
            f'下载内容不是有效的 {expected_format}: {destination}'
        )
    return 'downloaded'


def extract_pdf_text(path):
    # 优先使用当前机器已有的 pdftotext，不额外安装 Python PDF 包。
    executable = shutil.which('pdftotext')
    if executable is None:
        raise RuntimeError(
            '未找到 pdftotext。请手工把 PDF 转成同名 .txt 文件。'
        )
    result = subprocess.run(
        [executable, '-layout', str(path), '-'],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def extract_html_text(path):
    # 移除不属于正文的网页元素。
    html = Path(path).read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript']):
        tag.decompose()
    container = soup.find('main') or soup.find('article') or soup.body or soup
    return container.get_text('\\n', strip=True)


def clean_extracted_text(text):
    # 保留段落边界，同时压缩重复空格和过多空行。
    text = str(text).replace('\\x00', ' ')
    lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return '\\n'.join(lines)


download_rows = []
official_document_rows = []

for _, source in official_sources.iterrows():
    local_path = Path(source['local_path'])
    download_status = 'disabled'
    extraction_status = 'not_attempted'
    error = pd.NA
    text = ''

    try:
        if DOWNLOAD_OFFICIAL_SOURCES:
            download_status = download_source(
                source['url'], local_path, source['format']
            )
        elif local_path.exists():
            download_status = 'existing'

        if local_path.exists():
            if source['format'] == 'pdf':
                text = extract_pdf_text(local_path)
            elif source['format'] == 'html':
                text = extract_html_text(local_path)
            else:
                text = local_path.read_text(
                    encoding='utf-8', errors='ignore'
                )
            text = clean_extracted_text(text)
            extraction_status = 'ok' if text else 'empty'
    except Exception as exc:
        extraction_status = 'error'
        error = f'{type(exc).__name__}: {exc}'

    download_rows.append({
        'document_id': source['document_id'],
        'party': source['party'],
        'download_status': download_status,
        'extraction_status': extraction_status,
        'characters': len(text),
        'error': error,
    })

    if text:
        official_document_rows.append({
            'document_id': source['document_id'],
            'party': source['party'],
            'source_type': source['source_type'],
            'title': source['title'],
            'source_url': source['url'],
            'source_date': source['published_at'],
            'division_key': pd.NA,
            'motion_type': pd.NA,
            'motion_family': pd.NA,
            'policy_domain': pd.NA,
            'stance_label': pd.NA,
            'text': text,
        })

download_audit = pd.DataFrame(download_rows)
download_audit.to_csv(
    OUTPUT_DIR / 'official_source_download_audit.csv', index=False
)
display(download_audit)

if not download_audit['extraction_status'].eq('ok').all():
    print(
        '部分官方来源未成功提取。Notebook 会继续，但请根据 error 列补充手工文件。'
    )"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 3. Load optional manual policy documents

支持 `.txt`、`.md`、`.html`、`.htm` 和 `.pdf`。文件名格式：

```text
YYYY-MM-DD__short-title.ext
```

例如：

```text
rag_sources/manual/green/2024-09-20__autumn-conference-policy.pdf
```

手工资料应当来自官方政党来源，并在06使用前填写真实 URL；05不会把无 URL 的资料伪装成可引用来源。"""
))

cells.append(nbf.v4.new_code_cell(
    """SUPPORTED_EXTENSIONS = {'.txt', '.md', '.html', '.htm', '.pdf'}


def parse_date_from_filename(path):
    # 只接受文件名开头的 YYYY-MM-DD。
    match = re.match(r'^(\d{4}-\d{2}-\d{2})__', Path(path).name)
    return pd.to_datetime(match.group(1)) if match else pd.NaT


def read_local_document(path):
    # 根据扩展名选择提取方式。
    path = Path(path)
    if path.suffix.lower() == '.pdf':
        return clean_extracted_text(extract_pdf_text(path))
    if path.suffix.lower() in {'.html', '.htm'}:
        return clean_extracted_text(extract_html_text(path))
    return clean_extracted_text(
        path.read_text(encoding='utf-8', errors='ignore')
    )


manual_document_rows = []
manual_audit_rows = []

for party in PARTIES:
    for path in sorted((MANUAL_DIR / party).glob('*')):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        source_date = parse_date_from_filename(path)
        text = ''
        error = pd.NA
        try:
            text = read_local_document(path)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'

        allowed_for_time_retrieval = bool(
            pd.notna(source_date) or ALLOW_UNDATED_MANUAL_SOURCES
        )
        document_id = 'manual_' + hashlib.sha256(
            str(path).encode('utf-8')
        ).hexdigest()[:16]

        manual_audit_rows.append({
            'document_id': document_id,
            'party': party,
            'path': str(path),
            'source_date': source_date,
            'characters': len(text),
            'allowed_for_time_retrieval': allowed_for_time_retrieval,
            'error': error,
        })

        if text and allowed_for_time_retrieval:
            manual_document_rows.append({
                'document_id': document_id,
                'party': party,
                'source_type': 'manual_policy_document',
                'title': path.stem.split('__', 1)[-1].replace('_', ' '),
                'source_url': pd.NA,
                'source_date': source_date,
                'division_key': pd.NA,
                'motion_type': pd.NA,
                'motion_family': pd.NA,
                'policy_domain': pd.NA,
                'stance_label': pd.NA,
                'text': text,
            })

manual_audit = pd.DataFrame(manual_audit_rows)
if len(manual_audit):
    display(manual_audit)
else:
    print('No manual policy documents found. This is allowed for the first run.')"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 4. Build historical vote evidence

Train 和 2024 Validation 都早于 2025–2026 Test，因此可以作为历史案例。每条案例包含 motion 文本、party 和已经观察到的历史立场。

这不是泄漏：06检索时必须强制 `source_date < query_date`，并排除相同 `division_key`。"""
))

cells.append(nbf.v4.new_code_cell(
    """# Test 文件只读取 division_key，不读取其标签进入知识库。
train = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
validation = pd.read_csv(VALIDATION_PATH).reset_index(drop=True)
test_keys = pd.read_csv(TEST_PATH, usecols=['division_key'])

development_votes = pd.concat([
    train.assign(source_split='train'),
    validation.assign(source_split='validation'),
], ignore_index=True)
development_votes['motion_date'] = pd.to_datetime(
    development_votes['motion_date']
)

assert not set(development_votes['division_key']) & set(test_keys['division_key'])
assert development_votes['row_id'].is_unique


def safe_text(value):
    # 把缺失文本统一为空字符串。
    return '' if pd.isna(value) else str(value).strip()


historical_document_rows = []
for _, row in development_votes.iterrows():
    stance = 'support' if int(row[TARGET]) == 1 else 'oppose'
    title = safe_text(row.get('motion_title_clean'))
    motion_text = safe_text(row.get('motion_text_clean'))
    legislation = safe_text(row.get('legislation_name_clean'))

    text = clean_extracted_text(
        f"Historical parliamentary vote.\\n"
        f"Party: {row['party']}.\\n"
        f"Observed stance: {stance}.\\n"
        f"Motion title: {title}.\\n"
        f"Motion text: {motion_text}.\\n"
        f"Legislation: {legislation}.\\n"
        f"Motion type: {safe_text(row.get('motion_type'))}.\\n"
        f"Motion family: {safe_text(row.get('motion_family'))}.\\n"
        f"Policy domain: {safe_text(row.get('policy_domain_primary'))}."
    )

    historical_document_rows.append({
        'document_id': f"historical_vote_{row['row_id']}",
        'party': row['party'],
        'source_type': 'historical_vote',
        'title': title or safe_text(row.get('division_name_clean')),
        'source_url': pd.NA,
        'source_date': row['motion_date'],
        'division_key': row['division_key'],
        'motion_type': row.get('motion_type', pd.NA),
        'motion_family': row.get('motion_family', pd.NA),
        'policy_domain': row.get('policy_domain_primary', pd.NA),
        'stance_label': stance,
        'text': text,
    })

historical_documents = pd.DataFrame(historical_document_rows)
display(pd.DataFrame({
    'rows': [len(historical_documents)],
    'divisions': [historical_documents['division_key'].nunique()],
    'parties': [historical_documents['party'].nunique()],
    'min_date': [historical_documents['source_date'].min()],
    'max_date': [historical_documents['source_date'].max()],
}))"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 5. Build optional Bill reference documents

Bill 数据只有约129条，因此只作为补充说明，不作为知识库主体。没有 source URL 的 Bill chunk 可以帮助检索上下文，但06输出引用时应优先使用有 URL 的政策文件。"""
))

cells.append(nbf.v4.new_code_cell(
    """bill_document_rows = []

if BILL_INFO_PATH.exists():
    bill_info = pd.read_csv(BILL_INFO_PATH)
    for _, row in bill_info.iterrows():
        bill_id = safe_text(row.get('billId'))
        title = safe_text(row.get('shortTitle')) or safe_text(row.get('longTitle'))
        summary = safe_text(row.get('summary'))
        long_title = safe_text(row.get('longTitle'))
        stage = safe_text(row.get('currentStage'))
        source_date = pd.to_datetime(row.get('lastUpdate'), errors='coerce')

        text = clean_extracted_text(
            f"Bill reference.\\n"
            f"Bill title: {title}.\\n"
            f"Long title: {long_title}.\\n"
            f"Summary: {summary}.\\n"
            f"Current stage: {stage}."
        )
        if not text:
            continue

        bill_document_rows.append({
            'document_id': f'bill_{bill_id or len(bill_document_rows)}',
            'party': 'all',
            'source_type': 'bill_reference',
            'title': title,
            'source_url': pd.NA,
            'source_date': source_date,
            'division_key': pd.NA,
            'motion_type': pd.NA,
            'motion_family': pd.NA,
            'policy_domain': pd.NA,
            'stance_label': pd.NA,
            'text': text,
        })

bill_documents = pd.DataFrame(bill_document_rows)
print('Bill documents:', len(bill_documents))
if not BILL_INFO_PATH.exists():
    print('Bill info file not found; continuing without Bill references.')"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 6. Combine documents and create chunks

Manifesto 等长文档使用重叠分块；历史投票和 Bill reference 通常较短，会保留为一个 chunk。Chunk ID 根据 document ID、位置和文本生成，保证可追踪。"""
))

cells.append(nbf.v4.new_code_cell(
    """document_frames = [
    pd.DataFrame(official_document_rows),
    pd.DataFrame(manual_document_rows),
    historical_documents,
]
if len(bill_documents):
    document_frames.append(bill_documents)

documents = pd.concat(
    [frame for frame in document_frames if len(frame)],
    ignore_index=True,
)
documents['source_date'] = pd.to_datetime(
    documents['source_date'], errors='coerce'
)
documents['text'] = documents['text'].fillna('').astype(str)
documents['character_count'] = documents['text'].str.len()


def split_long_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    # 优先在空格或段落边界切分，并保留少量重叠上下文。
    text = clean_extracted_text(text)
    if len(text) <= chunk_size:
        return [(0, len(text), text)]

    chunks = []
    start = 0
    while start < len(text):
        target_end = min(start + chunk_size, len(text))
        end = target_end

        if target_end < len(text):
            candidates = [
                text.rfind('\\n', start + chunk_size // 2, target_end),
                text.rfind('. ', start + chunk_size // 2, target_end),
                text.rfind(' ', start + chunk_size // 2, target_end),
            ]
            valid_candidates = [position for position in candidates if position > start]
            if valid_candidates:
                end = max(valid_candidates) + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((start, end, chunk_text))

        if end >= len(text):
            break
        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


chunk_rows = []
for _, document in documents.iterrows():
    for chunk_index, (start, end, chunk_text) in enumerate(
        split_long_text(document['text'])
    ):
        digest = hashlib.sha256(
            f"{document['document_id']}|{chunk_index}|{chunk_text}".encode('utf-8')
        ).hexdigest()[:20]
        chunk_rows.append({
            'chunk_id': f'chunk_{digest}',
            'document_id': document['document_id'],
            'chunk_index': chunk_index,
            'party': document['party'],
            'source_type': document['source_type'],
            'title': document['title'],
            'source_url': document['source_url'],
            'source_date': document['source_date'],
            'division_key': document['division_key'],
            'motion_type': document['motion_type'],
            'motion_family': document['motion_family'],
            'policy_domain': document['policy_domain'],
            'stance_label': document['stance_label'],
            'start_char': start,
            'end_char': end,
            'text': chunk_text,
            'character_count': len(chunk_text),
            'word_count': len(chunk_text.split()),
        })

chunks = pd.DataFrame(chunk_rows)

# 删除完全相同的同政党文本，但保留不同政党的历史立场案例。
chunks['text_hash'] = chunks['text'].map(
    lambda value: hashlib.sha256(
        re.sub(r'\s+', ' ', value).strip().lower().encode('utf-8')
    ).hexdigest()
)
before_deduplication = len(chunks)
chunks = chunks.drop_duplicates(
    subset=['party', 'source_type', 'text_hash']
).reset_index(drop=True)

print('Documents:', len(documents))
print('Chunks before deduplication:', before_deduplication)
print('Chunks after deduplication:', len(chunks))"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 7. Knowledge-base audit

最低要求：每个政党至少有历史投票 chunk。Manifesto 下载失败不会静默通过，会在 `policy_source_coverage` 中明确显示。"""
))

cells.append(nbf.v4.new_code_cell(
    """source_type_audit = pd.crosstab(
    chunks['party'], chunks['source_type'], margins=True
)
display(source_type_audit)

policy_source_coverage = (
    chunks[chunks['source_type'].isin(['manifesto', 'manual_policy_document'])]
    .groupby('party')
    .agg(
        policy_documents=('document_id', 'nunique'),
        policy_chunks=('chunk_id', 'size'),
        dated_chunks=('source_date', lambda values: values.notna().sum()),
        url_chunks=('source_url', lambda values: values.notna().sum()),
    )
    .reindex(PARTIES, fill_value=0)
)
display(policy_source_coverage)

historical_coverage = (
    chunks[chunks['source_type'].eq('historical_vote')]
    .groupby('party')
    .agg(
        chunks=('chunk_id', 'size'),
        divisions=('division_key', 'nunique'),
        earliest=('source_date', 'min'),
        latest=('source_date', 'max'),
    )
    .reindex(PARTIES)
)
display(historical_coverage)

assert set(PARTIES).issubset(set(historical_coverage.dropna().index))
assert chunks['chunk_id'].is_unique
assert chunks['text'].str.len().gt(0).all()

missing_manifestos = policy_source_coverage.index[
    policy_source_coverage['policy_documents'].eq(0)
].tolist()
if missing_manifestos:
    print('Missing policy documents:', missing_manifestos)
    print('请根据官方下载审计表补充手工文件，然后重新运行05。')
else:
    print('All four parties have at least one policy document.')"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 8. Save the auditable corpus

CSV 方便人工查看；JSONL 方便06逐条读取。`source_documents` 和 `chunks` 分开保存，避免检索结果失去原始文档关系。"""
))

cells.append(nbf.v4.new_code_cell(
    """documents.to_csv(OUTPUT_DIR / 'rag_documents.csv', index=False)
chunks.to_csv(OUTPUT_DIR / 'rag_chunks.csv', index=False)
documents.to_json(
    OUTPUT_DIR / 'rag_documents.jsonl',
    orient='records',
    lines=True,
    force_ascii=False,
    date_format='iso',
)
chunks.to_json(
    OUTPUT_DIR / 'rag_chunks.jsonl',
    orient='records',
    lines=True,
    force_ascii=False,
    date_format='iso',
)

print('Saved:', OUTPUT_DIR / 'rag_documents.csv')
print('Saved:', OUTPUT_DIR / 'rag_chunks.csv')"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 9. Build the sparse retrieval index

索引只学习 chunk 文本，不读取 Test。Word unigram/bigram 检索适合作为可解释、可直接运行的第一版 RAG retriever。"""
))

cells.append(nbf.v4.new_code_cell(
    """# 在知识库 chunk 上建立并保存 TF-IDF sparse index。
sparse_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents='unicode',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    max_features=50000,
    sublinear_tf=True,
    norm='l2',
)
sparse_matrix = sparse_vectorizer.fit_transform(chunks['text'])

joblib.dump(
    sparse_vectorizer,
    OUTPUT_DIR / 'sparse_tfidf_vectorizer.joblib',
)
save_npz(OUTPUT_DIR / 'sparse_tfidf_matrix.npz', sparse_matrix)

print('Sparse matrix shape:', sparse_matrix.shape)
print('Vocabulary size:', len(sparse_vectorizer.vocabulary_))"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 10. Optional dense embedding index

当前环境没有安装 `sentence-transformers`，因此默认跳过。若以后安装并把开关改为 `True`，会生成归一化 dense embeddings；05的 sparse 检索结果不依赖该步骤。"""
))

cells.append(nbf.v4.new_code_cell(
    """dense_embeddings = None
dense_model = None

if ENABLE_DENSE_EMBEDDINGS:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            'ENABLE_DENSE_EMBEDDINGS=True，但当前环境没有 sentence-transformers。'
        ) from exc

    dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    dense_embeddings = dense_model.encode(
        chunks['text'].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(OUTPUT_DIR / 'dense_embeddings.npy', dense_embeddings)
    (OUTPUT_DIR / 'dense_model_name.txt').write_text(
        DENSE_MODEL_NAME, encoding='utf-8'
    )
    print('Dense embedding shape:', dense_embeddings.shape)
else:
    print('Dense embeddings skipped; sparse retrieval remains available.')"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 11. Leakage-safe retriever

检索规则：

- 只返回目标政党或 `party=all` 的资料；
- `source_date` 必须早于 query date；
- 排除当前 division；
- manifesto、历史投票和 Bill reference 都保留来源类型；
- dense index 启用时使用 Reciprocal Rank Fusion 合并 sparse/dense 排名。"""
))

cells.append(nbf.v4.new_code_cell(
    """def build_candidate_mask(
    party,
    query_date,
    exclude_division_key=None,
):
    # 构造 party、日期和 division 三重过滤。
    query_date = pd.Timestamp(query_date)
    party_mask = chunks['party'].isin([party, 'all']).to_numpy()

    source_dates = pd.to_datetime(chunks['source_date'], errors='coerce')
    dated_mask = source_dates.notna().to_numpy()
    time_mask = (source_dates < query_date).fillna(False).to_numpy()

    if exclude_division_key is None:
        division_mask = np.ones(len(chunks), dtype=bool)
    else:
        division_mask = chunks['division_key'].ne(
            exclude_division_key
        ).fillna(True).to_numpy()

    return party_mask & dated_mask & time_mask & division_mask


def reciprocal_rank_fusion(rank_lists, rank_constant=60):
    # 合并不同检索器的排名，而不是直接比较不可比的原始分数。
    scores = {}
    for ranking in rank_lists:
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (
                rank_constant + rank
            )
    return sorted(scores, key=scores.get, reverse=True), scores


def retrieve_evidence(
    query,
    party,
    query_date,
    exclude_division_key=None,
    top_k=TOP_K,
):
    # 先过滤候选，再计算 sparse 分数；可选 dense 排名通过 RRF 合并。
    candidate_mask = build_candidate_mask(
        party=party,
        query_date=query_date,
        exclude_division_key=exclude_division_key,
    )
    candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) == 0:
        return pd.DataFrame()

    query_vector = sparse_vectorizer.transform([query])
    sparse_scores = (
        sparse_matrix[candidate_indices] @ query_vector.T
    ).toarray().ravel()
    sparse_order = candidate_indices[np.argsort(-sparse_scores)]

    # 保留全部候选排名，保证数量较少的 manifesto 不会在截断前消失。
    rankings = [sparse_order.tolist()]
    sparse_score_map = {
        int(index): float(score)
        for index, score in zip(candidate_indices, sparse_scores)
    }

    dense_score_map = {}
    if dense_embeddings is not None and dense_model is not None:
        query_embedding = dense_model.encode(
            [query], normalize_embeddings=True
        )[0]
        dense_scores = dense_embeddings[candidate_indices] @ query_embedding
        dense_order = candidate_indices[np.argsort(-dense_scores)]
        rankings.append(dense_order.tolist())
        dense_score_map = {
            int(index): float(score)
            for index, score in zip(candidate_indices, dense_scores)
        }

    fused_order, fused_scores = reciprocal_rank_fusion(rankings)

    # 固定多通道配额，防止大量历史投票淹没官方政策证据。
    channel_specs = [
        ('policy', {'manifesto', 'manual_policy_document'}, 2),
        ('historical', {'historical_vote'}, 4),
        ('bill', {'bill_reference'}, 1),
    ]
    selected = []
    selected_channels = {}
    per_document_count = {}

    def add_candidate(index, channel):
        # 每个原始文档最多返回两个 chunk，并避免重复加入同一行。
        if index in selected:
            return False
        document_id = chunks.iloc[index]['document_id']
        count = per_document_count.get(document_id, 0)
        if count >= 2:
            return False
        selected.append(index)
        selected_channels[index] = channel
        per_document_count[document_id] = count + 1
        return True

    for channel, source_types, quota in channel_specs:
        added = 0
        for index in fused_order:
            if chunks.iloc[index]['source_type'] not in source_types:
                continue
            if add_candidate(index, channel):
                added += 1
            if added >= quota or len(selected) >= top_k:
                break

    # 配额不足或尚有空位时，再按全局相关性补齐。
    for index in fused_order:
        if len(selected) >= top_k:
            break
        source_type = chunks.iloc[index]['source_type']
        if source_type in {'manifesto', 'manual_policy_document'}:
            channel = 'policy'
        elif source_type == 'historical_vote':
            channel = 'historical'
        elif source_type == 'bill_reference':
            channel = 'bill'
        else:
            channel = 'other'
        add_candidate(index, channel)

    result = chunks.iloc[selected].copy()
    result['sparse_score'] = [sparse_score_map.get(int(i), 0.0) for i in selected]
    result['dense_score'] = [dense_score_map.get(int(i), np.nan) for i in selected]
    result['retrieval_score'] = [fused_scores[int(i)] for i in selected]
    result['evidence_channel'] = [selected_channels[int(i)] for i in selected]
    result['rank'] = np.arange(1, len(result) + 1)

    return result[[
        'rank', 'chunk_id', 'document_id', 'party', 'source_type',
        'title', 'source_date', 'source_url', 'division_key',
        'stance_label', 'evidence_channel', 'sparse_score', 'dense_score',
        'retrieval_score', 'text',
    ]]


def motion_query(row, party):
    # 只使用预测前可获得的 motion 信息构造检索查询。
    return clean_extracted_text(
        f"Party: {party}.\\n"
        f"Motion title: {safe_text(row.get('motion_title_clean'))}.\\n"
        f"Motion text: {safe_text(row.get('motion_text_clean'))}.\\n"
        f"Motion type: {safe_text(row.get('motion_type'))}.\\n"
        f"Motion family: {safe_text(row.get('motion_family'))}.\\n"
        f"Policy domain: {safe_text(row.get('policy_domain_primary'))}."
    )"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 12. Retrieval QA on difficult 2024 examples

优先从04E错误文件选择 Green 和其他政党的困难案例。即使这些 motion 已经存在于历史库，检索函数也会排除同 division，并屏蔽查询日期之后的资料。

这里只检查“是否检索到相关、可引用、时间安全的证据”，不根据真实标签调整检索参数。"""
))

cells.append(nbf.v4.new_code_cell(
    """ERROR_PATH = (
    DATA_DIR / 'smoothed_ensemble' /
    'smoothed_ensemble_temporal_predictions.csv'
)

if ERROR_PATH.exists():
    error_frame = pd.read_csv(ERROR_PATH)
    error_frame['motion_date'] = pd.to_datetime(error_frame['motion_date'])
    # 04E错误文件没有完整 motion text，因此从2024 Validation补回查询文本。
    if 'motion_text_clean' not in error_frame.columns:
        query_text_columns = [
            'row_id', 'motion_text_clean', 'legislation_name_clean'
        ]
        available_query_columns = [
            column for column in query_text_columns
            if column in validation.columns
        ]
        error_frame = error_frame.merge(
            validation[available_query_columns],
            on='row_id',
            how='left',
        )
    qa_candidates = error_frame[
        ~error_frame['is_correct'].astype(bool)
    ].copy()
else:
    # 若04E输出不存在，则从2024 Validation选择较晚样本。
    qa_candidates = validation.copy()
    qa_candidates['motion_date'] = pd.to_datetime(
        qa_candidates['motion_date']
    )

qa_examples = (
    qa_candidates.sort_values(['party', 'motion_date'])
    .groupby('party', group_keys=False)
    .head(1)
    .head(4)
)

qa_rows = []
for _, example in qa_examples.iterrows():
    query = motion_query(example, example['party'])
    evidence = retrieve_evidence(
        query=query,
        party=example['party'],
        query_date=example['motion_date'],
        exclude_division_key=example['division_key'],
        top_k=TOP_K,
    )

    print('\\n' + '=' * 90)
    print('Party:', example['party'])
    print('Date:', example['motion_date'].date())
    print('Motion:', safe_text(example.get('motion_title_clean'))[:180])
    display(evidence[[
        'rank', 'evidence_channel', 'source_type', 'title', 'source_date',
        'stance_label', 'sparse_score', 'text',
    ]])

    for _, item in evidence.iterrows():
        qa_rows.append({
            'query_row_id': example.get('row_id', pd.NA),
            'query_party': example['party'],
            'query_date': example['motion_date'],
            'query_division_key': example['division_key'],
            'query_title': example.get('motion_title_clean', pd.NA),
            **item.to_dict(),
        })

retrieval_qa = pd.DataFrame(qa_rows)
retrieval_qa.to_csv(OUTPUT_DIR / 'retrieval_qa_samples.csv', index=False)

if len(retrieval_qa):
    qa_source_coverage = pd.crosstab(
        retrieval_qa['query_party'],
        retrieval_qa['source_type'],
    ).reindex(PARTIES, fill_value=0)

    leakage_checks = pd.Series({
        'future_or_same_date_results': int(
            (pd.to_datetime(retrieval_qa['source_date']) >= retrieval_qa['query_date']).sum()
        ),
        'same_division_results': int(
            retrieval_qa['division_key'].eq(
                retrieval_qa['query_division_key']
            ).fillna(False).sum()
        ),
        'parties_covered': int(retrieval_qa['query_party'].nunique()),
        'source_types_returned': int(retrieval_qa['source_type'].nunique()),
    })
    display(qa_source_coverage)
    display(leakage_checks.to_frame('value'))
    assert leakage_checks['future_or_same_date_results'] == 0
    assert leakage_checks['same_division_results'] == 0"""
))

cells.append(nbf.v4.new_markdown_cell(
    """## 13. Summary for review

运行结束后，把下面 Cell 的文本和任意一组你认为明显不相关的 Top‑K 检索结果发给我。06会在05通过后构建带引用的立场 Agent。"""
))

cells.append(nbf.v4.new_code_cell(
    """policy_parties_covered = int(
    (policy_source_coverage['policy_documents'] > 0).sum()
)
historical_vote_chunks = int(
    chunks['source_type'].eq('historical_vote').sum()
)
manifesto_chunks = int(chunks['source_type'].eq('manifesto').sum())
bill_chunks = int(chunks['source_type'].eq('bill_reference').sum())

if 'retrieval_qa' in globals() and len(retrieval_qa):
    qa_source_coverage = pd.crosstab(
        retrieval_qa['query_party'],
        retrieval_qa['source_type'],
    ).reindex(PARTIES, fill_value=0)
    qa_leakage_summary = pd.Series({
        'future_or_same_date_results': int(
            (
                pd.to_datetime(retrieval_qa['source_date'])
                >= retrieval_qa['query_date']
            ).sum()
        ),
        'same_division_results': int(
            retrieval_qa['division_key'].eq(
                retrieval_qa['query_division_key']
            ).fillna(False).sum()
        ),
    })
else:
    qa_source_coverage = pd.DataFrame()
    qa_leakage_summary = pd.Series(dtype='int64')

summary_lines = [
    '=== RAG KNOWLEDGE BASE SUMMARY FOR REVIEW ===',
    'Notebook version: 05-rag-v2-multichannel',
    f'Documents: {len(documents)}',
    f'Chunks: {len(chunks)}',
    f'Parties with policy documents: {policy_parties_covered}/4',
    f'Manifesto chunks: {manifesto_chunks}',
    f'Historical vote chunks: {historical_vote_chunks}',
    f'Bill reference chunks: {bill_chunks}',
    f'Sparse index shape: {sparse_matrix.shape}',
    f'Sparse vocabulary size: {len(sparse_vectorizer.vocabulary_)}',
    f'Dense embeddings enabled: {ENABLE_DENSE_EMBEDDINGS}',
    f'Retrieval QA queries: {len(qa_examples)}',
    (
        'Retrieval QA rows: '
        f'{len(retrieval_qa) if "retrieval_qa" in globals() else 0}'
    ),
    f'Missing policy documents: {missing_manifestos}',
    'Policy source coverage:',
    policy_source_coverage.to_string(),
    'Retrieval QA source coverage:',
    qa_source_coverage.to_string(),
    'Retrieval QA leakage checks:',
    qa_leakage_summary.to_string(),
    'Knowledge-base source types:',
    chunks['source_type'].value_counts().to_string(),
    '=== END RAG KNOWLEDGE BASE SUMMARY ===',
]

summary_text = '\\n'.join(summary_lines)
print(summary_text)

with open(OUTPUT_DIR / 'rag_knowledge_base_summary.txt', 'w', encoding='utf-8') as file:
    file.write(summary_text)"""
))

nb['cells'] = cells

# 只检查代码 Cell 的语法，不下载资料、不读取数据、不建立索引。
for index, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        compile(cell['source'], f'<cell {index}>', 'exec')

nbf.write(nb, OUTPUT)
print(OUTPUT)
