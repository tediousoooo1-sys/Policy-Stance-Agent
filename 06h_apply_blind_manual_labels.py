from pathlib import Path

import pandas as pd


# 该脚本记录 Codex 对06h盲审队列的人工判断。
# 它只依据查询与证据正文，不读取检索策略、排名或分数。
PROJECT_DIR = Path.cwd()
QUEUE_PATH = (
    PROJECT_DIR / 'processed' / 'rag_hybrid_v7'
    / 'manual_hybrid_review_queue_v7.csv'
)
OUTPUT_PATH = (
    PROJECT_DIR / 'processed' / 'rag_hybrid_v7'
    / 'manual_hybrid_review_completed_v7.csv'
)


# 直接证据必须针对同一具体政策，并且能够提供政党方向。
DIRECTIONAL_DIRECT = {
    '616c7f8c2d6ac354',  # Conservative明确反对私校学费VAT。
    'dc3aa4736c459667',  # Conservative对私校学费VAT具体条款投反对票。
}


# 背景资料解释法案或制度，但不能证明政党立场。
BACKGROUND_REFERENCE = {
    '844f63bed4effbdf',  # Scotland Self-Determination Bill参考资料。
    'f012037140967794',  # Leasehold Reform (Ground Rent) Act参考资料。
    '243179f28b737016',  # 同名Employment Rights Bill参考资料，版本并不等同当前法案。
    '15b2326b0a2555f6',  # Non-Domestic Rating Act参考资料。
}


# 相关上下文与查询处于同一Bill、政策工具或紧密相邻问题，
# 但不足以直接证明对当前具体条款的方向。
RELEVANT_CONTEXT = {
    # Q001 NHS dentistry：广义NHS战略与健康不平等背景。
    '3a92850c5127aa19',
    '7b4dfe9ba5065237',
    # Q002 Rwanda costs：同一Rwanda法案的总体立场。
    '99c01ff5720b9617',
    # Q003 Scotland self-determination：相关的苏格兰议会同意与Scotland Act权限。
    'dd08ce67735619f6',
    'efe0ad15afd3b26e',
    # Q007 freehold first refusal：相邻的住房、租赁和ground-rent改革。
    '2aa13520b39d2604',
    '49a0242003051e75',
    'f86d4916ba59906e',
    # Q008 peppercorn rent：同一leasehold领域，但不能确定当前删除修正案方向。
    '1deb295031f0dec5',
    '430a1b54b329dcad',
    # Q010 Safety of Rwanda amendment：同一法案的总体立场。
    '777e8e5b94f0f028',
    # Q011 CPTPP impact report：相似贸易协定影响评估与贸易审查。
    '48a2c416e026e557',
    'bdb8db4c8212c7d0',
    # Q012 ISC engagement report：同一Investigatory Powers政策体系。
    '12209b80272e656b',
    # Q015 renters possession assessment：相邻住房、租赁与leasehold保护。
    '3da4bd181ec62582',
    '4b60fafafec7b6bd',
    '631cd157102b5d13',
    'a1286d4cbe1552a8',
    # Q019 ninja swords：相邻的offensive-weapons立法。
    '09f4689d887d031b',
    # Q020 immigration tribunal rules：相邻的tribunal与immigration instruments。
    '1754e4176523ea0d',
    '42f50f1b970cf531',
    '52232ae725b8b89c',
    '59253c1a166679ed',
    '9502413bf60abc61',
    # Q021 rail ticketing report：铁路公共运营和同一法案总体立场。
    '0661d5569d9e0cdf',
    '8037a43071a8174f',
    # Q023 household energy-bill target：同一GB Energy Bill和相似降价政策。
    '655a56995684058c',
    '9446f178b2caedc9',
    # Q024 GB Energy delivery review：同一法案总体立场。
    '12d753bd16f4218f',
    # Q025 BADR CGT increase：相邻但不相同的CGT rate政策。
    '5e6ad6e1d0ae88e9',
    '83d76d23a4a3c87e',
    'e3882088ba4c355b',
    # Q026 passenger railway Lords amendment：同一法案总体立场。
    'a19c8167d6c60382',
    # Q028 terrorism-premises regulator：相邻的terrorism-content监管机制。
    'e85c78199f5697bd',
    # Q030 employer NIC amendment：同一法案及相邻NIC政策。
    '1b1407ea48df72b2',
    '6c1e7942ffcaaf20',
    'cd7beaea8d847cb5',
    'f37cd219210ef8e6',
}


queue = pd.read_csv(QUEUE_PATH, low_memory=False)
all_candidate_ids = set(queue['candidate_id'].astype(str))

# 三组人工标签不得重叠；未进入三组的证据严格标记为reject。
label_sets = [DIRECTIONAL_DIRECT, BACKGROUND_REFERENCE, RELEVANT_CONTEXT]
for index, left in enumerate(label_sets):
    for right in label_sets[index + 1:]:
        overlap = left & right
        if overlap:
            raise ValueError(f'Overlapping manual labels: {sorted(overlap)}')

known_labels = set().union(*label_sets)
unknown_candidate_ids = known_labels - all_candidate_ids
if unknown_candidate_ids:
    raise ValueError(
        f'Manual labels contain IDs absent from the blind queue: '
        f'{sorted(unknown_candidate_ids)}'
    )


def assign_role(candidate_id):
    # 按最严格的人工证据定义分配角色。
    candidate_id = str(candidate_id)
    if candidate_id in DIRECTIONAL_DIRECT:
        return 'directional_direct'
    if candidate_id in BACKGROUND_REFERENCE:
        return 'background_reference'
    if candidate_id in RELEVANT_CONTEXT:
        return 'relevant_context'
    return 'reject'


def assign_notes(row):
    # 备注说明角色含义，不引用检索方法或排名。
    role = row['manual_role']
    if role == 'directional_direct':
        return 'Same concrete policy and usable party-direction evidence.'
    if role == 'background_reference':
        return 'Useful institutional or bill background, but not party-direction evidence.'
    if role == 'relevant_context':
        return 'Related bill, instrument or policy area; insufficient for the exact proposition.'
    return 'Different policy, generic procedure, or insufficient semantic specificity.'


queue['manual_role'] = queue['candidate_id'].map(assign_role)
queue['manual_notes'] = queue.apply(assign_notes, axis=1)

valid_roles = {
    'directional_direct', 'background_reference', 'relevant_context', 'reject'
}
if not queue['manual_role'].isin(valid_roles).all():
    raise ValueError('At least one row has an invalid manual role.')
if not queue['candidate_id'].is_unique:
    raise ValueError('Candidate IDs must be unique in the blind-review file.')

queue.to_csv(OUTPUT_PATH, index=False)

print('Completed blind-review rows:', len(queue))
print(queue['manual_role'].value_counts().to_string())
print('Output:', OUTPUT_PATH)


# 使用盲审标签计算五种策略的前三名质量。
STRATEGY_MAP_PATH = (
    PROJECT_DIR / 'processed' / 'rag_hybrid_v7'
    / 'candidate_strategy_map_v7.csv'
)
METRICS_PATH = (
    PROJECT_DIR / 'processed' / 'rag_hybrid_v7'
    / 'strategy_metrics_v7.csv'
)
MANIFEST_PATH = (
    PROJECT_DIR / 'processed' / 'rag_hybrid_v7'
    / 'run_manifest_v7.json'
)

strategy_map = pd.read_csv(STRATEGY_MAP_PATH, low_memory=False)
scored = strategy_map.merge(
    queue[['candidate_id', 'manual_role']],
    on='candidate_id',
    how='left',
    validate='many_to_one',
)
scored['is_direct'] = scored['manual_role'].eq('directional_direct')
scored['is_useful'] = scored['manual_role'].isin([
    'directional_direct', 'relevant_context'
])
scored['is_reject'] = scored['manual_role'].eq('reject')

metric_rows = []
for strategy, group in scored.groupby('strategy'):
    query_direct = group.groupby('query_id')['is_direct'].any()
    query_useful = group.groupby('query_id')['is_useful'].any()
    metric_rows.append({
        'strategy': strategy,
        'dense_weight': float(group['dense_weight'].iloc[0]),
        'evidence_rows': int(len(group)),
        'direct_precision_at_3': float(group['is_direct'].mean()),
        'useful_precision_at_3': float(group['is_useful'].mean()),
        'reject_rate_at_3': float(group['is_reject'].mean()),
        'direct_query_coverage': float(query_direct.mean()),
        'useful_query_coverage': float(query_useful.mean()),
    })

strategy_metrics = pd.DataFrame(metric_rows).sort_values(
    ['direct_precision_at_3', 'direct_query_coverage', 'useful_precision_at_3'],
    ascending=False,
)
strategy_metrics.to_csv(METRICS_PATH, index=False)

selected_strategy = strategy_metrics.iloc[0]['strategy']
selected_row = strategy_metrics.iloc[0]
baselines = strategy_metrics[
    strategy_metrics['strategy'].isin(['tfidf_only', 'dense_only'])
]
best_baseline_precision = baselines['direct_precision_at_3'].max()
best_baseline_coverage = baselines['direct_query_coverage'].max()
hybrid_acceptance = bool(
    str(selected_strategy).startswith('hybrid_')
    and selected_row['direct_precision_at_3'] >= best_baseline_precision + 0.05
    and selected_row['direct_query_coverage'] >= best_baseline_coverage
)

if MANIFEST_PATH.exists():
    import json

    with open(MANIFEST_PATH, encoding='utf-8') as file:
        manifest = json.load(file)
    manifest.update({
        'evaluation_status': 'manual_evaluation_complete',
        'completed_label_rows': int(len(queue)),
        'selected_strategy': selected_strategy,
        'hybrid_acceptance': hybrid_acceptance,
    })
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

print('\nStrategy metrics:')
print(strategy_metrics.round(4).to_string(index=False))
print('Selected strategy:', selected_strategy)
print('Hybrid acceptance:', hybrid_acceptance)
print('Metrics:', METRICS_PATH)
