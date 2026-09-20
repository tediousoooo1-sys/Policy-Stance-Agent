# 数据说明

## 原始投票数据

`data/raw/all.csv` 是项目的原始抓取结果，共 18,739 行、27 列和 4,372 个 division。每行表示一个 `division_key + party` 组合，不是一个独立议案。

主要字段包括：

- `division_key`、`division_name`、`motion_date`
- `motion_title`、`motion_text`、`motion_type`
- `legislation_id`、`legislation_name`、`legislation_slug`
- 全院和政党的赞成、反对、缺席及多数票字段

文件 SHA-256：

```text
3b782f0d73cbecc0457e285b7192b29c9cee540f66423060ec56308e9f7c3a46
```

项目只对 House of Commons 建模。Scotland 数据保留在原始文件中，用于审计。

## Bill 数据

`data/raw/bill_info.csv` 有 129 条 UK Parliament Bill 记录。数量较少，所以它主要提供 Bill 名称、摘要、阶段和 sponsor 背景，不作为每条议案都必须存在的特征，也不直接证明政党立场。

## 官方政策资料

`rag_sources/official_downloads/` 保存四个政党 2024 年 Manifesto 的下载副本。来源 URL 记录在 `processed/rag_v1/official_policy_source_registry.csv` 的生成逻辑和 Notebook 05 中。

| 政党 | 日期 | 来源类型 |
|---|---|---|
| Labour | 2024-06-13 | 官方 Manifesto PDF |
| Conservative | 2024-06-11 | 官方 Manifesto PDF |
| Liberal Democrat | 2024-06-10 | 官方 Manifesto PDF/HTML |
| Green | 2024-06-12 | 官方 Manifesto PDF |

这些文件属于各自发布机构。仓库只将它们用于课程研究和可复现的文本检索实验。

## 时间切分

数据按 `division_key` 和日期切分：

| Split | 日期范围 | 原始 Commons division 数 |
|---|---|---:|
| Train | 2016-01-06 至 2023-12-13 | 1,453 |
| Validation | 2024-01-09 至 2024-12-17 | 200 |
| Test | 2025-01-08 至 2026-04-27 | 438 |

标签审核后，进入监督学习的数据会少于上表。最终三党 Test 有 975 行、354 个 division。一个 division 的所有政党行使用同一个 split。

## 生成文件

`processed/` 下的大多数文件由 Notebook 生成，包括：

- 清洗和标签数据；
- TF-IDF 与 embedding 索引；
- 模型对象和预测；
- RAG 检索结果；
- 人工审核队列；
- API 结果缓存。

这些文件合计约 183 MB，而且可以按 Notebook 顺序重建，因此 Git 默认忽略。`results/` 保存论文、README 和展示需要的少量最终快照。

API 结果可能包含生成文本和本地路径。公开仓库前应检查其中是否有不希望发布的内容。本项目没有把 API Key 写入 Notebook 或输出文件。
