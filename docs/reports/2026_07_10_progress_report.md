# 2026-07-10 阶段检查项目进度报告

## 一、项目基本信息

- 论文题目方向：基于 RAG 技术的高校政策知识问答系统构建与检索策略优化研究。
- 研究对象：杭州电子科技大学信息工程学院学生手册相关制度文本。
- 当前项目路径：`F:\GraduationThesis\rag-campus-policy`
- 外部数据根目录：`F:\GraduationThesis\datasets`
- 外部实验结果根目录：`F:\GraduationThesis\experiment_outputs`
- 模型与缓存目录：`F:\GraduationThesis\models`、`F:\GraduationThesis\cache`

当前阶段已经完成学生手册语料导入、清洗、Chunk 切分、问答测试集构建、检索实验基线与 retrieval-only QA Demo。需要特别说明的是，当前 Demo 仅展示检索结果，尚未实现完整 RAG 生成答案流程。

## 二、当前 Git 提交状态

- 当前 HEAD：`89579b935490f6951519735c459893cef88361bb`
- 当前 HEAD 提交说明：`feat(demo): 添加学生手册检索问答演示`
- 工作区状态：报告整理前工作区干净。
- 全量测试结果：`99 passed`
- 测试命令：`.\.venv\Scripts\python.exe -m pytest`

关键阶段提交如下：

| 提交哈希 | 提交说明 |
| --- | --- |
| `89579b9` | `feat(demo): 添加学生手册检索问答演示` |
| `f7f9468` | `feat(retrieval): 构建真实 Embedding Hybrid 检索实验` |
| `49a9c01` | `feat(retrieval): 运行真实中文 Embedding 检索实验` |
| `f1721cb` | `feat(retrieval): 支持真实中文 Embedding 检索实验` |
| `dacf1d7` | `feat(retrieval): 构建 Hybrid Search 检索基线` |
| `8536184` | `feat(retrieval): 构建 BM25 检索基线` |
| `8d8cbaf` | `feat(retrieval): 构建 Dense Search 检索基线` |
| `ffa3831` | `feat(eval): 构建信工学生手册问答测试集` |
| `5652976` | `feat(chunking): 完成信工学生手册 Chunk 切分` |
| `1663884` | `feat(data): 完成信工学生手册正文清洗` |
| `c699c6e` | `feat: 完成杭电信工学生手册结构化解析与清洗` |

项目未将 `models`、`cache`、`experiment_outputs` 等外部结果目录提交到 Git。

## 三、数据处理进度

| 阶段 | 当前状态 | 关键说明 |
| --- | --- | --- |
| raw 导入 | 已完成 | 已将学生手册源文本导入外部数据目录，并保留原始哈希用于校验。 |
| 哈希校验 | 已完成 | raw、cleaned、三种 Chunk 文件和 QA 测试集均已记录 SHA-256。 |
| 元数据 | 已完成 | 已生成学生手册相关元数据和导入记录。 |
| 制度目录 | 已完成 | 学生手册制度目录已提取，当前 QA 测试集覆盖 32 项制度。 |
| 正文清洗 | 已完成 | 已形成清洗后的全文和按制度拆分的正文文本。 |
| Chunk 切分 | 已完成 | 已生成 `fixed_length`、`article_based`、`structure_aware` 三种策略的 Chunk 文件。 |
| QA 测试集 | 已完成第一版 | 当前测试集包含 40 个问题，关联 evidence_text 和三种策略下的相关 chunk_id。 |

## 四、关键数据文件与哈希

| 文件类别 | 文件路径 | SHA-256 |
| --- | --- | --- |
| raw | `F:\GraduationThesis\datasets\raw\hdu_ies\student_handbook\hdu_ies_student_handbook_source.txt` | `83937E33D6E10780DF6BEDEBFB9FD26A2D5D6DE940CF85E8D3F13E3D32165189` |
| cleaned | `F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\cleaned\hdu_ies_student_handbook_cleaned.txt` | `13C309934F6413A02D2D4CAFFD3CFD737FB9316B8C4ED285C4443853FDADC2CB` |
| fixed_length Chunk | `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunks_fixed_length.jsonl` | `B89EF648A39BAA1E332FA63F6BF76FF6AFDC8B52C8245F233E68999BDDFADC36` |
| article_based Chunk | `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunks_article_based.jsonl` | `8B96C8D166FE05B6443CE35A939AB9CE86FEDE2DBBDCC491FE2514B40EF25803` |
| structure_aware Chunk | `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks\hdu_ies_student_handbook_chunks_structure_aware.jsonl` | `78304E60CA6BCC9E2F7544F66D6059525C449E6A24B1BF4A8D2917297AF06ADF` |
| QA 测试集 | `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl` | `4ECCD47E5FD6FEF7F5BBB77F13FC7646A9EE674961C3BE87A42CA0D7BE2D9CC5` |

## 五、Chunk 策略与统计

| Chunk 策略 | chunk_count | avg_chars | max_chars | 覆盖率 | 空 Chunk |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_length` | 244 | 452.59 | 500 | 1.000 | 0 |
| `article_based` | 358 | 251.68 | 900 | 1.000 | 0 |
| `structure_aware` | 358 | 251.68 | 900 | 1.000 | 0 |

说明：覆盖率为基于已解析正文结构记录的唯一覆盖率，不按重叠 Chunk 字符数重复累计。`fixed_length` 是固定长度控制策略；`article_based` 和 `structure_aware` 更贴近制度结构，但当前统计结果在本语料上数量一致。

## 六、QA 测试集统计

- 问题总数：40
- 覆盖制度数量：32
- 平均 evidence_text 长度：116.0 字符
- 缺失 evidence_text 数量：0
- 缺失相关 Chunk 关联数量：0
- 三种 Chunk 策略均缺失关联的数量：0
- 重复问题数量：0
- 异常数量：0

问题类型分布：

| 类型 | 数量 |
| --- | ---: |
| `fact_lookup` | 6 |
| `condition` | 6 |
| `procedure` | 6 |
| `responsible_body` | 6 |
| `time_limit` | 6 |
| `qualification` | 5 |
| `comparison` | 5 |

难度分布：

| 难度 | 数量 |
| --- | ---: |
| `easy` | 10 |
| `medium` | 16 |
| `hard` | 14 |

QA 测试集保留了 `evidence_text`，并通过 `related_chunk_ids_fixed_length`、`related_chunk_ids_article_based`、`related_chunk_ids_structure_aware` 关联三种 Chunk 策略下的相关片段。

## 七、检索实验结果

以下表格汇总 Recall@5。Dense Hash 是用于验证检索流程的确定性哈希向量基线，不是真实语义 Embedding。Real Dense 使用真实中文 Embedding 模型 `BAAI/bge-small-zh-v1.5`，embedding 维度为 512。Real Hybrid weighted 是 Real Dense 与 BM25 按 0.5/0.5 进行 query 内归一化加权融合。

| Chunk 策略 | Dense Hash R@5 | BM25 R@5 | old Hybrid weighted R@5 | Real Dense R@5 | Real Hybrid weighted R@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_length` | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 |
| `article_based` | 0.600 | 0.675 | 0.725 | 0.750 | 0.750 |
| `structure_aware` | 0.600 | 0.675 | 0.725 | 0.750 | 0.750 |

相关实验输出路径如下：

- Dense Hash：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval`
- BM25：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval`
- old Hybrid：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval`
- Real Dense：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding`
- Real Hybrid：`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval_real_embedding`

真实中文 Embedding 已成功运行，模型信息如下：

- Provider：`sentence_transformers`
- Model：`BAAI/bge-small-zh-v1.5`
- Embedding dimension：512
- Model cache：`F:\GraduationThesis\models`
- HF cache：`F:\GraduationThesis\cache\huggingface`

## 八、Demo 说明

当前已经完成 retrieval-only QA Demo，文件如下：

- Demo 脚本：`scripts/demo_retrieval_qa.py`
- 可复用接口：`src/retrieval/demo.py`
- 测试文件：`tests/test_retrieval_demo.py`
- 说明文档：`docs/retrieval/retrieval_qa_demo.md`

支持的检索方法：

- `bm25`
- `real_dense`
- `real_hybrid_weighted`

支持的 Chunk 策略：

- `fixed_length`
- `article_based`
- `structure_aware`

示例命令：

```powershell
.\.venv\Scripts\python.exe scripts\demo_retrieval_qa.py --query "学生申请解除处分需要满足什么条件？" --method real_dense --strategy fixed_length --top-k 5
```

示例问题“学生申请解除处分需要满足什么条件？”在 `real_dense` + `fixed_length` 下的 Top 3 结果概述：

1. `杭州电子科技大学信息工程学院学生解除处分管理办法`：命中解除处分申请、学院审核、钉钉 OA 审批、学生工作部审核批准等程序内容。
2. `普通高等学校学生管理规定`：命中处分类型、严重违纪处分等相关条款。
3. `普通高等学校学生管理规定`：命中处分期限、解除处分后权益不再受原处分影响、处分材料归档等内容。

该 Demo 明确输出“当前为检索结果展示，不是大模型生成答案。”当前阶段不调用大模型 API，不生成最终自然语言答案，不运行 Reranker，也不读取固定 QA 测试集答案来回答新问题。

## 九、当前未完成事项

- 完整 RAG 生成模块尚未完成。
- Reranker 尚未完成。
- Web 或 Streamlit 前端尚未完成。
- 多语料扩展尚未完成。
- 官网逐项制度核验仍需人工完成。
- 当前 QA 测试集仍属于第一版，需要随论文实验设计继续复核问题合理性和标准答案表述。

## 十、下一步计划

1. 整理现有检索实验结果表，形成论文实验章节草稿。
2. 在不破坏现有检索基线的前提下，增加 RAG 生成模块。
3. 增加答案引用、来源制度标题、Chunk ID 和原文片段展示。
4. 可选增加 Reranker，并与现有 Dense/BM25/Hybrid 结果区分记录。
5. 可选实现简单 Web Demo 或 Streamlit Demo，用于毕业设计演示。
6. 对学生手册制度逐项进行人工来源核验，记录官网或学院正式发布依据。

## 十一、本轮报告整理边界

本轮仅新增阶段检查 Markdown 报告，不做新实验，不改数据，不重跑模型，不安装新依赖。未修改 raw、cleaned、Chunk、QA、`experiment_outputs`、`models` 或 `cache`。
