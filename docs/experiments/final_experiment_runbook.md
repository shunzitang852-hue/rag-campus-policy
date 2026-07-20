# 最终实验运行手册

## 1. 目的与执行边界

本文用于《基于RAG技术的高校政策知识问答系统构建与检索策略优化研究》的最终规范化实验。所有正式实验必须通过公共 `ExperimentRecorder` 写入新的 `runs\<experiment_id>`，不得覆盖、移动或删除历史结果。

正式矩阵开始前必须满足：

- 仓库处于明确 commit，`git status --short` 为空；
- 全量 pytest、Python compileall、`git diff --check` 通过；
- 数据、Chunk、QA 哈希与本手册固定值一致；
- 本地模型以离线模式加载成功，不触发下载；
- 每种真实检索方法先单独通过一次规范化冒烟；
- 主指标和选择规则已按本文冻结。

## 2. 固定输入与环境

| 项目 | 固定值 |
| --- | --- |
| raw 数据 | `F:\GraduationThesis\datasets\raw\hdu_ies\student_handbook\hdu_ies_student_handbook_source.txt` |
| raw SHA-256 | `83937E33D6E10780DF6BEDEBFB9FD26A2D5D6DE940CF85E8D3F13E3D32165189` |
| cleaned 数据 | `F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\cleaned\hdu_ies_student_handbook_cleaned.txt` |
| cleaned SHA-256 | `13C309934F6413A02D2D4CAFFD3CFD737FB9316B8C4ED285C4443853FDADC2CB` |
| QA | `F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl` |
| QA SHA-256 | `4ECCD47E5FD6FEF7F5BBB77F13FC7646A9EE674961C3BE87A42CA0D7BE2D9CC5` |
| QA 数量 | 40 |
| Embedding 模型 | `BAAI/bge-small-zh-v1.5` |
| `final_top_k` | 5 |
| `dense_candidate_pool_k` | 50 |
| `bm25_candidate_pool_k` | 50 |
| `fusion_candidate_pool_k` | 50 |
| `random_seed` | 42 |
| `rrf_k` | 60 |
| Python | 3.12 系列；实际小版本和依赖由 `environment.json` 固化 |

三种 Chunk 文件及固定 SHA-256：

| chunk_strategy | 文件 | SHA-256 |
| --- | --- | --- |
| fixed_length | `hdu_ies_student_handbook_chunks_fixed_length.jsonl` | `B89EF648A39BAA1E332FA63F6BF76FF6AFDC8B52C8245F233E68999BDDFADC36` |
| article_based | `hdu_ies_student_handbook_chunks_article_based.jsonl` | `8B96C8D166FE05B6443CE35A939AB9CE86FEDE2DBBDCC491FE2514B40EF25803` |
| structure_aware | `hdu_ies_student_handbook_chunks_structure_aware.jsonl` | `78304E60CA6BCC9E2F7544F66D6059525C449E6A24B1BF4A8D2917297AF06ADF` |

BM25 不使用 Embedding；其 `embedding_model` 必须明确记录为 `not_applicable`，不能暗示使用了 BGE。real_dense 和两种 real_hybrid 必须使用上表同一 BGE 模型、同一缓存和同一编码参数。

候选池协议在查看新的 real_dense 或 Hybrid 结果前固定如下：

- Dense 和 BM25 均为 Hybrid 提供各自排序的 Top 50 候选；
- Weighted 和 RRF 必须使用相同问题、相同 Chunk、相同 Dense Top 50、相同 BM25 Top 50；
- 两路候选取并集并计算融合分数后，按确定性规则保留融合 Top 50；最终论文指标只截取该序列的 Top 5；
- `dense_candidate_pool_k=50`、`bm25_candidate_pool_k=50`、`fusion_candidate_pool_k=50` 不得根据实验结果临时调整；
- 若有效语料少于 50，候选池才允许等于实际语料量，并在配置和逐题结果中记录实际深度；
- 单路 real_dense 正式实验同样保留 Top 50 候选，并以其前 5 条计算最终指标。

## 3. 最终检索矩阵

共 12 组，除表中两个自变量外，其余条件全部固定。

| 序号 | chunk_strategy | retrieval_method |
| ---: | --- | --- |
| 1 | fixed_length | bm25 |
| 2 | fixed_length | real_dense |
| 3 | fixed_length | real_hybrid_weighted |
| 4 | fixed_length | real_hybrid_rrf |
| 5 | article_based | bm25 |
| 6 | article_based | real_dense |
| 7 | article_based | real_hybrid_weighted |
| 8 | article_based | real_hybrid_rrf |
| 9 | structure_aware | bm25 |
| 10 | structure_aware | real_dense |
| 11 | structure_aware | real_hybrid_weighted |
| 12 | structure_aware | real_hybrid_rrf |

执行要求：

- 每组独立生成新 `experiment_id`；
- 每组目录必须包含规范规定的 8 个文件；
- 每组总账记录必须为 `legacy=false`、`status=completed`；
- 正式候选记录必须为 `git_worktree_clean=true`、`reproducible=full`；
- 每组必须有 40 条唯一 `question_id` 的逐题结果；
- 所有最终评价指标统一按 Top 5 计算，MRR 截断为 5，不得把 `MRR@5` 标成 `MRR@10`；
- 不重新下载模型，不覆盖旧目录。

## 4. 指标预注册与选择规则

在查看最终新结果前冻结以下规则：

1. 主指标为 `Recall@5`。
2. 次指标为 `MRR@5`。
3. 辅助指标为来源政策命中率、相关 Chunk 命中率和运行耗时。
4. 最优方案先按 `Recall@5` 排序。
5. `Recall@5` 相同时比较 `MRR@5`。
6. 效果基本一致时，才比较运行耗时和方案复杂度。
7. 运行耗时允许合理波动，不能单独作为确定性失败条件。
8. 禁止在实验完成后临时改变主指标、次指标或排序优先级。

“效果基本一致”必须在看结果前另行给出数值阈值并记录；在阈值未冻结前，不得用该表述任意挑选方案。

## 5. Hybrid 固定参数

最终 12 组主矩阵中的 `real_hybrid_weighted` 统一使用：

```text
dense_weight=0.5
bm25_weight=0.5
normalization=per_query_min_max
equal_score_value=0.0
```

从现有 `src/retrieval/hybrid.py` 核实得到的 Weighted 真实公式为：对每个问题分别对 Dense 和 BM25 候选分数做 min-max 归一化，

```text
normalized_source_score(c) =
    (source_score(c) - source_min) / (source_max - source_min)

weighted_score(c) =
    dense_weight * normalized_dense_score(c)
    + bm25_weight * normalized_bm25_score(c)
```

候选集合为 Dense Top 50 与 BM25 Top 50 的并集。某候选未出现在一路结果中时，该路归一化分数按 `0.0` 处理；若一路所有候选原始分数相等，该路全部归一化为 `equal_score_value=0.0`。Adapter 必须直接复用当前核心实现，不得改变归一化范围、缺失候选语义、权重或打分公式。

`real_hybrid_rrf` 统一使用：

```text
rrf_k=60
```

当前核心融合实现的确定性排序规则不是简单的“总分降序、chunk_id 升序”，而是：

1. `hybrid_score` 降序；
2. `bm25_score` 降序；
3. `dense_score` 降序；
4. `retrieved_chunk_id` 升序。

Weighted 和 RRF Adapter 必须保持这一已存在且确定性的 tie-break，不得静默改成另一套规则。两者必须使用相同的 real_dense、BM25 候选输入及相同候选池规则，避免融合方式之外的隐含变量。

## 6. Hybrid 权重补充实验

该实验仅使用 `fixed_length`，且必须在 12 组主矩阵完成和核验后运行。

| 序号 | dense_weight | bm25_weight |
| ---: | ---: | ---: |
| 1 | 0.3 | 0.7 |
| 2 | 0.4 | 0.6 |
| 3 | 0.5 | 0.5 |
| 4 | 0.6 | 0.4 |
| 5 | 0.7 | 0.3 |

配置、总账和论文表格中必须使用完整字段名 `dense_weight`、`bm25_weight`，禁止只写含糊的“0.3/0.7”。

## 7. RAG v1/v2/v3 控制变量

只有最终检索方案确定后才运行 RAG 最终实验。三组必须固定：

- 同一最终 `retrieval_method`；
- 同一最终 `chunk_strategy`；
- `top_k=5`；
- 同一 QA 文件及 SHA-256；
- 同一数据和 Chunk 版本；
- 同一模型、Python 和依赖环境；
- 同一随机种子；
- 同一检索结果或可证明完全一致的检索过程。

三组只允许改变明确记录的 RAG 版本差异：

- RAG v1：现有 v1 规则式 extractive 生成；
- RAG v2：现有 v2 答案质量改进规则；
- RAG v3：现有 v3 定向结构化规则。

不得在比较 RAG 版本时同时改变检索方法、Chunk、Top K、权重、模型或 QA。

## 8. 失败、重跑、作废与保留

- 失败：记录器保留目录，并写 `status=failed`、异常类型、消息和 traceback。
- 重跑：必须生成新 `experiment_id`，并在参数中记录 `rerun_of=<原 experiment_id>` 和重跑原因。
- 结果不一致：先停止后续矩阵，比较配置、环境、输入哈希、逐题结果和 Git 状态；原因未定位前不得继续。
- 作废：不得删除或改写原目录。当前 Recorder 尚无正式的不可变作废 sidecar 机制；如确需作废，必须先实现并测试独立的 invalidation 记录，再由 registry 显式显示 `invalid`。禁止手工隐藏记录。
- 保留：dirty、failed、重复和被作废的运行都保留用于审计，但不得进入正式最优方案候选集。
- 正式候选集默认只接受 `legacy=false`、`status=completed`、`git_worktree_clean=true`、`reproducible=full`。

当前论文原始导出仍展示所有历史记录，尚未实现正式候选优先筛选。后续若增加筛选，必须保留全量审计表，并额外生成候选表，不能隐藏 legacy 或 dirty 记录。

## 9. 分阶段执行门禁

1. BM25 clean-worktree 冒烟通过。
2. real_dense 离线冒烟通过。
3. real_hybrid_weighted 离线冒烟通过。
4. real_hybrid_rrf 离线冒烟通过。
5. 四种方法逐题结构和指标口径一致。
6. 再逐组执行 12 组矩阵，每完成一组立即核验并重建 registry。
7. 主矩阵完成后运行 Hybrid 权重补充实验。
8. 选定最终检索方案后运行 RAG v1/v2/v3。
9. 最后导出论文表格、错误统计、人工复核包和最优方案选择说明。

任一门禁失败时停止后续阶段，不得以批量运行掩盖单组问题。

## 10. 真实检索规范化适配前置检查

本节只记录入口和适配条件，不表示已经运行最终矩阵。

### 10.1 real_dense

- 现有入口：`scripts/build_real_embedding_dense_retrieval.py` 和 `scripts/evaluate_real_embedding_dense_retrieval.py`，配置为 `configs/retrieval_dense_real_embedding.yaml`。
- 可复用实现：`SentenceTransformersProvider`、余弦相似度检索、既有三种 512 维 Chunk 向量文件。
- 最小适配：增加只运行一个 Chunk 策略的规范化 runner；强制离线加载已验证的本地 snapshot；将 top_k 固定为 5；复用统一 `MRR@5` 指标；通过 `ExperimentRecorder` 写入 8 个标准文件；把既有向量文件作为输入资产记录路径和 SHA-256。
- 预计逐题字段：`question_id`、问题、命中状态、首个相关排名、来源政策命中、相关 Chunk 命中，以及包含 `rank`、`retrieved_chunk_id`、`score` 的 top-5 结果。
- 风险：现有评估入口会重新构建全部三种向量并写固定 legacy 目录，且当前使用 top-10/MRR@10；配置中的 `local_model_path` 不存在，Provider 也没有显式传入 `local_files_only=True`。因此不得直接把现有入口作为规范化冒烟命令。
- 冒烟条件：本地模型和向量资产可用，但须先完成上述最小 runner 适配。

### 10.2 real_hybrid_weighted

- 现有入口：`scripts/evaluate_real_embedding_hybrid_retrieval.py`；融合实现为 `src/retrieval/real_hybrid.py` 的 `fuse_real_weighted_score`。
- 固定参数：规范化配置必须记录 `dense_weight=0.5`、`bm25_weight=0.5`、`normalization=per_query_min_max`。现有函数参数 `real_dense_weight` 仅作为内部兼容名，不得成为最终总账中的含糊字段。
- 最小适配：读取同一规范化 real_dense/BM25 候选输入；冻结相同候选池深度；将 canonical `dense_weight` 映射到现有内部参数；只输出 top-5；复用统一指标和 Recorder。
- 预计逐题字段：除公共字段外，top-5 结果应包含 `real_dense_score`、`bm25_score`、`hybrid_score`、`fusion_method=weighted_score`。
- 风险：现有脚本固定读取 legacy top-10 文件、同时执行两种融合和三种 Chunk，并写固定目录；候选池深度尚未预注册。候选池过浅会改变融合结果。
- 冒烟条件：融合函数已存在且有测试基础，但须先有规范化 real_dense 输入并冻结候选池规则。

### 10.3 real_hybrid_rrf

- 现有入口：`scripts/evaluate_real_embedding_hybrid_retrieval.py`；融合实现为 `src/retrieval/real_hybrid.py` 的 `fuse_real_rrf`。
- 固定参数：`rrf_k=60`。
- 最小适配：与 weighted 使用完全相同的 real_dense/BM25 候选输入和候选池深度；只输出 top-5；复用统一指标和 Recorder。
- 预计逐题字段：除公共字段外，top-5 结果应包含两个来源排名/分数、`hybrid_score`、`fusion_method=rrf` 和 `rrf_k`。
- 风险：与 weighted 相同，现有入口依赖 legacy top-10 文件且使用 MRR@10；若候选池或 tie-break 规则不同，将引入融合方式之外的变量。
- 冒烟条件：RRF 融合函数和确定性 tie-break 已存在，但须先完成统一候选输入的规范化适配。

### 10.4 本地 BGE 资产

已在禁止联网的环境变量下，以 CPU 从以下本地 snapshot 成功加载并编码一条文本：

```text
F:\GraduationThesis\models\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620
```

正式模型身份固定为：

| 字段 | 固定值 |
| --- | --- |
| `model_id` | `BAAI/bge-small-zh-v1.5` |
| `resolved_snapshot_path` | `F:\GraduationThesis\models\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620` |
| snapshot revision | `7999e1d3359715c523056ef9478215996d62a620` |
| snapshot 文件数 | 11 |
| snapshot 目录指纹 | `D0A8E83A6221E4B4F6347279D8DA7B6AB46DE0389DB20B4C376FB4E8D27D5314` |
| 目录指纹算法 | 对按相对路径排序的 `relative_path<TAB>file_sha256` 记录以 LF 连接后计算 SHA-256 |
| `embedding_dimension` | 512 |

输出为有限值、L2 范数为 1 的 512 维向量。正式 runner 必须同时设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，并向模型加载接口显式传入 `local_files_only=True`。不得仅依赖缓存碰巧命中。

禁止使用不存在的旧路径：

```text
F:\GraduationThesis\models\BAAI\bge-small-zh-v1.5
```

如果 snapshot 文件缺失、本地加载失败或依赖试图联网，实验必须立即以 `status=failed` 结束；禁止自动联网、自动下载或隐式回退到其他模型或 deterministic hash embedding。

现有三种 Chunk 向量资产均为 512 维：

| chunk_strategy | 行数 | SHA-256 |
| --- | ---: | --- |
| fixed_length | 244 | `80DF9F36DE30BBD86E440F01F73E1CDBEB1497ABA3F917A9083D303798FF5A53` |
| article_based | 358 | `203086025E91023A3D2A8392CF9B7F36CB99682A318F3D4DAB7AC8F920ADA6FC` |
| structure_aware | 358 | `E18939753506AE31E00C59E3008ED8519111CF56B7BD9E3CF2BB524EC5CDBAE6` |
