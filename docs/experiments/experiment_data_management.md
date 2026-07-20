# 实验数据管理规范

## 主输出目录

本项目唯一主输出目录：

`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook`

仓库内不创建第二个 `experiment_outputs`。原始学生手册、清洗数据、Chunk 和 QA 继续保存在 `F:\GraduationThesis\datasets`，不移动、不修改。

## 规范运行目录

每个新实验必须创建独立目录：

```text
runs/<experiment_id>/
  config.json
  metrics.json
  per_question_results.jsonl
  summary.csv
  environment.json
  data_manifest.json
  git_commit.txt
  run.log
```

`experiment_id` 格式为：

```text
YYYYMMDDTHHMMSSffffffZ_<task>_<retrieval>_<chunk>_topk<K>_<nonce>
```

时间包含 UTC 微秒，末尾有随机短后缀。目录使用 `mkdir(exist_ok=False)` 创建，已存在目录会直接报错，因此不会同名覆盖。

## 必记字段

`ExperimentRecorder` 自动写入：

- UTC 开始/结束时间和状态；
- Python 版本、解释器、操作系统和全部可见依赖版本；
- Git commit 和实验开始时工作区是否干净；
- 数据源、Chunk、QA 的路径、大小和 SHA-256；
- Chunk 策略、检索方法、Embedding 模型、Top K、QA 版本；
- Hybrid 权重或 RRF 参数；
- 随机种子；
- 汇总指标和逐题结果；
- 失败实验的 `status=failed`、异常类型、消息和 traceback。

无法获得的值写 `unknown` 或空值，不猜测。

## 在现有算法外层记录

公共模块位于 `src/experiments`。现有检索和 RAG 算法无需重写；正式新实验在现有核心函数外增加记录器：

```python
from pathlib import Path

from src.experiments.recorder import ExperimentRecorder

with ExperimentRecorder(
    task_type="retrieval",
    retrieval_method="real_hybrid_weighted_score",
    chunk_strategy="fixed_length",
    top_k=10,
    embedding_model="BAAI/bge-small-zh-v1.5+bm25",
    qa_version="v1",
    generation_script="scripts/evaluate_real_embedding_hybrid_retrieval.py",
    hybrid_parameters={"real_dense_weight": 0.5, "bm25_weight": 0.5},
    random_seed=42,
    inputs={
        "data_source": Path(r"F:\GraduationThesis\datasets\interim\...\cleaned.txt"),
        "chunk_file": Path(r"F:\GraduationThesis\datasets\processed\...\chunks_fixed_length.jsonl"),
        "qa_file": Path(r"F:\GraduationThesis\datasets\processed\...\qa_testset_v1.jsonl"),
    },
) as run:
    metrics, per_question_rows = existing_evaluation_function()
    run.record_results(metrics, per_question_rows)
```

兼容原则：

- 现有 CLI、算法函数和旧结果目录没有被改写。
- 旧脚本的固定默认输出仍属于 legacy 工作流。进行正式新实验时，应让算法返回指标/逐题结果并交给记录器，或者将旧脚本的 `--output-root` 指向本次唯一的新目录；不要再次写入历史固定目录。
- `pathlib` 负责路径，JSON 使用 UTF-8 和 `ensure_ascii=False`。
- JSON、JSONL、CSV 和日志采用同目录临时文件加 `os.replace` 的原子写入。

## 实验总账

总账不是手工唯一来源。运行：

```powershell
.\.venv\Scripts\python.exe scripts\list_experiments.py --rebuild
```

会扫描：

1. `runs\<experiment_id>` 规范目录；
2. 当前已知旧格式检索和 RAG 报告；
3. 逐题文件是否实际存在。

然后原子重建：

`F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\experiment_registry.csv`

旧结果只生成 `legacy=true` 行，不修改源 JSON/JSONL/CSV。

## 第一组规范化冒烟实验

使用现有 BM25 实现和公共记录器运行 `fixed_length + BM25 + top_k=5`：

```powershell
.\.venv\Scripts\python.exe scripts\run_normalized_bm25_smoke.py `
  --chunk-strategy fixed_length `
  --top-k 5 `
  --random-seed 42 `
  --k1 1.5 `
  --b 0.75
```

脚本在内存中构建 BM25 索引，不写入或覆盖旧 `bm25_retrieval` 目录。输出只进入新的 `runs\<experiment_id>`。逐题结果严格采用一题一行；`MRR` 和 `MRR@5` 均按本次实际 `top_k=5` 计算，不把该结果误标为 `MRR@10`。

## 查询实验

```powershell
# 全部
.\.venv\Scripts\python.exe scripts\list_experiments.py

# 按任务
.\.venv\Scripts\python.exe scripts\list_experiments.py --task-type retrieval

# 按检索方法
.\.venv\Scripts\python.exe scripts\list_experiments.py --retrieval-method real_dense

# 按 Chunk 策略
.\.venv\Scripts\python.exe scripts\list_experiments.py --chunk-strategy fixed_length

# 单个实验详情
.\.venv\Scripts\python.exe scripts\list_experiments.py --experiment-id legacy-rag-v3-quality

# CSV 或 JSON 输出
.\.venv\Scripts\python.exe scripts\list_experiments.py --format csv
.\.venv\Scripts\python.exe scripts\list_experiments.py --format json
```

## 失败与恢复

- 上下文中出现异常时，记录器仍写出完整目录，并在 `metrics.json` 写 `status=failed` 和错误信息。
- 如果进程被强制终止，已有 `status=running` 目录会保留，重建总账时不会删除。
- 重新运行必须生成新 ID，不复用失败目录，不手工覆盖。
- Windows 临时目录不可写时，测试使用项目内 `--basetemp .pytest_tmp`。
