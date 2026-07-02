# Project Paths

This file records the fixed paths used by the undergraduate thesis project:

《基于RAG技术的高校政策知识问答系统构建与检索策略优化研究》

## Fixed Root

`F:\GraduationThesis`

All formal project files for this stage are stored under the F drive. No formal project directory is created under `C:\`, Desktop, Downloads, or temporary folders.

## Directory Usage

| Directory | Fixed Path | Usage |
| --- | --- | --- |
| Source project | `F:\GraduationThesis\rag-campus-policy` | Source code, configuration files, tests, experiment scripts, README, and project documents. |
| Datasets | `F:\GraduationThesis\datasets` | Raw policy documents, cleaned text, chunk results, test question sets, and annotated data. |
| Models | `F:\GraduationThesis\models` | Local embedding models, reranking models, and other model files. |
| Cache | `F:\GraduationThesis\cache` | Model download cache, Python cache, and runtime cache. |
| Experiment outputs | `F:\GraduationThesis\experiment_outputs` | Retrieval experiment results, evaluation metrics, logs, tables, and charts. |
| Thesis documents | `F:\GraduationThesis\thesis_documents` | Proposal materials, experiment records, stage reports, thesis drafts, and references. |

## Dataset Structure

| Directory | Fixed Path | Usage |
| --- | --- | --- |
| HDU raw policy files | `F:\GraduationThesis\datasets\raw\hdu` | Raw Hangzhou Dianzi University policy files after manual source verification. |
| HDU metadata | `F:\GraduationThesis\datasets\metadata\hdu` | Metadata templates, candidate lists, and verification records for HDU policy collection. |
| HDU interim data | `F:\GraduationThesis\datasets\interim\hdu` | Intermediate files generated during later cleaning or parsing stages. |
| HDU processed data | `F:\GraduationThesis\datasets\processed\hdu` | Cleaned text, normalized data, chunks, and downstream dataset artifacts after verification. |

## Source Project Structure

| Directory or File | Usage |
| --- | --- |
| `src` | Future source code. |
| `tests` | Future test code. |
| `configs` | Configuration files. |
| `scripts` | Experiment and utility scripts. |
| `notebooks` | Research notebooks. |
| `docs` | Project development documents. |
| `logs` | Local logs, ignored by Git. |
| `requirements.txt` | Python dependency list placeholder. |
| `README.md` | Project overview and stage description. |
| `PROJECT_PATHS.md` | Fixed path documentation. |
| `environment_check.md` | Environment check record. |
| `.gitignore` | Git ignore rules. |
