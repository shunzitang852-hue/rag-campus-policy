# 研究对象调整记录

## 调整日期

2026-07-03

## 原研究对象

杭州电子科技大学。

该对象对应的候选检索记录、下载核验清单和外部 `datasets\metadata\hdu` 目录保留为历史状态，不直接改写为杭州电子科技大学信息工程学院材料。

## 新研究对象

杭州电子科技大学信息工程学院。

简称：杭电信工

英文标识建议：Hangzhou Dianzi University Information Engineering School

内部机构代码：`hdu_ies`

论文正式题目：

《基于RAG技术的高校政策知识问答系统构建与检索策略优化研究》

本次调整仅涉及论文项目的主要案例研究对象和政策语料范围。此前曾误将案例对象追加至论文正式题目，现已纠正。论文系统登记题目未发生变化。

## 调整原因

杭州电子科技大学本部部分政策文件难以稳定获取和核验，而杭州电子科技大学信息工程学院学生手册及相关制度材料相对完整。为保证数据来源真实、政策主体准确、语料规模满足RAG实验要求，将主要案例对象调整为杭州电子科技大学信息工程学院。

## 已修改文件

- `README.md`
- `PROJECT_PATHS.md`
- `configs/institution.yaml`
- `docs/data/hdu_ies_policy_dataset_specification.md`
- `docs/data/hdu_ies_policy_verification_checklist.md`
- `docs/data/hdu_policy_candidate_search_log.md`
- `docs/data/hdu_first_batch_download_manifest.md`
- `docs/research_object_change_log.md`

## 未修改的原始文件

未发现已下载的原始政策正文文件。

以下目录原则上只读，后续不得直接全局替换其中的政策正文：

- `F:\GraduationThesis\datasets\raw`
- 原始 PDF
- 原始 DOC/DOCX
- 原始 TXT
- 官网下载文件
- 人工扫描文件

## 数据迁移情况

已建立杭州电子科技大学信息工程学院数据目录：

- `F:\GraduationThesis\datasets\raw\hdu_ies`
- `F:\GraduationThesis\datasets\raw\hdu_ies\student_handbook`
- `F:\GraduationThesis\datasets\metadata\hdu_ies`
- `F:\GraduationThesis\datasets\interim\hdu_ies`
- `F:\GraduationThesis\datasets\processed\hdu_ies`
- `F:\GraduationThesis\datasets\evaluation\hdu_ies`

已建立待核验原始文件目录：

- `F:\GraduationThesis\datasets\raw\pending_verification`

原 `hdu` 目录保留为历史候选记录。当前未发现可确认属于杭州电子科技大学信息工程学院的本地原始学生手册文件，因此未执行原始文件复制，也未生成学生手册 SHA-256。

## 待核验问题

1. 需要用户提供或确认“杭州电子科技大学信息工程学院学生手册”的 TXT、PDF 或 DOCX 原始文件位置。
2. 需要补充学生手册正式发布页、下载来源或纸质版本信息。
3. 需要确认学生手册中各制度的发布主体、适用范围、发布日期、文号和有效性。
4. 需要将杭州电子科技大学本部候选与杭州电子科技大学信息工程学院候选彻底分开管理。
5. 需要基于杭电信工实际政策重建测试问题集，不能只替换学校名称。

## 对后续实验的影响

1. 后续数据采集、元数据登记、问题集设计和系统展示默认使用 `hdu_ies`。
2. 历史 `hdu` 候选记录不得直接并入 `hdu_ies` 语料。
3. 国家及教育部层面政策只能作为上位政策和补充语料。
4. 检索实验的评测问题、标准答案、条款和页码必须重新基于杭州电子科技大学信息工程学院材料核验。
5. 系统回答必须展示政策发布机构、文件名称、适用范围、引用条款或片段、页码或章节位置和来源核验状态。
