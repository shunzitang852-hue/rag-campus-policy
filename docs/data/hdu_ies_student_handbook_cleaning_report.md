# 杭电信工学生手册正文清洗报告

## 一、清洗范围

- 输入文件：`F:\GraduationThesis\datasets\raw\hdu_ies\student_handbook\hdu_ies_student_handbook_source.txt`
- 输出文件：`F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\cleaned\hdu_ies_student_handbook_cleaned.txt`
- 清洗原则：不改写政策正文，只排除前言和目录，并折叠连续多余空行。
- 前言处理：已单独识别为 `preface`，本轮不进入正式检索正文。
- 目录处理：目录不进入 cleaned 正文，已记录删除行号范围和字符数。

## 二、统计结果

| 指标 | 数值 |
| -- | --: |
| raw 总行数 | 2007 |
| raw 总字符数 | 91099 |
| cleaned 总行数 | 1901 |
| cleaned 总字符数 | 89584 |
| 删除目录字符数 | 973 |
| 删除空白或格式字符数 | 307 |
| 识别制度数量 | 32 |
| 目录匹配数量 | 32 |
| 异常项数量 | 2 |

## 三、目录与正文对照

- 目录项数量：32
- 已识别正文制度数量：32
- 未匹配目录项：无

## 四、人工核验项

| 类型 | 制度 | 说明 |
| -- | -- | -- |
| title_mismatch | 杭州电子科技大学信息工程学院学生解除处分管理办法 | 目录规范标题为“杭州电子科技大学信息工程学院学生解除处分管理办法”，正文标题为“杭州电子科技大学信息工程学院学生校解除处分管理办法（试行）”。 |
| missing_document_number | 杭州电子科技大学信息工程学院学生公寓管理办法（暂行） | 目录元数据中未登记明确文号。 |

未识别文号制度：

- 杭州电子科技大学信息工程学院学生公寓管理办法（暂行）

## 五、验收结论

- 是否修改 raw 文件：否
- 是否生成 cleaned 文件：是
- 是否进入 Chunk：否
- 是否进入 Embedding：否
- 是否创建向量库：否
- 是否做 RAG：否
