# 离线并排盲评

生成器只读取匿名 `deepseek_cross_model_manual_review.csv`，不读取身份映射文件、自动评价或生成记录。HTML 只显示 Question、共享 Evidence、Model A / Model B 的 Answer、四个评分和备注；额外身份列不会被嵌入。原 CSV 在生成和浏览时不改动。

## 使用页面

双击当前实验目录的 `blind_review_side_by_side.html`，无需服务器或网络。左侧导航包含 40 题，每题桌面端两列并排，窄屏上下排列。每个模型独立填写 correctness、faithfulness、completeness（0/1/2）、hallucination（0/1）和 review_note。空值不等于 0，备注可空。

本机浏览器会尝试自动保存草稿；文件页面的 localStorage 支持可能因浏览器设置不同而变化，请定期“下载草稿”。“载入草稿”检查实验 ID、原 CSV SHA-256、80 个匿名题目键及评分范围后才恢复。“下载完整评分”只有全部 320 个评分有效时才能下载 `blind_review_scores.json`。浏览器不会直接写 CSV，完整评分 JSON 仅包含匿名 ID、评分和来源 hash。

## 生成 / 重新生成

```powershell
Set-Location 'F:\GraduationThesis\rag-campus-policy'
.\.venv\Scripts\python.exe -m src.experiments.blind_review_sheet build --experiment-id 20261005T124451501115Z_deepseek_cross_model
```

## 人工完成后导回

```powershell
Set-Location 'F:\GraduationThesis\rag-campus-policy'
.\.venv\Scripts\python.exe -m src.experiments.blind_review_sheet import --experiment-id 20261005T124451501115Z_deepseek_cross_model --scores 'C:\Users\ASUS\Downloads\blind_review_scores.json'
```

将 `--scores` 路径替换为实际下载位置。导回不解盲、不调用 API，只更新四个评分及 review_note；其他列、原问题顺序、匿名标签和完整回答保持不变。

导回先验证当前 CSV 的 80 行 / 40 题 / 每题 A+B、评分 JSON 的精确匿名键集合、所有评分范围，以及来源 SHA-256。任何异常 FAIL FAST，不删除行、不填补缺失，不创建评分后结果。

通过后先在当前实验目录 `manual_review_backups/<UTC时间>_<随机后缀>/` 保存原 CSV 的逐字节 `.bak` 备份和 `backup_manifest.json`（原 SHA-256、备份 SHA-256、导入 JSON SHA-256、预计新 CSV SHA-256）。确认备份可读且原 CSV 未变化后，使用同目录临时文件原子替换 CSV；随后保存 `import_receipt.json`。备份不覆盖旧备份。失败时原 CSV 保持原样，已有备份保留用于审计。

过期 HTML / JSON 的源 hash 与当前 CSV 不一致时会停止。已有非空评分/备注若发生冲突，默认也停止；只有确实要修订评分才增加 `--replace-existing-scores`，同样保留备份。重复或并发导入受锁保护；若进程崩溃留下 `.blind_review_import.lock`，请先核对 CSV/备份状态和确认无导入进程，之后处理锁，不自动删除他人的锁。

导回后旧 HTML 仍对应导回前的 CSV。需要继续修改时重新 build；然后只使用新页面及对应来源 hash 的导出文件。已有人工统计需要在评分导回后重新运行，不能继续引用旧输入 hash 的统计结果。

## 验证

pytest 使用独立合成 fixture，覆盖原 CSV 不变、导入只更新评分、80/40 及匿名模型校验、范围/空值/NaN、过期来源、先备份再写入、备份失败、并发编辑、已有评分冲突、路径边界及身份文件拒绝。网络和身份文件读取在测试中均被禁止。

浏览器验收使用本机 Chromium，以 `file://` 页面、offline 模式验证桌面并排、手机无横向溢出、题目切换、自动草稿恢复、不完整评分下载阻止、草稿载入及 80 行完整 JSON 下载。填写与导回流程仅在 synthetic fixture 上测试，真实 CSV 仍为空评分。
