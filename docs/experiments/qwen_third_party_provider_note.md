# Qwen 第三方 API 调用的方法学说明

论文题目：《基于RAG技术的高校政策知识问答系统构建与检索策略优化研究》。

## Provider 与模型标识

本轮授权模型为 **Qwen3.8-Max**，通过第三方 OpenAI-compatible API 网关
`https://api.tokenpony.cn/v1` 调用。`provider=tokenpony`，
`base_url_type=third_party_openai_compatible`，`requested_model=qwen3.8-max`。
第三方账户仅绑定该模型是用户提供的账户条件；本项目没有独立核验网关后端权重。
每次响应的 `actual_model` 必须原样记录，不能以请求名称替代缺失的响应字段。

本实验不声称使用阿里云 DashScope 官方 endpoint、`qwen3.8-max-0902`
或固定模型权重快照。历史失败的 DashScope health-check attempt 保留且不覆盖。

## 环境变量与凭据

新接口优先读取 Windows 环境变量 `TOKENPONY_API_KEY` 和
`TOKENPONY_BASE_URL`。后者未配置时使用用户明确授权的上述 URL，
而非猜测地域；已配置为其他地址则停止。
若新 key 变量未配置，短期兼容读取 `DASHSCOPE_API_KEY`。
这个历史变量名下的凭据本轮作为第三方 provider 的凭据使用，
**不表示来自阿里云官方 DashScope**。正式元数据始终记录 `tokenpony`。
代码只保存变量名和 configured/missing，不保存密钥或 authorization header。

## 证据与生成控制

计划直接读取已锁定 DeepSeek 实验
`20261005T124451501115Z_deepseek_cross_model` 的
`frozen_evidence_40.jsonl` 和 `frozen_evidence_manifest.json`。
不重新 Retrieval。三个模型采用同一 40 questions、Frozen Evidence、
Question、Context、system/user prompt，逐题核验 context/prompt SHA-256。
正式参数为 `temperature=0.0`、`max_tokens=512`、
`prompt_version=policy_grounded_qa_v1`，检索设置维持原 manifest。
自动评价必须使用原 DeepSeek 评价程序及原 evaluation_config，不调整阈值。

## Thinking 与当前 Health Check

2026-10-06 首轮 Health 未发送 thinking 参数；随后依用户授权执行有限的
参数兼容检查。没有发送 tools、function calling 或 search 请求。

本次独立 attempt：`20261006T065148355806Z_qwen_tokenpony_cross_family`。
单次极短 Health Check 返回 HTTP 200，`actual_model=qwen3.8-max`，
模型身份 gate 通过。响应同时存在 reasoning_content，
`reasoning_tokens=32`，`thinking_effective=true`。
Health 使用 `max_tokens=32`；返回 `answer=""`、`finish_reason=length`，
全部输出额度用于 reasoning，回答成功 gate 未通过。
按任务停止规则没有再请求、没有运行 smoke 或正式问题。
该 32-token 配额属于 health 请求，不能冒充正式实验的 512-token 配额。
该 attempt 已锁定，后续未修改。

### 本轮实际参数兼容检查

独立 diagnostics：`20261006T065734707002Z_reasoning_compatibility`。
沿用此前已确认的模型身份，不另发身份探测请求。两次参数检查的
temperature=0、max_tokens=128，使用相同极短 system/user messages。

| 检查 | 顶层参数 | HTTP | actual_model | 最终 content | reasoning present | reasoning_tokens | completion_tokens | finish_reason |
|---|---|---:|---|---|---|---:|---:|---|
| A | reasoning_effort="none" | 200 | qwen3.8-max | OK | 是 | 20 | 23 | stop |
| B | enable_thinking=false | 200 | qwen3.8-max | OK | 否 | null | 1 | stop |

Test A 未能关闭 reasoning；Test B 没有 reasoning_content，单个 completion
token 已生成完整 OK。程序首次对缺失 reasoning_tokens 判定过严，额外发出
一次无关闭参数、max_tokens=512 的预算对照：HTTP 200、content=OK、
reasoning present=true、reasoning_tokens=19、completion_tokens=23、finish_reason=stop。
该请求完整保留，不被删除或算入正式 QA。开启/关闭参数的实际响应行为不同，
据此将 Test B 判为关闭控制通过；这属于 API 响应层面的判定，
不是对第三方后端内部计算过程的独立审计。

正式配置选择顶层 `enable_thinking=false`，不同时发送 reasoning_effort
或 thinking_budget。记录 `thinking_requested=false`、`thinking_effective=false`、
`thinking_control_method=enable_thinking_false`。reasoning_tokens 缺失时原样保存
null，绝不人为补成 0。若正式请求重新出现 reasoning 则停止。

正式独立 experiment：`20261006T070200000000Z_qwen38max_tokenpony_cross_family`。
保持 temperature=0、max_tokens=512、原 Prompt 和 Frozen Evidence。
前三题与 DeepSeek smoke 一致；已成功的 smoke 记录直接复用于正式40题。
执行时间以每条记录的 UTC timestamp 为准，experiment_id 是实验唯一标识。
generation_records.jsonl 使用逐次 append、flush/fsync；请求前保存 intent，
未确认结果的中断请求不自动重试，避免重复收费。

### 续跑前的历史停止结果

前三题 smoke 均通过，随后复用于正式实验。本轮向 Frozen Evidence 中
23个唯一问题各请求一次：22条 completed，1条 failed。
第23题 `HDU_IES_QA_023` 返回 model=qwen3.8-max、HTTP 200、
finish_reason=length、completion_tokens=512，无 reasoning_content，
reasoning_tokens=null。该回答被保留为截断失败，不列为成功。
余17题未请求；没有自动重试或增加 max_tokens。
这次失败发生在最终回答额度，并非原32-token Health 的 reasoning 耗尽。
由于40/40完整性未满足，不生成正式 Qwen 自动评价、三模型比较或人工盲评资产。
本轮共26次收费 API 请求：参数/预算 diagnostics 3次，QA生成23次；
其中3题 smoke 已计入这23次，没有重复请求。

## 自动评价与人工盲评

### 统一512-token预算与最终结果

本研究将不同模型正式回答的最大输出长度统一设为 max_tokens=512。
Qwen3.8-Max 在 HDU_IES_QA_023 上达到上限而发生 generation truncation。
为避免生成预算不一致，没有针对该模型或该问题提高 token budget，
也没有重新生成回答。该截断是统一约束下的真实观察结果，不是 API failure。

按用户后续决定，继续同一 experiment_id，仅请求未尝试的 Q024–Q040。
最终 attempted=40、completed=39、truncated=1、api_error=0、
completion_rate=97.5%、truncation_rate=2.5%。所有 actual_model=qwen3.8-max，
正式参数没有变化，没有重试或重新请求前23题。全部上下文/Prompt哈希与DeepSeek一致。

Q023 原始 JSONL 行仍保留历史 failed 标记及原 error 字段，整行字节不变。
新增 generation_status_overrides.json 以原始行 SHA-256 绑定该问题的
effective_status=truncated；generation_records_effective.jsonl 和评价输出使用该分类。
这个分类视图不是新增 attempt，不改变 partial answer、finish_reason、usage、latency
或任何其他原始响应。前23条原始文件前缀以及 Q023 原始行均有前后哈希验证。

主分析为 all-attempted analysis，N=40，保留 Q023 实际 partial answer。
敏感性分析为 completed-only sensitivity analysis，三模型使用同一39个 question IDs，
仅排除 Q023。40题主分析的 keyword/evidence overlap 及 good 数量均为
Qwen > Flash > V4-Pro；同一39题分析保持该排序。
排除截断没有改变自动指标相对排序，不能据此宣称人工正确性或忠实性显著优越。

新增80行盲评继续包含 Q023 的两个实际答案；共享 Evidence/Question、匿名 Model A/B，
评分字段为空。页面不显示模型身份、generation_status 或 finish_reason。
未来人工统计配置提供 all40 和 completed_common39 两套 question IDs，
复用同一份人工评分，不重新评分、不自动执行统计或解盲。

自动评价读取同一 evaluation_config，直接复用原 evaluate_answer_quality_record
及 summarize_quality。Keyword/evidence overlap 是规则辅助信号，不能替代
语义正确性或人工忠实性判断。三模型指标在40题完整性校验通过后才导出。

Qwen vs DeepSeek-Flash 增量盲评使用独立 seed=42 随机映射，
不读取或沿用旧 DeepSeek blind key。80行评分初始为空；CSV、单独保存的 key
和程序生成的 HTML 不改动旧 DeepSeek 人工评分。HTML 仅投影匿名问题/证据/答案，
使用不含模型身份的 review ID。评分 JSON 导回通过既有校验与备份流程。
统计配置只准备、不自动执行统计或解盲。

## 复现与归因局限

## 实验阶段收口（2026-10-06）

正式40题及人工盲评统计已完成，生成模型实验阶段关闭，后续只做离线核验和论文写作。
本项目不再安排额外生成模型、Q023重试或Challenge生成。前文关于空评分、
等待评分、23题停止的描述属于历史阶段，不代表最终实验状态。

本次Qwen与Flash独立盲评的正确性、忠实性和完整性均值分别为
1.975/1.850、1.975/1.725和1.850/1.675；幻觉计数为0/40和1/40。
三项Wilcoxon的Holm校正p分别为0.257956、0.059083和0.257956，
exact McNemar p=1。忠实性的原始p=0.019694不构成校正后的显著优势。
common39排除唯一截断Q023后，主要方向及显著性结论均不变。
描述统计与推断以正式实验目录的CSV/JSON为准。

旧Flash与V4-Pro盲评和本轮Qwen与Flash盲评是两次独立评分；
不得制造统一三模型人工检验，或把旧批次的满分解释为模型等效。

离线分析入口为`python -m src.experiments.analyze_qwen_manual_review --experiment-id ...`。
已存在统计输出时拒绝覆盖。当前正式输出已存在，不应为了论文写作再次运行分析入口。
生成、身份和reasoning脚本保留为方法追踪代码，本轮不得执行其真实网络入口。

以下局限在最终论文中保留：

API 的 model 字段只能证明网关返回的标识，不能独立验证后端模型权重、
路由或版本稳定性。滚动模型标识和第三方网关都是复现局限。
reasoning 有效状态与 DeepSeek 可能不同，provider 的计费/token 统计口径
也未确认一致。因此即使 Frozen Evidence 和 Prompt 相同，
也不能把 provider、reasoning 或路由差异全部归因于模型本身。
未完成的 Qwen 自动指标、人工评分、统计结果均不得补造。
