"""Export thesis fragments from locked results, without generation or statistics reruns."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO.parent / "experiment_outputs/hdu_ies/student_handbook/cross_llm_robustness"
DS_ID = "20261005T124451501115Z_deepseek_cross_model"
QW_ID = "20261006T070200000000Z_qwen38max_tokenpony_cross_family"
TITLE = "基于RAG技术的高校政策知识问答系统构建与检索策略优化研究"
ORDINAL = ("correctness", "faithfulness", "completeness")
CHINESE = dict(zip(ORDINAL, ("正确性", "忠实性", "完整性")))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def escape(value):
    text = str(value)
    for old, new in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def latex_p(value):
    if value >= 0.001:
        return f"{value:.4g}"
    mantissa, exponent = f"{value:.2e}".split("e")
    return mantissa + r"\times10^{" + str(int(exponent)) + "}"


def table(caption, label, headers, rows, *, wide=False, note=""):
    spec = "l" + "r" * (len(headers)-1) if not wide else "l" + "r" * (len(headers)-2) + "X"
    begin = "\\begin{tabular}{"+spec+"}" if not wide else "\\begin{tabularx}{\\linewidth}{"+spec+"}"
    end = r"\end{tabular}" if not wide else r"\end{tabularx}"
    return "\n".join(["% Derived from locked experiment data; original outputs are read-only.",
        r"\begin{table}[htbp]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        "\\caption{"+caption+"}", "\\label{"+label+"}", begin, r"\toprule",
        " & ".join(headers)+r" \\", r"\midrule",
        *[" & ".join(str(v) for v in row)+r" \\" for row in rows],
        r"\bottomrule", end, (r"\par\smallskip\raggedright " + note) if note else "", r"\end{table}", ""])


def collect():
    ds, qw = BASE/"deepseek"/DS_ID, BASE/"qwen"/QW_ID
    manifest = load_json(qw/"qwen_deepseek_statistics_manifest.json")
    for name, expected in manifest["output_hashes"].items():
        if digest(qw/name) != expected:
            raise ValueError("locked statistics output changed: "+name)
    for root, mapping in ((qw, manifest["source_hashes"]), (ds, manifest["deepseek_source_hashes"])):
        for name, expected in mapping.items():
            if digest(root/name) != expected:
                raise ValueError("locked source changed: "+name)
    frozen = load_json(qw/"frozen_evidence_manifest.json")
    assert (qw/"frozen_evidence_40.jsonl").read_bytes() == (ds/"frozen_evidence_40.jsonl").read_bytes()
    assert digest(qw/"frozen_evidence_40.jsonl") == frozen["frozen_evidence_sha256"]
    for record in frozen["inputs"].values():
        assert digest(Path(record["path"])) == record["sha256"].lower()
    with (qw/"cross_family_three_model_auto_summary.csv").open(encoding="utf-8-sig", newline="") as f:
        auto = list(csv.DictReader(f))
    q = load_json(qw/"qwen_deepseek_manual_statistics.json")
    d = load_json(ds/"deepseek_manual_statistics.json")
    config = load_json(qw/"config.json")
    records = [json.loads(line) for line in (qw/"generation_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 40 and len({r["question_id"] for r in records}) == 40
    assert all(r["actual_model"] == r["requested_model"] == "qwen3.8-max" for r in records)
    assert config["request_options"] == {"enable_thinking": False}
    assert config["temperature"] == 0 and config["max_tokens"] == 512
    assert q["n_pairs_all40"] == 40 and q["n_pairs_common39"] == 39
    assert q["excluded_common39"] == ["HDU_IES_QA_023"]
    assert all(t["status"] == "test_not_applicable" and t["raw_p"] is None
        for t in d["statistical_tests"]["ordinal"])
    metrics_path = Path(frozen["inputs"]["retrieval_config"]["path"]).parent/"metrics.json"
    metrics = load_json(metrics_path)["metrics"]
    sources = [qw/"cross_family_three_model_auto_summary.csv", qw/"qwen_deepseek_manual_statistics.json",
        qw/"qwen_deepseek_statistics_manifest.json", qw/"cross_family_robustness_final_report.md",
        qw/"frozen_evidence_manifest.json", qw/"config.json", ds/"deepseek_manual_statistics.json",
        ds/"deepseek_manual_evaluation_report.md", metrics_path,
        *qw.glob("*table.tex"), *ds.glob("*table.tex")]
    return ds, qw, auto, q, d, frozen, metrics, sources


def build_assets(auto, q, d, frozen, metrics):
    primary, common = q["results"]["all40"], q["results"]["common39"]
    summaries = {r["requested_model"]: r for r in primary["summary"]}
    qwen, flash = summaries["qwen3.8-max"], summaries["deepseek-flash"]
    tests = {r["metric"]: r for r in primary["tests"]}
    corr = {(r["automatic_metric"],r["manual_metric"]): r for r in q["correlations"] if r["scope"] == "combined"}
    outputs = {}
    outputs["tables/automatic_all40.tex"] = table("不同生成模型下RAG回答自动评价结果", "tab:cm-auto",
        ["模型", "关键词重合率", "证据关键词重合率", "Good", "Needs Review", "完成率"],
        [[r["model"], f"{float(r['keyword_overlap_ratio_all40']):.6f}", f"{float(r['evidence_overlap_ratio_all40']):.6f}",
          r["good_all40"],r["needs_review_all40"], f"{float(r['completion_rate'])*100:.1f}\\%"] for r in auto],
        note="主分析包含全部40题及Q023实际部分答案；完成率按完整回答计。")
    outputs["tables/manual_all40.tex"] = table("Qwen3.8-Max与DeepSeek-Flash人工盲评结果", "tab:cm-manual",
        ["维度", "Qwen3.8-Max", "DeepSeek-Flash"],
        [[CHINESE[m], *[f"${r[m+'_mean']:.3f} \\pm {r[m+'_std']:.3f}$" for r in (qwen,flash)]] for m in ORDINAL] +
        [["幻觉发生", *[f"{r['hallucination_count']}/{r['N']}（{100*r['hallucination_rate']:.1f}\\%）" for r in (qwen,flash)]]],
        note="40个配对问题；前三项为均值与样本标准差，分值范围0--2；幻觉为0/1。")
    outputs["tables/significance_all40.tex"] = table("Qwen3.8-Max与DeepSeek-Flash配对统计检验结果", "tab:cm-tests",
        ["维度", r"\shortstack{平均\\差值}", "$W$", "Raw $p$", "Holm $p$", r"\shortstack{秩二列\\相关系数}", "结论"],
        [[CHINESE[m], f"{tests[m]['mean_paired_difference']:+.3f}", f"{tests[m]['statistic']:.1f}",
          f"{tests[m]['raw_p']:.6f}", f"{tests[m]['adjusted_p']:.6f}", f"{tests[m]['effect_size']:+.6f}",
          "优势趋势，未显著" if m == "faithfulness" else "未显著"] for m in ORDINAL], wide=True,
        note="差值为Qwen减Flash。双侧Wilcoxon省略零差值、并列绝对差值取平均秩，采用渐近p值；Holm校正三项检验，不使用显著性星号。")
    outputs["tables/manual_common39.tex"] = table("排除截断题后的人工评价敏感性分析", "tab:cm-common",
        ["模型", "N", "正确性", "忠实性", "完整性", "幻觉发生"],
        [[r["model"],r["N"],*[f"{r[m+'_mean']:.6f}" for m in ORDINAL],f"{r['hallucination_count']}/{r['N']}"] for r in common["summary"]],
        note=r"复用原人工评分，仅排除HDU\_IES\_QA\_023；三项W、原始p、Holm p和效应量均与主分析相同。")
    outputs["tables/automatic_common39.tex"] = table("共同完整回答集的自动评价敏感性分析", "tab:cm-auto-common",
        ["模型", "N", "关键词重合率", "证据关键词重合率", "Good", "Needs Review"],
        [[r["model"],r["completed_common_N"],f"{float(r['keyword_overlap_ratio_completed_common']):.6f}",
          f"{float(r['evidence_overlap_ratio_completed_common']):.6f}",r["good_completed_common"],r["needs_review_completed_common"]] for r in auto])
    outputs["tables/design.tex"] = table("跨生成模型稳健性实验设计", "tab:cm-design", ["环节", "控制与分析"],
        [["固定输入", "同一40题、冻结检索证据、上下文和提示模板"],
         ["生成配置", "DeepSeek-Flash、DeepSeek-V4-Pro、Qwen3.8-Max；输出预算512 tokens"],
         ["评价", "统一规则自动评价；两次独立匿名人工盲评"],
         ["统计", "配对检验、效应量、All40与Common39敏感性分析"]],wide=True)
    ci_sentences = []
    for m in ORDINAL:
        ci_sentences.append(CHINESE[m]+"均值的95\\%置信区间："+
            "，".join(name+f"为[{r[m+'_ci_low']:.3f}, {r[m+'_ci_high']:.3f}]" for name,r in (("Qwen",qwen),("Flash",flash)))+"。")
    h = primary["mcnemar"]
    hp = "；".join(name+f"幻觉率Wilson 95\\%区间为[{r['hallucination_wilson_ci_low']*100:.3f}\\%, {r['hallucination_wilson_ci_high']*100:.3f}\\%]" for name,r in (("Qwen",qwen),("Flash",flash)))+"。"
    faith = tests["faithfulness"]
    params = frozen["parameters"]
    section = r"""% Chapter fragment, not a standalone thesis. Requires booktabs and tabularx.
% Override CrossModelAssetRoot if the host thesis lives outside the repository.
\providecommand{\CrossModelAssetRoot}{docs/thesis}
\section{稳健性实验设计与评价指标}
\label{sec:cm-design}
本文面向杭州电子科技大学信息工程学院学生手册政策材料开展实验。为降低实验结论对单一生成模型的依赖，本文在固定检索证据和提示模板的条件下，引入DeepSeek系列不同生成配置及Qwen3.8-Max进行稳健性验证。
检索阶段比较BM25、真实语义Dense及Hybrid，并考察固定长度、条款和结构感知切分；Hash embedding只用于流程基线，不作为真实语义模型。下述验证固定已有检索配置，不再次执行检索或利用评分结果改变证据。
""" + f"% Retrieval source: {frozen['retrieval_source_experiment_id']}\n冻结证据来自已登记的加权Hybrid检索实验，采用\\texttt{{{escape(params['retrieval'])}}}与\\texttt{{{escape(params['chunk'])}}}，Top-k={params['top_k']}，Dense模型为\\texttt{{{escape(params['dense_model'])}}}。Dense与BM25权重分别为{params['dense_weight']:.1f}和{params['bm25_weight']:.1f}，候选池为{params['candidate_pool']}，按查询进行min--max归一化。该已登记配置在当前问题集的Recall@5={metrics['Recall@5']:.3f}、MRR@5={metrics['MRR@5']:.3f}，故选作本轮RAG检索输入；该结果不意味着对其他政策材料同样达到满召回。\n"
    section += r"""
三种生成配置均使用相同40题、按顺序冻结的证据片段、Context及system/user Prompt；逐题核验Context与Prompt的SHA-256。正式参数统一为temperature=0、max\_tokens=512，Prompt版本为\texttt{policy\_grounded\_qa\_v1}。配置名与API实际返回的模型标识分开记录，缺失或不匹配时停止。
\input{\CrossModelAssetRoot/tables/design.tex}

关键词重合率以参考答案关键词被生成答案覆盖的比例计算，证据关键词重合率以证据关键词被答案覆盖的比例计算；它们衡量词面覆盖，不能直接判断语义正确性。Good、Needs Review由既有规则及阈值决定，本轮未调整评价规则。人工正确性、忠实性和完整性采用0、1、2三级评分，幻觉采用0/1标记。
描述统计报告均值、样本标准差和95\%置信区间，均值区间采用10000次百分位bootstrap（seed=42），幻觉率采用Wilson区间。ordinal指标执行双侧Wilcoxon配对检验，对三个预定维度做Holm校正，并报告匹配配对秩二列相关效应量；幻觉独立使用exact McNemar检验。所有统计使用已保存的人工评分，不自动补分。主分析All40保留真实截断答案，Common39仅排除唯一截断题，复用相同评分。

\section{生成模型稳健性验证}
\label{sec:cm-results}
\subsection{DeepSeek同家族模型对照}
DeepSeek-Flash与DeepSeek-V4-Pro在相同40题、Frozen Evidence、Prompt和生成参数下分别生成回答，API返回模型标识不同。两模型在该独立盲评批次的正确性、忠实性和完整性均为$2.000\pm0.000$，幻觉均为0/40。
\input{\CrossModelAssetRoot/tables/deepseek_manual.tex}
三个等级指标的40个配对差值全部为0，故Wilcoxon及秩二列相关效应量记为不适用，p值保留为空。该结果反映人工量表的明显天花板效应，不是模型等效性证明。两个生成配置在当前高校政策问答任务上均表现出较高稳定性，但由于人工评分全部达到量表上限，无法进一步通过配对统计区分模型差异。

\subsection{Qwen跨模型家族对照}
Qwen3.8-Max通过TokenPony第三方OpenAI-compatible API调用，requested\_model与actual\_model均为\texttt{qwen3.8-max}。本实验不是DashScope官方调用，也不声称使用固定权重快照。正式请求设置enable\_thinking=false；从返回行为未观察到reasoning内容，记录thinking\_effective=false，缺失的reasoning\_tokens仍保留为空，该观察不构成后端内部计算过程的独立审计。
Qwen共尝试40题，完成39题、截断1题、API错误0次，完成率为97.5\%、截断率为2.5\%。HDU\_IES\_QA\_023达到512-token上限，实际部分答案保留于主分析；为维持统一输出预算，没有重试、补全或提高token上限。
\input{\CrossModelAssetRoot/tables/automatic_all40.tex}
在关键词重合率、证据关键词重合率及Good数量上，Qwen均高于Flash，Flash高于V4-Pro。排除Q023后共同完整回答集仍保持该排序，说明本次截断没有改变整体自动评价排序（见表\ref{tab:cm-auto-common}）。这只是既有规则下的结果，不代表所有语义评价维度均更优。
\input{\CrossModelAssetRoot/tables/automatic_common39.tex}

Qwen与Flash人工评分来自本轮独立盲评。旧Flash与V4-Pro盲评和本轮Qwen与Flash盲评不是一次统一三模型盲评；Flash在两个批次的评分不同，描述统计与推断均按批次分开，不能将三个模型组织为统一人工显著性排名。
\input{\CrossModelAssetRoot/tables/manual_all40.tex}
"""
    section += "\n".join(ci_sentences)+"\n"+hp+"\n"
    section += r"\input{\CrossModelAssetRoot/tables/significance_all40.tex}"+"\n"
    section += f"Qwen三个等级维度的人工均值均较高，但Holm校正后均未达到0.05显著性水平。忠实性的raw p={faith['raw_p']:.6f}，Holm p={faith['adjusted_p']:.6f}，秩二列相关系数为{faith['effect_size']:+.6f}，非零差值配对仅{faith['n_nonzero']}题。忠实性指标呈现较大的配对效应方向，但统计证据尚不足以在Holm校正后的0.05水平下拒绝零假设，尚不能确认其显著优于Flash。\n"
    section += f"当前样本中Qwen未记录明显幻觉，Flash有1题。exact McNemar的Qwen-only={h['qwen_only_count']}、Flash-only={h['flash_only_count']}、Both={h['both_count']}、Neither={h['neither_count']}，p={h['p_value']:.6f}，未显示显著差异，不能声称Qwen显著降低幻觉。\n"
    section += r"""
\input{\CrossModelAssetRoot/tables/manual_common39.tex}
排除Q023后，三个等级维度的W、原始p、Holm p和效应量均保持不变，主要方向和显著性结论不变。该题两模型的原人工评分均为2/2/2、幻觉为0，排除后减少一个零差值配对；评分保持原样，不因截断状态重新赋分。

\subsection{自动评价与人工评价一致性分析}
在本轮Qwen与Flash合并的80个答案中，探索性Spearman分析显示词面覆盖与人工评分存在一定关联。
"""
    for metric, cname in (("keyword_overlap_ratio","关键词重合率"),("evidence_overlap_ratio","证据关键词重合率")):
        section += cname+"与"+"、".join(CHINESE[m]+f"的相关系数为$\\rho={corr[(metric,m)]['spearman_rho']:.3f}$（$p={latex_p(corr[(metric,m)]['p_value'])}$）" for m in ORDINAL)+"。\n"
    section += r"""
合并样本中，证据关键词重合率与三个人工维度表现出比普通关键词重合率更强的关联；这提示证据覆盖可能更贴近政策回答质量，但不能据此证明指标更有效。并非所有单模型内部相关关系均达到未校正的0.05水平，尤其Qwen正确性和忠实性评分接近上限，区分度有限。
这些分析未做多重比较校正，Combined也未针对同一题的配对聚类结构进一步调整，所得p值仅作探索性描述。自动与人工评价方向总体一致，不能将相关性或规则标签当作人工语义评价的替代。

\section{实验讨论}
\label{sec:cm-discussion}
\subsection{跨模型鲁棒性}
本文从单一生成配置扩展到三个API模型配置、两个模型家族，包含DeepSeek-Flash与DeepSeek-V4-Pro同家族对照，以及Qwen3.8-Max与DeepSeek-Flash跨家族对照，较好回应了增加多模型验证的建议。三种生成配置均在固定证据下产生政策回答，其中Qwen有一次截断，不能表述为全部完整生成。当前结果支持检索证据与提示组织在这些生成配置下具有一定稳定性，但并未重新比较各模型下的全部检索策略组合，不能据此断言最优检索排序在任意模型或数据集上均保持不变。
\subsection{自动评价局限性}
规则评价易受关键词表达、答案长度及改写方式影响。更高的词面覆盖可以与更高人工均值同时出现，但本轮校正后的配对推断尚不足以确认显著模型优势。零差值检验不适用、未显著与模型等效是不同情形，应分别报告；观察到零幻觉也不表示总体幻觉风险为零。
\subsection{实验局限与有效性威胁}
问题集仅40题，来源集中于当前高校政策材料，结论的外部适用性有限。DeepSeek批次出现人工评分天花板效应，本轮Qwen评分也接近上限；单评审流程尚未报告评审者间一致性。两次独立盲评的评分尺度与批次差异不能排除，不能拼接为统一三模型推断。
Qwen经第三方网关调用，无法核验固定底层权重、路由及版本稳定性。不同provider的token统计口径可能不同，统一max\_tokens并不保证相同字符长度或内部计算预算。Qwen有一次512-token截断，主分析与敏感性分析都如实保留这一生成限制。
自动指标无法完全代替语义评价，相关分析未校正多重比较或合并样本的题内聚类。有效非零配对较少、量表离散且含并列秩，当前渐近Wilcoxon与效应量应结合样本和评分分布解释。结果仅支持当前冻结证据下的稳健性观察，不构成模型优劣的普遍证明。
"""
    outputs["cross_model_experiment_section.tex"] = section
    outputs["cross_model_conclusion.tex"] = r"""% Insert this paragraph into the existing conclusion; no new chapter number.
本文不仅比较了不同检索策略，还进一步考察了当前推荐RAG配置对生成模型变化的敏感性。在相同问题、检索证据与提示模板下，DeepSeek-Flash、DeepSeek-V4-Pro和Qwen3.8-Max三种生成配置均产生政策回答，Qwen有一次截断。Qwen的自动评价指标和本轮人工均值较高，但Holm校正后三个人工等级维度均未达到统计显著，幻觉配对检验也未显示显著差异；旧DeepSeek批次的人工满分主要体现量表天花板效应。排除截断题后主要结论不变。上述结果支持检索与证据组织在当前样本范围内具有一定跨模型稳定性，论文核心仍是高校政策RAG问答系统与检索策略优化。
"""
    return outputs


def export(output):
    ds,qw,auto,q,d,frozen,metrics,sources = collect()
    before = {str(p):digest(p) for p in sources}
    outputs = build_assets(auto,q,d,frozen,metrics)
    # Reuse the locked original DeepSeek tables without maintaining duplicate numbers.
    outputs["tables/deepseek_manual.tex"] = (ds/"deepseek_manual_evaluation_table.tex").read_text(encoding="utf-8")
    outputs["tables/deepseek_significance.tex"] = (ds/"deepseek_significance_table.tex").read_text(encoding="utf-8")
    targets = [output/name for name in outputs]
    if any(p.exists() for p in targets) or (output/"cross_model_sources.json").exists():
        raise ValueError("thesis assets already exist; refusing overwrite")
    output.mkdir(parents=True,exist_ok=True)
    for name,text in outputs.items():
        target=output/name
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("x",encoding="utf-8",newline="\n") as f: f.write(text)
    assert all(digest(Path(p)) == h for p,h in before.items())
    manifest={"title":TITLE,"experiment_ids":[DS_ID,QW_ID],"source_hashes":before,
        "output_hashes":{name:digest(output/name) for name in outputs},"llm_api_calls":0,
        "formal_thesis_main":None,"compilation":"chapter fragments only; no formal thesis main located",
        "original_results_modified":False,"statistics_rerun":False,"independent_blind_batches":True}
    (output/"cross_model_sources.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=REPO/"docs/thesis")
    args=parser.parse_args()
    print(json.dumps(export(args.output),ensure_ascii=False,indent=2))
