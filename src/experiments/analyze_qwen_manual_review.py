"""Offline paired manual statistics, reusing the locked DeepSeek statistical methods."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import stats

from . import analyze_deepseek_manual_review as shared
from .blind_review_sheet import load_source, SCORE_FIELDS
from .manifests import atomic_write_csv, atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[3] / "experiment_outputs/hdu_ies/student_handbook/cross_llm_robustness/qwen"
DEEPSEEK = ROOT.parent / "deepseek/20261005T124451501115Z_deepseek_cross_model"
MODELS = ("qwen3.8-max", "deepseek-flash")
PREFIXES = ("qwen", "flash")
NAMES = {"qwen3.8-max": "Qwen3.8-Max", "deepseek-flash": "DeepSeek-Flash", "deepseek-v4-pro": "DeepSeek-V4-Pro"}
ORDINAL = shared.ORDINAL
CSV_NAME = "qwen_deepseek_manual_review.csv"
EXCLUDED = "HDU_IES_QA_023"
METHODS = {**shared.METHODS,
    "difference_direction": "Qwen - DeepSeek-Flash; positive ordinal delta favors Qwen; negative hallucination delta favors Qwen",
    "bootstrap": {**shared.METHODS["bootstrap"], "seed": 42, "iterations": 10000,
        "streams": "SeedSequence(42) children for two models x four metrics, reset separately per analysis set",
        "sampling": "descriptive bootstrap of each model mean; no paired-difference CI inferred"},
    "analysis_sets": "primary all40 includes actual partial answer; common39 removes only Q023 and reuses the same manual scores",
    "correlations": "Spearman, all 3 automatic x 3 ordinal manual metrics per model and pooled exploratory; undefined for constant inputs; pooled pairing not adjusted",
    "auto_quality_label_encoding": {"insufficient": 0, "needs_review": 1, "good": 2},
    "representative_cases": "automatic preference requires both overlap deltas >=0 and at least one >0; close means both absolute deltas <=0.05; manual preference requires nonnegative ordinal deltas and one positive; hallucination categories separate; overlap of preference/close categories allowed"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(path):
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_pairs(unblinded):
    grouped = defaultdict(dict)
    for row in unblinded:
        qid, model = row["question_id"], row["requested_model"]
        if model not in MODELS or model in grouped[qid]:
            raise shared.ReviewError("duplicate/unexpected real-model pair")
        grouped[qid][model] = row
    if len(unblinded) != 80 or len(grouped) != 40:
        raise shared.ReviewError("expected exactly 40 pairs")
    paired = []
    for qid, models in sorted(grouped.items()):
        if set(models) != set(MODELS):
            raise shared.ReviewError("missing real-model pair")
        qwen, flash = (models[m] for m in MODELS)
        for field in ("question", "evidence", "context_sha256", "prompt_sha256"):
            if qwen.get(field) != flash.get(field):
                raise shared.ReviewError("paired evidence/prompt mismatch")
        row = {"question_id": qid, "question": qwen["question"],
               "context_sha256": qwen["context_sha256"], "prompt_sha256": qwen["prompt_sha256"]}
        for field in SCORE_FIELDS:
            row.update({"qwen_" + field: qwen[field], "flash_" + field: flash[field],
                        "delta_" + field: qwen[field] - flash[field]})
        row["hallucination_discordance"] = ("qwen_only" if row["delta_hallucination"] == 1 else
            "flash_only" if row["delta_hallucination"] == -1 else "both" if qwen["hallucination"] else "neither")
        paired.append(row)
    return paired


def descriptive(pairs):
    summaries, distributions = [], []
    streams = np.random.SeedSequence(42).spawn(8)
    for model_index, (model, prefix) in enumerate(zip(MODELS, PREFIXES)):
        row = {"requested_model": model, "model": NAMES[model], "N": len(pairs)}
        for metric_index, metric in enumerate(SCORE_FIELDS):
            values = np.array([p[prefix + "_" + metric] for p in pairs])
            low, high = shared.bootstrap_mean_ci(values, seed=streams[model_index * 4 + metric_index])
            stat = {"N": len(values), "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                "median": float(np.median(values)), "Q1": float(np.quantile(values, .25)), "Q3": float(np.quantile(values, .75)),
                "ci_low": low, "ci_high": high}
            row.update({metric + "_" + k: v for k, v in stat.items()})
            counts = Counter(int(v) for v in values)
            for value in range(2 if metric == "hallucination" else 3):
                distributions.append({"requested_model": model, "metric": metric, "score": value,
                    "count": counts[value], "N": len(values), "percent": counts[value] / len(values) * 100})
        count = int(sum(p[prefix + "_hallucination"] for p in pairs))
        low, high = shared.wilson_interval(count, len(pairs))
        row.update(hallucination_count=count, hallucination_rate=count / len(pairs),
                   hallucination_wilson_ci_low=low, hallucination_wilson_ci_high=high)
        summaries.append(row)
    return summaries, distributions


def significance(pairs):
    tests = []
    for metric in ORDINAL:
        row = shared.signed_rank_test([p["delta_" + metric] for p in pairs])
        row.update(metric=metric, qwen_better_count=row.pop("flash_better"),
                   flash_better_count=row.pop("pro_better"), tie_count=row.pop("tie"),
                   direction="Qwen - DeepSeek-Flash")
        tests.append(row)
    for row, adjusted in zip(tests, shared.holm_adjust([r["raw_p"] for r in tests])):
        row["adjusted_p"] = adjusted
        row["significant_at_0_05"] = adjusted < .05 if adjusted is not None else None
    legacy_pairs = [{"flash_hallucination": p["qwen_hallucination"],
                     "pro_hallucination": p["flash_hallucination"]} for p in pairs]
    binary = shared.exact_mcnemar(legacy_pairs)
    qwen_only, flash_only = binary.pop("flash_only_count"), binary.pop("pro_only_count")
    binary.update(qwen_only_count=qwen_only, flash_only_count=flash_only,
                  direction="Qwen - DeepSeek-Flash; lower hallucination is better")
    return tests, binary


def join_auto(unblinded, automatic):
    lookup = {r["question_id"]: r for r in automatic}
    if len(automatic) != 40 or len(lookup) != 40 or set(lookup) != {r["question_id"] for r in unblinded}:
        raise shared.ReviewError("automatic/manual question IDs do not match")
    joined = []
    for row in unblinded:
        auto = lookup[row["question_id"]]
        prefix = "qwen" if row["requested_model"] == MODELS[0] else "flash"
        if (auto[prefix + "_answer"] != row["answer"] or any(auto[f] != row[f]
            for f in ("context_sha256", "prompt_sha256"))):
            raise shared.ReviewError("automatic/manual answer or hash mismatch")
        item = {**row, "keyword_overlap_ratio": float(auto[prefix + "_keyword_overlap"]),
                "evidence_overlap_ratio": float(auto[prefix + "_evidence_overlap"]),
                "auto_quality_label": auto[prefix + "_quality_label"],
                "generation_status": auto[prefix + "_generation_status"]}
        for metric in ("keyword_overlap_ratio", "evidence_overlap_ratio"):
            if not math.isfinite(item[metric]) or not 0 <= item[metric] <= 1:
                raise shared.ReviewError("invalid automatic metric")
        if item["auto_quality_label"] not in METHODS["auto_quality_label_encoding"]:
            raise shared.ReviewError("invalid automatic quality label")
        item["auto_quality_ordinal"] = METHODS["auto_quality_label_encoding"][item["auto_quality_label"]]
        joined.append(item)
    return joined


def correlations(joined):
    result = []
    for scope in (*MODELS, "combined"):
        rows = joined if scope == "combined" else [r for r in joined if r["requested_model"] == scope]
        for automatic in ("keyword_overlap_ratio", "evidence_overlap_ratio", "auto_quality_ordinal"):
            for manual in ORDINAL:
                x, y = [r[automatic] for r in rows], [r[manual] for r in rows]
                item = {"scope": scope, "automatic_metric": "auto_quality_label" if automatic == "auto_quality_ordinal" else automatic,
                    "manual_metric": manual, "N": len(rows), "spearman_rho": None, "p_value": None,
                    "status": "ok", "notes": "exploratory; no multiplicity correction"}
                if scope == "combined":
                    item["notes"] += "; paired-question clustering not adjusted"
                if len(set(y)) <= 1:
                    item.update(status="test_not_applicable", notes="rho undefined due to constant manual scores")
                elif len(set(x)) <= 1:
                    item.update(status="test_not_applicable", notes="rho undefined due to constant automatic values")
                else:
                    rho, p = stats.spearmanr(x, y)
                    item.update(spearman_rho=float(rho), p_value=float(p))
                result.append(item)
    return result


def representative_cases(paired, joined):
    categories = {name: [] for name in ("A_auto_qwen_manual_qwen", "B_auto_qwen_manual_tie",
        "C_auto_qwen_manual_flash", "D_auto_close_manual_qwen", "E_auto_close_manual_flash",
        "F_qwen_only_hallucination", "G_flash_only_hallucination", "H_both_hallucination")}
    lookup = {(r["question_id"], r["requested_model"]): r for r in joined}
    totals = Counter()
    for p in paired:
        q, f = (lookup[(p["question_id"], m)] for m in MODELS)
        automatic = [q[k] - f[k] for k in ("keyword_overlap_ratio", "evidence_overlap_ratio")]
        manual = [p["delta_" + k] for k in ORDINAL]
        auto_qwen = all(d >= 0 for d in automatic) and any(d > 0 for d in automatic)
        close = all(abs(d) <= .05 for d in automatic)
        man_qwen = all(d >= 0 for d in manual) and any(d > 0 for d in manual)
        man_flash = all(d <= 0 for d in manual) and any(d < 0 for d in manual)
        tied = all(d == 0 for d in manual)
        selected = []
        if auto_qwen:
            if man_qwen: selected.append("A_auto_qwen_manual_qwen")
            if tied: selected.append("B_auto_qwen_manual_tie")
            if man_flash: selected.append("C_auto_qwen_manual_flash")
        if close and man_qwen: selected.append("D_auto_close_manual_qwen")
        if close and man_flash: selected.append("E_auto_close_manual_flash")
        binary = p["hallucination_discordance"]
        if binary == "qwen_only": selected.append("F_qwen_only_hallucination")
        if binary == "flash_only": selected.append("G_flash_only_hallucination")
        if binary == "both": selected.append("H_both_hallucination")
        for name in selected:
            totals[name] += 1
            if len(categories[name]) < 3:
                categories[name].append({"question_id": p["question_id"], "ordinal_deltas": dict(zip(ORDINAL, manual)),
                    "keyword_delta": automatic[0], "evidence_delta": automatic[1], "hallucination_case": binary})
    return {"criteria": METHODS["representative_cases"], "total_counts": {k: totals[k] for k in categories}, "examples": categories}


def conclusions(primary, secondary):
    lines = []
    for metric in ORDINAL:
        main = next(r for r in primary["tests"] if r["metric"] == metric)
        common = next(r for r in secondary["tests"] if r["metric"] == metric)
        delta = main["mean_paired_difference"]
        if main["status"] == "test_not_applicable" and delta == 0:
            text = f"{metric}：40个配对差值均为0，人工评分没有观察到差异；Wilcoxon与rank-biserial不适用，p值和效应量保留为空。"
        else:
            direction = "Qwen较高" if delta > 0 else "Flash较高" if delta < 0 else "均值相同"
            significance_text = "Holm校正后显著" if main["significant_at_0_05"] else "Holm校正后不显著或不可检验"
            text = f"{metric}：Qwen-Flash平均差={delta:.6f}，{direction}；{significance_text}，校正p={shared.fmt_p(main['adjusted_p'])}，rank-biserial={main['effect_size']}。"
        same = np.sign(delta) == np.sign(common["mean_paired_difference"]) and main["significant_at_0_05"] == common["significant_at_0_05"]
        text += " 排除Q023后方向及显著性判断" + ("一致。" if same else "发生变化，见敏感性结果。")
        lines.append(text)
    binary = primary["mcnemar"]
    lines.append(f"Hallucination：Qwen-only={binary['qwen_only_count']}，Flash-only={binary['flash_only_count']}，both={binary['both_count']}，neither={binary['neither_count']}，exact McNemar p={binary['p_value']:.6f}；较低幻觉率更好。")
    return lines


def descriptive_md(summary):
    rows = []
    for r in summary:
        rows.append([r["model"], r["N"], *[f"{r[m+'_mean']:.4f} ± {r[m+'_std']:.4f} [{r[m+'_ci_low']:.4f}, {r[m+'_ci_high']:.4f}]" for m in ORDINAL],
            f"{r['hallucination_count']}/{r['N']} [{r['hallucination_wilson_ci_low']:.4f}, {r['hallucination_wilson_ci_high']:.4f}]"])
    return shared.md_table(["模型", "N", "Correctness均值±SD及95%CI", "Faithfulness", "Completeness", "Hallucination及Wilson CI"], rows)


def significance_md(tests):
    return shared.md_table(["维度", "均值差Qwen-Flash", "W", "raw p", "Holm p", "rank-biserial", "Qwen高/相同/Flash高", "状态"],
        [[r["metric"], f"{r['mean_paired_difference']:.6f}", r["statistic"], shared.fmt_p(r["raw_p"]),
          shared.fmt_p(r["adjusted_p"]), r["effect_size"], f"{r['qwen_better_count']}/{r['tie_count']}/{r['flash_better_count']}", r["status"]] for r in tests])


def latex_table(caption, label, headers, rows):
    def escape(value):
        text = str(value).replace("\\", r"\textbackslash{}")
        for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
            text = text.replace(a, b)
        return text
    return "\n".join(["% Fragment only; requires booktabs and a Chinese-capable thesis host.",
        r"\begin{table}[htbp]", r"\centering", r"\small", "\\caption{" + caption + "}", "\\label{" + label + "}",
        "\\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}", r"\toprule",
        " & ".join(escape(x) for x in headers) + r" \\", r"\midrule",
        *[" & ".join(escape(x) for x in row) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def analyze(directory):
    original, fields, rows, evidence_field = load_source(directory, CSV_NAME, allow_empty_answers=True)
    validation = shared.validate_scores(rows)
    if not validation["valid"]:
        first = validation["issues"][0]
        raise shared.ReviewError("manual validation failed: " + json.dumps(first, ensure_ascii=False))
    # Unblind only after complete post-import validation.
    mapping = shared.read_blind_mapping(directory / "qwen_deepseek_blind_key.json", models=MODELS)
    unblinded = shared.unblind(rows, mapping, evidence_field)
    for row in unblinded:
        row["provider"] = "tokenpony" if row["requested_model"] == MODELS[0] else "deepseek"
    paired = build_pairs(unblinded)
    _, automatic = shared.read_csv(directory / "cross_family_three_model_per_question.csv")
    joined = join_auto(unblinded, automatic)
    states = {(r["question_id"], r["requested_model"]): r["generation_status"] for r in joined}
    common_ids = sorted({p["question_id"] for p in paired if all(states[(p["question_id"], m)] == "completed" for m in MODELS)})
    if len(common_ids) != 39 or set(p["question_id"] for p in paired) - set(common_ids) != {EXCLUDED}:
        raise shared.ReviewError("common39 must exclude only Q023")
    # Verify score answers against raw responses as well as persisted automatic results.
    raw_lookup = {(r["question_id"], r["requested_model"]): r for r in [*jsonl(directory / "generation_records.jsonl"), *jsonl(DEEPSEEK / "generation_records.jsonl")]}
    evidence = jsonl(directory / "frozen_evidence_40.jsonl")
    refs = {e["question_id"]: e for e in evidence}
    manifest = json.loads((directory / "frozen_evidence_manifest.json").read_text(encoding="utf-8"))
    if len(refs) != 40 or sha(directory / "frozen_evidence_40.jsonl") != manifest["frozen_evidence_sha256"]:
        raise shared.ReviewError("Frozen Evidence mismatch")
    for r in unblinded:
        source = raw_lookup[(r["question_id"], r["requested_model"])]
        e = refs[r["question_id"]]
        if r["answer"] != source["answer"] or r["question"] != e["question"] or r["context"] != e["context"] or any(
            r[k] != e[k] or r[k] != source[k] for k in ("context_sha256", "prompt_sha256")):
            raise shared.ReviewError("manual/raw answer, context or prompt mismatch")
    before_qwen, before_deepseek = snapshot(directory), snapshot(DEEPSEEK)
    analyses = {}
    for name, selected in (("all40", paired), ("common39", [p for p in paired if p["question_id"] in common_ids])):
        summary, distributions = descriptive(selected)
        tests, mcnemar = significance(selected)
        analyses[name] = {"N": len(selected), "question_ids": [p["question_id"] for p in selected],
                          "summary": summary, "distributions": distributions, "tests": tests, "mcnemar": mcnemar}
    corr = correlations(joined)
    cases = representative_cases(paired, joined)
    old_fields, old_summary = shared.read_csv(DEEPSEEK / "deepseek_manual_evaluation_summary.csv")
    _, old_tests = shared.read_csv(DEEPSEEK / "deepseek_manual_significance_tests.csv")
    _, automatic_summary = shared.read_csv(directory / "cross_family_three_model_auto_summary.csv")
    generated = datetime.now(timezone.utc).isoformat()
    data = {"experiment_id": directory.name, "models": MODELS,
        "providers": {"qwen3.8-max": "tokenpony", "deepseek-flash": "deepseek"},
        "n_pairs_all40": 40, "n_pairs_common39": 39, "excluded_common39": [EXCLUDED],
        "generated_at": generated, "methods": METHODS, "source_hashes": before_qwen,
        "results": analyses, "correlations": corr, "representative_cases": cases,
        "previous_deepseek_descriptive": old_summary, "previous_deepseek_inference": old_tests,
        "independent_blind_review_batches": True, "llm_api_calls": 0}
    conclusion_lines = conclusions(analyses["all40"], analyses["common39"])
    main_tests = analyses["all40"]["tests"]
    positive = [r["metric"] for r in main_tests if r["mean_paired_difference"] > 0]
    significant = [r["metric"] for r in main_tests if r["significant_at_0_05"]]
    manual_support = ("人工均值较高的维度：" + ("、".join(positive) or "无") +
        "；Holm校正后显著的维度：" + ("、".join(significant) or "无") +
        "。自动优势在这些均值方向上得到描述性支持；未显著不表示模型等效，不能据此证明Qwen最好。")
    report = ["# Qwen vs DeepSeek-Flash人工评价", "", "Experiment: " + directory.name, "",
        "差值方向：Qwen - DeepSeek-Flash；正值表示ordinal评分Qwen更高，幻觉率较低更好。",
        "Bootstrap 10000次，seed=42；sample SD；Wilcoxon三维度分别做Holm校正；exact McNemar独立报告。", ""]
    for name in ("all40", "common39"):
        block = analyses[name]
        report += ["## " + name, "", descriptive_md(block["summary"]), "", significance_md(block["tests"]), "",
                   "McNemar: " + json.dumps(block["mcnemar"], ensure_ascii=False), ""]
    report += ["## 结论与敏感性", "", manual_support, "", *conclusion_lines, "", "## 自动评价与人工评价", "",
        shared.md_table(["范围", "自动指标", "人工指标", "rho", "p", "状态/说明"],
            [[r["scope"], r["automatic_metric"], r["manual_metric"], r["spearman_rho"], r["p_value"], r["notes"]] for r in corr]), "",
        "Combined是探索性80答案汇总，未调整同题配对聚类；相关分析未做多重比较校正，不据此进行模型优势推断。",
        "常量输入下rho未定义，保留null，不能解释为rho=0。", ""]
    evaluation_report = "\n".join(report)
    final = ["# 跨模型家族稳健性最终报告", "", "论文：《基于RAG技术的高校政策知识问答系统构建与检索策略优化研究》", "",
        "## 实验控制与生成可靠性", "",
        "DeepSeek-Flash、DeepSeek-V4-Pro与Qwen3.8-Max使用同一40题、Frozen Evidence和policy_grounded_qa_v1 Prompt，temperature=0、max_tokens=512。",
        "Qwen通过TokenPony第三方OpenAI-compatible网关调用，实际返回model=qwen3.8-max；不是阿里官方固定权重快照。",
        "Qwen attempted=40、completed=39、truncated=1（Q023）、API errors=0。Q023不重试、不补全，在all40分析中使用实际partial answer。", "",
        "## 三模型自动评价", "", shared.md_table(["模型", "Keyword overlap", "Evidence overlap", "good", "needs_review"],
            [[r["model"], r["keyword_overlap_ratio_all40"], r["evidence_overlap_ratio_all40"], r["good_all40"], r["needs_review_all40"]] for r in automatic_summary]), "",
        "规则自动指标排序为Qwen > Flash > V4-Pro；completed-only共同39题保持排序。规则指标不能替代人工正确性。", "",
        "## 人工评价批次", "",
        "人工结果来自两次独立盲评，不是一次三模型同时盲评。Flash在两个批次分别列示；不能伪造统一三模型人工显著性检验。", "",
        shared.md_table(["模型", "盲评批次", "N", "Correctness", "Faithfulness", "Completeness", "Hallucination rate"],
            [[NAMES[r["requested_model"]], "原DeepSeek双模型", r["N"], *[r[m+"_mean"] for m in ORDINAL], r["hallucination_rate"]] for r in old_summary] +
            [[r["model"], "本轮Qwen vs Flash", r["N"], *[r[m+"_mean"] for m in ORDINAL], r["hallucination_rate"]] for r in analyses["all40"]["summary"]]), "",
        "原Flash vs V4-Pro推断沿用锁定实验：ordinal全零配对差值，Wilcoxon不适用，不填造p值。",
        "本轮Qwen vs Flash推断仅使用本轮配对人工评分：", "", *conclusion_lines, "",
        "## 正式中文结论", "",
        "在相同40题、冻结证据、Prompt及512-token输出预算下，本研究完成DeepSeek同家族双模型和Qwen跨模型家族验证。Qwen有1/40生成截断，该实际partial answer纳入主分析。",
        *conclusion_lines, "",
        manual_support,
        "若人工评分无变异或存在天花板效应，自动指标与人工指标的相关性不可估计，不能把undefined写为零相关。",
        "", "## 自动与人工相关性", "",
        shared.md_table(["范围", "自动指标", "人工维度", "Spearman rho", "p", "说明"],
            [[r["scope"], r["automatic_metric"], r["manual_metric"], r["spearman_rho"], r["p_value"], r["notes"]] for r in corr]),
        "探索性相关分析未校正多重比较；combined未调整同题配对聚类，不作为模型优势的显著性证据。",
        "", "## 代表性差异", "", json.dumps(cases, ensure_ascii=False, indent=2), "",
        "TokenPony的版本路由、后端权重和计费/token口径无法独立审计；API model字段仅提供网关返回身份，provider差异不能全部归因于模型差异。", "",
        "## 对老师多模型验证建议的回应", "",
        "当前已具备三个API模型配置、两个模型家族的生成/自动评价证据及两次独立盲评，可响应增加多模型验证的建议。",
        "范围限定为当前40题和冻结证据，不等于三模型统一盲评或广泛泛化验证。既有Challenge设计仍未运行，评分区分度和样本规模限制需如实保留。", ""]
    outputs = {
        "qwen_deepseek_manual_review_unblinded.csv": unblinded,
        "qwen_deepseek_paired_manual_scores.csv": paired,
        "qwen_deepseek_auto_manual_joined.csv": joined,
        "qwen_deepseek_auto_manual_correlation.csv": corr,
        "qwen_deepseek_manual_evaluation_report.md": evaluation_report,
        "cross_family_robustness_final_report.md": "\n".join(final),
        "qwen_deepseek_manual_statistics.json": data,
        "qwen_deepseek_manual_representative_cases.json": cases}
    for name, block in analyses.items():
        outputs["qwen_deepseek_manual_summary_" + name + ".csv"] = block["summary"]
        outputs["qwen_deepseek_significance_" + name + ".csv"] = block["tests"]
        outputs["qwen_deepseek_hallucination_mcnemar_" + name + ".csv"] = [block["mcnemar"]]
        outputs["qwen_deepseek_score_distributions_" + name + ".csv"] = block["distributions"]
    ordinal_rows = [[name, r["model"], r["N"], *[f"{r[m+'_mean']:.3f}" for m in ORDINAL], f"{r['hallucination_rate']:.3f}"] for name,b in analyses.items() for r in b["summary"]]
    outputs["qwen_deepseek_manual_evaluation_table.tex"] = latex_table("Qwen与Flash人工评价均值及幻觉率", "tab:qwen-manual", ["分析", "模型", "N", "正确性", "忠实性", "完整性", "幻觉率"], ordinal_rows)
    outputs["qwen_deepseek_significance_table.tex"] = latex_table("Qwen减Flash配对统计", "tab:qwen-significance", ["分析", "维度", "均值差", "raw p", "Holm p", "效应量"],
        [[name, r["metric"], f"{r['mean_paired_difference']:.3f}", shared.fmt_p(r["raw_p"]), shared.fmt_p(r["adjusted_p"]), "NA" if r["effect_size"] is None else f"{r['effect_size']:.3f}"] for name,b in analyses.items() for r in b["tests"]])
    outputs["cross_family_three_model_summary_table.tex"] = latex_table("三模型自动指标与独立盲评人工结果", "tab:cross-family-three-model", ["模型", "批次", "Keyword", "Evidence", "正确性", "忠实性", "完整性"],
        [[NAMES[r["requested_model"]], "原盲评", next(a["keyword_overlap_ratio_all40"] for a in automatic_summary if a["requested_model"]==r["requested_model"]), next(a["evidence_overlap_ratio_all40"] for a in automatic_summary if a["requested_model"]==r["requested_model"]), *[r[m+"_mean"] for m in ORDINAL]] for r in old_summary] +
        [[r["model"], "本轮盲评", next(a["keyword_overlap_ratio_all40"] for a in automatic_summary if a["requested_model"]==r["requested_model"]), next(a["evidence_overlap_ratio_all40"] for a in automatic_summary if a["requested_model"]==r["requested_model"]), *[f"{r[m+'_mean']:.3f}" for m in ORDINAL]] for r in analyses["all40"]["summary"]])
    names = [*outputs, "qwen_deepseek_statistics_manifest.json"]
    if any((directory / name).exists() for name in names):
        raise shared.ReviewError("statistics outputs already exist; refusing overwrite")
    for name, value in outputs.items():
        target = directory / name
        if name.endswith(".csv"): atomic_write_csv(target, value)
        elif name.endswith(".json"): atomic_write_json(target, value)
        else: atomic_write_text(target, value)
    after_qwen, after_deepseek = snapshot(directory), snapshot(DEEPSEEK)
    unchanged = all(after_qwen.get(k) == v for k,v in before_qwen.items()) and before_deepseek == after_deepseek
    if not unchanged or (directory / CSV_NAME).read_bytes() != original:
        raise shared.ReviewError("source preservation failed")
    atomic_write_json(directory / "qwen_deepseek_statistics_manifest.json", {
        "experiment_id": directory.name, "models": MODELS, "providers": data["providers"], "generated_at": generated,
        "n_pairs_all40": 40, "n_pairs_common39": 39, "methods": METHODS,
        "source_hashes": before_qwen, "deepseek_source_hashes": before_deepseek,
        "output_hashes": {name: sha(directory / name) for name in outputs},
        "sources_unchanged": unchanged, "llm_api_calls": 0})
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in (".", ".."):
        raise shared.ReviewError("invalid experiment ID")
    directory = (ROOT / args.experiment_id).resolve()
    if directory.parent != ROOT.resolve():
        raise shared.ReviewError("experiment directory outside root")
    def offline(*args, **kwargs):
        raise RuntimeError("network/API prohibited during manual statistics")
    with patch.object(socket.socket, "connect", offline), patch.object(socket.socket, "connect_ex", offline), patch.object(socket, "create_connection", offline):
        data = analyze(directory)
    print(json.dumps({"all40": data["results"]["all40"], "common39": data["results"]["common39"],
        "llm_api_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
