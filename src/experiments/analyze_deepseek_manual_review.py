"""Offline validation, unblinding and paired statistics for DeepSeek reviews.

No generation/provider imports or network calls. Original inputs are never written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import stats

from .manifests import atomic_write_csv, atomic_write_text


EXPERIMENT_ROOT = Path(__file__).resolve().parents[3] / "experiment_outputs/hdu_ies/student_handbook/cross_llm_robustness/deepseek"
MODELS = ("deepseek-flash", "deepseek-v4-pro")
LABELS = {"Model A", "Model B"}
ORDINAL = ("correctness", "faithfulness", "completeness")
SCORES = (*ORDINAL, "hallucination")
SEED = 42
BOOTSTRAP_ITERATIONS = 10000
INPUT_NAMES = ("deepseek_cross_model_manual_review.csv", "deepseek_cross_model_blind_key.json", "deepseek_cross_model_per_question.csv")
METHODS = {
    "difference_direction": "deepseek-flash - deepseek-v4-pro; positive ordinal difference favors Flash; lower hallucination is better",
    "bootstrap": {"method": "percentile bootstrap of each model's mean over the same 40 paired question IDs", "iterations": BOOTSTRAP_ITERATIONS,
                  "seed": SEED, "confidence_level": 0.95, "sampling": "independent descriptive resamples for each model and ordinal metric; no CI of paired difference is inferred"},
    "hallucination_ci": "Wilson score interval, two-sided 95%, no continuity correction",
    "wilcoxon": "two-sided signed-rank; zero_method=wilcox; zeros omitted; tied absolute differences receive average ranks; asymptotic tie-adjusted p; continuity correction=False",
    "all_zero_differences": "test_not_applicable; p-value null; rank-biserial null (no nonzero ranks)",
    "effect_size": "matched-pairs rank-biserial=(W_positive-W_negative)/(W_positive+W_negative), average ranks of nonzero absolute differences",
    "multiple_comparison": "Holm-Bonferroni over the prespecified three ordinal tests; unavailable p-values kept null and reserved as p=1 only for family-size correction",
    "mcnemar": "exact two-sided scipy.stats.binomtest(b, b+c, 0.5) for all discordance counts; statistic=min(b,c); if b+c=0, exact degenerate p=1",
    "correlations": "Spearman separately by model; pooled 80-answer correlations are exploratory, with paired question clustering not adjusted; constant inputs -> test_not_applicable",
    "std": "sample standard deviation (ddof=1)", "quartiles": "numpy.quantile method=linear",
    "error_analysis": "ordinal difference >=1 defines clearly higher; auto/manual reversal requires opposite nonzero keyword-overlap and correctness differences; at most 3 per category, ordered by difference magnitude then question_id",
}


class ReviewError(ValueError):
    """Unrecoverable source structure or pairing error; never silently repaired."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)) or any(not f for f in fields):
            raise ReviewError(f"invalid/duplicate CSV headers: {path.name}")
        rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ReviewError(f"malformed CSV row width: {path.name}")
    return fields, rows


def validate_structure(fields: list[str], rows: list[dict[str, str]], experiment_id: str, *,
                       allow_empty_answers: bool = False) -> str:
    required = {"question_id", "anonymous_model", "question", "answer", "review_note", *SCORES}
    missing = required - set(fields)
    if missing:
        raise ReviewError("missing required columns: " + ", ".join(sorted(missing)))
    evidence_field = "evidence" if "evidence" in fields else "context" if "context" in fields else None
    if evidence_field is None:
        raise ReviewError("manual CSV must contain evidence or context")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    keys = set()
    for row in rows:
        qid, label = row["question_id"], row["anonymous_model"]
        if not qid.strip() or label not in LABELS:
            raise ReviewError("empty question_id or unexpected anonymous_model")
        if (qid, label) in keys:
            raise ReviewError(f"duplicate question/model row: {qid}, {label}")
        keys.add((qid, label))
        if "experiment_id" in fields and row["experiment_id"] != experiment_id:
            raise ReviewError("experiment_id does not match the selected experiment")
        nonempty_fields = ("question", evidence_field) if allow_empty_answers else ("question", "answer", evidence_field)
        if any(not row[k].strip() for k in nonempty_fields):
            raise ReviewError(f"empty question/answer/evidence: {qid}")
        if "evidence" in fields and "context" in fields and row["evidence"] != row["context"]:
            raise ReviewError(f"evidence and context disagree: {qid}")
        if "context_sha256" in fields and row["context_sha256"] != hashlib.sha256(row[evidence_field].encode("utf-8")).hexdigest():
            raise ReviewError(f"context hash does not match evidence: {qid}")
        grouped[qid].append(row)
    if len(rows) != 80 or len(grouped) != 40:
        raise ReviewError(f"expected 80 rows / 40 questions, found {len(rows)} / {len(grouped)}")
    for qid, pair in grouped.items():
        if len(pair) != 2 or {r["anonymous_model"] for r in pair} != LABELS:
            raise ReviewError(f"missing/extra anonymous pair: {qid}")
        for field in ("question", evidence_field, "context_sha256", "prompt_sha256"):
            if field in fields and pair[0][field] != pair[1][field]:
                raise ReviewError(f"paired {field} differs: {qid}")
    return evidence_field


def parse_score(value: str, field: str) -> tuple[int | None, str | None]:
    text = value.strip()
    if not text or text.lower() in {"nan", "na", "n/a", "null", "none"}:
        return None, "missing"
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None, "invalid_string"
    if not number.is_finite():
        return None, "nonfinite"
    limit = 1 if field == "hallucination" else 2
    if number != number.to_integral_value() or not 0 <= number <= limit:
        return None, "out_of_range_or_noninteger"
    return int(number), None


def validate_scores(rows: list[dict[str, str]]) -> dict[str, Any]:
    issues = []
    for index, row in enumerate(rows, start=2):
        for field in SCORES:
            _number, issue = parse_score(row[field], field)
            if issue:
                issues.append({"csv_line": index, "question_id": row["question_id"], "anonymous_model": row["anonymous_model"],
                               "field": field, "value": row[field], "issue": issue})
    missing = sum(r["issue"] == "missing" for r in issues)
    return {"status": "manual review incomplete" if issues else "manual review complete", "valid": not issues,
            "rows": len(rows), "n_questions": len({r["question_id"] for r in rows}), "required_score_cells": len(rows) * 4,
            "missing_count": missing, "invalid_count": len(issues) - missing, "issue_count": len(issues),
            "affected_question_ids": sorted({r["question_id"] for r in issues}),
            "affected_anonymous_models": sorted({r["anonymous_model"] for r in issues}),
            "issues": issues, "blind_key_read": False}


def read_blind_mapping(path: Path, *, models: tuple[str, str] = MODELS) -> dict[str, str]:
    # Called only after every required score in all 80 rows validates.
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReviewError("duplicate JSON key in blind key")
            result[key] = value
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ReviewError("invalid blind-key JSON") from exc
    mapping = data.get("mapping") if isinstance(data, dict) else None
    if (not isinstance(mapping, dict) or set(mapping) != LABELS
            or not all(isinstance(v, str) for v in mapping.values())
            or set(mapping.values()) != set(models) or len(mapping) != 2):
        raise ReviewError("blind key must uniquely map Model A / Model B onto the two requested models")
    return mapping


def unblind(rows: list[dict[str, str]], mapping: dict[str, str], evidence_field: str) -> list[dict[str, Any]]:
    if not validate_scores(rows)["valid"]:
        raise ReviewError("manual review incomplete; unblinding forbidden")
    result = []
    for row in rows:
        result.append({**row, "requested_model": mapping[row["anonymous_model"]], "evidence": row[evidence_field],
                       **{k: parse_score(row[k], k)[0] for k in SCORES}})
    return result


def build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        qid, model = row["question_id"], row["requested_model"]
        if model not in MODELS or model in grouped[qid]:
            raise ReviewError(f"duplicate/unexpected real-model pair: {qid}")
        grouped[qid][model] = row
    if len(rows) != 80 or len(grouped) != 40:
        raise ReviewError("paired dataset must contain exactly 40 pairs")
    paired = []
    for qid in sorted(grouped):
        models = grouped[qid]
        if set(models) != set(MODELS):
            raise ReviewError(f"missing requested model: {qid}")
        flash, pro = (models[m] for m in MODELS)
        for field in ("question", "evidence", "context_sha256", "prompt_sha256"):
            if flash.get(field) != pro.get(field):
                raise ReviewError(f"paired {field} differs: {qid}")
        row = {"question_id": qid, "question": flash["question"]}
        for field in SCORES:
            row.update({f"flash_{field}": flash[field], f"pro_{field}": pro[field], f"delta_{field}": flash[field] - pro[field]})
        row["hallucination_discordance"] = "flash_only" if row["delta_hallucination"] == 1 else "pro_only" if row["delta_hallucination"] == -1 else "both" if flash["hallucination"] else "neither"
        paired.append(row)
    return paired


def bootstrap_mean_ci(values: list[int] | np.ndarray, seed: int = SEED, iterations: int = BOOTSTRAP_ITERATIONS) -> tuple[float, float]:
    if iterations < 10000:
        raise ReviewError("bootstrap requires at least 10000 iterations")
    values = np.asarray(values, dtype=float)
    samples = np.random.default_rng(seed).choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975), method="linear")
    return float(low), float(high)


def wilson_interval(count: int, n: int) -> tuple[float, float]:
    if not n or not 0 <= count <= n:
        raise ReviewError("invalid binary count/denominator")
    z = stats.norm.ppf(0.975)
    p = count / n
    divisor = 1 + z * z / n
    center = (p + z * z / (2 * n)) / divisor
    half = z / divisor * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, float(center - half)), min(1.0, float(center + half))


def descriptive(paired: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries, distributions = [], []
    # SeedSequence child streams keep independent model/metric bootstrap draws reproducible.
    seeds = np.random.SeedSequence(SEED).spawn(6)
    for model_index, (model, prefix) in enumerate(zip(MODELS, ("flash", "pro"))):
        row: dict[str, Any] = {"requested_model": model, "N": len(paired)}
        for metric_index, metric in enumerate(ORDINAL):
            values = np.array([p[f"{prefix}_{metric}"] for p in paired])
            low, high = bootstrap_mean_ci(values, seed=seeds[model_index * 3 + metric_index])
            stats_values = {"N": len(values), "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                "median": float(np.median(values)), "Q1": float(np.quantile(values, 0.25)), "Q3": float(np.quantile(values, 0.75)),
                "min": int(np.min(values)), "max": int(np.max(values)), "ci_low": low, "ci_high": high}
            row.update({f"{metric}_{k}": v for k, v in stats_values.items()})
        count = sum(p[f"{prefix}_hallucination"] for p in paired)
        low, high = wilson_interval(count, len(paired))
        row.update(hallucination_N=len(paired), hallucination_count=count, hallucination_rate=count / len(paired),
                   hallucination_ci_low=low, hallucination_ci_high=high)
        summaries.append(row)
        for metric in SCORES:
            values = Counter(p[f"{prefix}_{metric}"] for p in paired)
            for score in range(2 if metric == "hallucination" else 3):
                distributions.append({"requested_model": model, "metric": metric, "score": score,
                    "N": len(paired), "count": values[score], "percent": 100 * values[score] / len(paired)})
    return summaries, distributions


def signed_rank_test(differences: list[int]) -> dict[str, Any]:
    differences = np.asarray(differences, dtype=int)
    nonzero = differences[differences != 0]
    row = {"test": "Wilcoxon signed-rank", "n_pairs": len(differences), "n_nonzero": len(nonzero),
           "statistic": None, "raw_p": None, "adjusted_p": None, "significant_at_0_05": None,
           "effect_size_name": "matched_pairs_rank_biserial", "effect_size": None,
           "mean_paired_difference": float(np.mean(differences)), "flash_better": int(np.sum(differences > 0)),
           "tie": int(np.sum(differences == 0)), "pro_better": int(np.sum(differences < 0)),
           "method": METHODS["wilcoxon"], "status": "ok", "notes": ""}
    if not len(nonzero):
        row.update(status="test_not_applicable", notes="all paired differences are zero; no p-value or rank-biserial is assigned")
        return row
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive, negative = float(ranks[nonzero > 0].sum()), float(ranks[nonzero < 0].sum())
    row.update(effect_size=(positive - negative) / (positive + negative), positive_rank_sum=positive, negative_rank_sum=negative,
               tied_absolute_differences=len(np.unique(np.abs(nonzero))) < len(nonzero))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            test = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="asymptotic", correction=False)
        if not math.isfinite(test.pvalue) or not math.isfinite(test.statistic):
            raise ValueError("nonfinite Wilcoxon output")
        row.update(statistic=float(test.statistic), raw_p=float(test.pvalue), notes="; ".join(str(w.message) for w in caught))
    except ValueError as exc:
        row.update(status="test_not_applicable", notes=str(exc))
    return row


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    order = sorted(range(len(p_values)), key=lambda i: 1.0 if p_values[i] is None else p_values[i])
    corrected: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        p = p_values[index]
        running = max(running, min(1.0, (len(p_values) - rank) * (1.0 if p is None else p)))
        if p is not None:
            corrected[index] = running
    return corrected


def exact_mcnemar(paired: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter((p["flash_hallucination"], p["pro_hallucination"]) for p in paired)
    a, b, c, d = (counts[cell] for cell in ((0, 0), (0, 1), (1, 0), (1, 1)))
    discordant = b + c
    p = float(stats.binomtest(b, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {"metric": "hallucination", "test": "McNemar", "n_pairs": len(paired), "a": a, "b": b, "c": c, "d": d,
        "neither_count": a, "pro_only_count": b, "flash_only_count": c, "both_count": d,
        "discordant_pairs": discordant, "statistic": min(b, c), "p_value": p, "method": "exact_binomial_two_sided",
        "significant_at_0_05": p < 0.05, "mean_paired_difference": (c - b) / len(paired),
        "notes": "lower hallucination is better; independently reported, outside the ordinal Holm family" +
                 ("; zero discordance gives degenerate exact p=1" if not discordant else "")}


def join_automatic(unblinded: list[dict[str, Any]], fields: list[str], auto_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    if not {"question_id", "requested_model"} <= set(fields):
        raise ReviewError("automatic CSV lacks join keys")
    lookup = {}
    for row in auto_rows:
        key = (row["question_id"], row["requested_model"])
        if key in lookup or row["requested_model"] not in MODELS:
            raise ReviewError("duplicate/unexpected automatic question/model")
        lookup[key] = row
    expected = {(r["question_id"], r["requested_model"]) for r in unblinded}
    if set(lookup) != expected or len(auto_rows) != 80:
        raise ReviewError("automatic/manual join is not exactly one-to-one over 80 answers")
    aliases = {"keyword_overlap_ratio": ("keyword_overlap_ratio", "keyword_overlap"),
               "evidence_overlap_ratio": ("evidence_overlap_ratio", "evidence_keyword_overlap_ratio", "evidence_keyword_overlap"),
               "auto_quality_label": ("auto_quality_label",)}
    selected = {k: next((f for f in choices if f in fields), None) for k, choices in aliases.items()}
    skipped = [k for k, source in selected.items() if source is None]
    joined = []
    for manual in unblinded:
        automatic = lookup[(manual["question_id"], manual["requested_model"])]
        for field in ("experiment_id", "question", "answer", "context_sha256", "prompt_sha256"):
            if field in automatic and field in manual and automatic[field] != str(manual[field]):
                raise ReviewError(f"manual/automatic {field} mismatch: {manual['question_id']}")
        row = dict(manual)
        for k, source in selected.items():
            if source is not None:
                value = automatic[source]
                if k != "auto_quality_label":
                    try:
                        value = float(value)
                    except ValueError as exc:
                        raise ReviewError(f"invalid automatic metric: {source}") from exc
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ReviewError(f"invalid automatic metric range: {source}")
                row[k] = value
        if "auto_quality_label" in row:
            if row["auto_quality_label"] not in {"good", "needs_review", "insufficient"}:
                raise ReviewError("unexpected automatic quality label")
            row["auto_good"] = int(row["auto_quality_label"] == "good")
            row["auto_needs_review"] = int(row["auto_quality_label"] == "needs_review")
        if "prompt_sha256" in automatic:
            row["prompt_sha256"] = automatic["prompt_sha256"]
        joined.append(row)
    # Prompt hashes are also compared across the two automatic records.
    by_question = defaultdict(list)
    for r in joined:
        by_question[r["question_id"]].append(r)
    if "prompt_sha256" in fields and any(p[0]["prompt_sha256"] != p[1]["prompt_sha256"] for p in by_question.values()):
        raise ReviewError("automatic paired prompt hashes differ")
    return joined, skipped


def correlations(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations = (("keyword_overlap_ratio", "correctness"), ("evidence_overlap_ratio", "faithfulness"),
                 ("auto_good", "correctness"), ("auto_needs_review", "correctness"))
    result = []
    for scope in (*MODELS, "pooled_exploratory"):
        rows = joined if scope == "pooled_exploratory" else [r for r in joined if r["requested_model"] == scope]
        for automatic, manual in relations:
            item = {"scope": scope, "automatic_metric": automatic, "manual_metric": manual, "n": len(rows),
                    "spearman_rho": None, "p_value": None, "status": "ok", "notes": "exploratory; no multiplicity correction"}
            if scope == "pooled_exploratory":
                item["notes"] += "; 80 answers from 40 paired questions, clustering not adjusted"
            if not all(automatic in r for r in rows):
                item.update(status="skipped_missing_column", notes="automatic column absent")
            else:
                x, y = [r[automatic] for r in rows], [r[manual] for r in rows]
                if len(set(x)) == 1 or len(set(y)) == 1:
                    item.update(status="test_not_applicable", notes="constant input; Spearman undefined")
                else:
                    rho, p = stats.spearmanr(x, y)
                    item.update(spearman_rho=float(rho), p_value=float(p))
            result.append(item)
    return result


def error_examples(paired: list[dict[str, Any]], joined: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lookup = {(r["question_id"], r["requested_model"]): r for r in joined}
    categories = {k: [] for k in ("A_flash_higher", "B_pro_higher", "C_auto_flash_manual_pro", "D_auto_pro_manual_flash", "E_both_hallucination", "F_discordant_hallucination")}
    for p in paired:
        qid = p["question_id"]
        f, v = (lookup[(qid, m)] for m in MODELS)
        auto_delta = f.get("keyword_overlap_ratio", 0) - v.get("keyword_overlap_ratio", 0) if "keyword_overlap_ratio" in f else None
        item = {"question_id": qid, **{f"delta_{k}": p[f"delta_{k}"] for k in ORDINAL},
                "auto_keyword_delta": auto_delta, "flash_hallucination": p["flash_hallucination"], "pro_hallucination": p["pro_hallucination"]}
        if any(p[f"delta_{k}"] >= 1 for k in ORDINAL):
            categories["A_flash_higher"].append(item)
        if any(p[f"delta_{k}"] <= -1 for k in ORDINAL):
            categories["B_pro_higher"].append(item)
        if auto_delta is not None and auto_delta > 0 and p["delta_correctness"] < 0:
            categories["C_auto_flash_manual_pro"].append(item)
        if auto_delta is not None and auto_delta < 0 and p["delta_correctness"] > 0:
            categories["D_auto_pro_manual_flash"].append(item)
        if p["hallucination_discordance"] == "both":
            categories["E_both_hallucination"].append(item)
        if p["hallucination_discordance"] in {"flash_only", "pro_only"}:
            categories["F_discordant_hallucination"].append(item)
    for category, items in categories.items():
        def priority(r):
            deltas = [r[f"delta_{k}"] for k in ORDINAL]
            size = max(deltas) if category.startswith("A") else -min(deltas) if category.startswith("B") else abs(r["delta_correctness"])
            return (-size, r["question_id"])
        categories[category] = sorted(items, key=priority)[:3]
    return categories


def conclusions(tests: list[dict[str, Any]], mcnemar: dict[str, Any]) -> list[str]:
    result = []
    for test in tests:
        metric = test["metric"]
        if test["status"] != "ok":
            result.append(f"{metric}：检验不适用（{test['notes']}）；不赋予 p-value，不能据此证明模型等效。")
        elif test["adjusted_p"] >= 0.05:
            result.append(f"{metric}：在当前 40 题样本及人工评价标准下，未观察到统计显著差异（Holm p={fmt_p(test['adjusted_p'])}）。这不证明两个模型表现完全一致。")
        else:
            direction = "Flash" if test["effect_size"] > 0 else "V4-Pro"
            mean_direction = "Flash" if test["mean_paired_difference"] > 0 else "V4-Pro" if test["mean_paired_difference"] < 0 else "相同"
            result.append(f"{metric}：配对秩检验支持 {direction} 评分更高（Holm p={fmt_p(test['adjusted_p'])}，rank-biserial={test['effect_size']:.3f}）；均值方向={mean_direction}，Flash−Pro 均值差={test['mean_paired_difference']:.3f}。统计显著性不自动意味着实践重要性。")
    if mcnemar["p_value"] >= 0.05:
        result.append("幻觉：当前样本下未观察到统计显著的配对发生率差异；这不证明风险相同或不存在幻觉。")
    else:
        lower = "Flash" if mcnemar["b"] > mcnemar["c"] else "V4-Pro"
        result.append(f"幻觉：exact McNemar 检验支持 {lower} 发生率较低（p={fmt_p(mcnemar['p_value'])}）；未据此推断实践重要性或因果机制。")
    return result


def fmt_p(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2e}" if value < 0.0001 else f"{value:.4f}"


def md_table(fields: list[str], rows: list[list[Any]]) -> str:
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |",
                      *("| " + " | ".join(escape(v) for v in row) + " |" for row in rows)])


def render_report(data: dict[str, Any]) -> str:
    summaries, tests, mcnemar = data["descriptive_statistics"], data["statistical_tests"]["ordinal"], data["statistical_tests"]["hallucination"]
    lines = ["# DeepSeek 人工盲评统计报告", "", "experiment_id: " + data["experiment_id"], "",
        "## 1. 实验设计", "", "40 道相同问题由 deepseek-flash 和 deepseek-v4-pro 回答，使用相同 Frozen Evidence、Prompt 和 Context hash；按 question_id 构成 40 个配对样本。人工填写 Model A / Model B 盲评表后，程序校验全部 80 行评分并解盲。",
        "评分：correctness 0=核心结论错误、1=部分正确或轻微错误/遗漏、2=正确；faithfulness 0=明显脱离证据、1=部分有证据支持、2=关键内容均有证据支持；completeness 0=明显缺失关键内容、1=部分完整、2=基本完整；hallucination 0=无明显幻觉、1=存在无证据支持的事实性内容。",
        "三个等级指标越高越好，幻觉越低越好。所有差值均为 Flash−V4-Pro。API 返回标识用于模型区分，不构成底层权重审计。", "",
        "## 2. 描述统计", "", "等级评分为 mean ± sample std；均值 CI 使用 seed=42、10,000 次 percentile bootstrap，按每个模型独立描述计算。幻觉发生率使用 Wilson 95% CI。",
        md_table(["模型", "N", "正确性 mean ± std [95% CI]", "忠实性 mean ± std [95% CI]", "完整性 mean ± std [95% CI]", "幻觉 count/N, rate [95% CI]"],
            [[s["requested_model"], s["N"], *(f"{s[k+'_mean']:.3f} ± {s[k+'_std']:.3f} [{s[k+'_ci_low']:.3f}, {s[k+'_ci_high']:.3f}]" for k in ORDINAL),
              f"{s['hallucination_count']}/{s['N']}, {s['hallucination_rate']:.3f} [{s['hallucination_ci_low']:.3f}, {s['hallucination_ci_high']:.3f}]" ] for s in summaries]), "",
        "各等级 count 和百分比见 deepseek_manual_score_distributions.csv；中位数、四分位数及范围见汇总 CSV/JSON。", "",
        "## 3. 显著性检验", "", "Wilcoxon 双侧检验去除零差，绝对差 ties 使用平均秩，并使用 ties 修正的渐近 p-value。三个等级指标固定组成 Holm-Bonferroni 校正族；全零差记录 test_not_applicable，p-value 为空，仍保留三项检验的族大小。效应量为 matched-pairs rank-biserial，正值偏向 Flash。",
        md_table(["metric", "test", "raw p", "Holm p", "rank-biserial", "mean difference", "结论"],
            [[t["metric"], t["test"], fmt_p(t["raw_p"]), fmt_p(t["adjusted_p"]), "N/A" if t["effect_size"] is None else f"{t['effect_size']:.3f}", f"{t['mean_paired_difference']:.3f}",
              "检验不适用" if t["status"] != "ok" else "有统计显著差异" if t["significant_at_0_05"] else "当前样本下未观察到统计显著差异"] for t in tests]), "",
        "幻觉使用 exact McNemar 单独报告，不并入上述校正族。列为 Pro，行为 Flash：",
        md_table(["", "Pro=0", "Pro=1"], [["Flash=0", mcnemar["a"], mcnemar["b"]], ["Flash=1", mcnemar["c"], mcnemar["d"]]]),
        f"discordant pairs={mcnemar['discordant_pairs']}，b={mcnemar['b']}，c={mcnemar['c']}，statistic=min(b,c)={mcnemar['statistic']}，exact p={fmt_p(mcnemar['p_value'])}。", "",
        "## 4. 自动评价 vs 人工评价", "", "关联已有关键词重合率、证据关键词重合率及 good/needs_review 标签。相关性为探索性分析，不做因果推断。每模型单独计算，合并 80 回答的 p-value 未校正同题相关性。常量输入或缺列不赋予 rho/p。",
        md_table(["scope", "automatic", "manual", "N", "Spearman rho", "p", "status"],
            [[r["scope"], r["automatic_metric"], r["manual_metric"], r["n"], "N/A" if r["spearman_rho"] is None else f"{r['spearman_rho']:.3f}", fmt_p(r["p_value"]), r["status"]] for r in data["correlations"]]),
        "关键词指标不能直接代表语义正确性或忠实性。自动/人工方向不一致可定位进一步复核对象；是否因为语义等价表达被低估，需要逐题复核，本程序不自动认定机制。",
        "缺少的自动指标列：" + (", ".join(data["skipped_automatic_columns"]) or "无"), "",
        "## 5. Error Analysis", "", "每类最多列出 3 个题目索引；等级差至少 1 分作为明显更高的筛选标准。A/B 可重叠（不同指标方向不同）。C/D 比较关键词重合率和人工 correctness 的方向。这里只提供统计索引，不替代人工判断政策正确性。"]
    for category, examples in data["error_analysis"].items():
        lines.extend(["", "### " + category, "", md_table(["question_id", "Δcorrectness", "Δfaithfulness", "Δcompleteness", "Δkeyword", "Flash hallucination", "Pro hallucination"],
            [[e["question_id"], *(e[f"delta_{k}"] for k in ORDINAL), "N/A" if e["auto_keyword_delta"] is None else f"{e['auto_keyword_delta']:.3f}", e["flash_hallucination"], e["pro_hallucination"]] for e in examples]) if examples else "当前数据未命中该类别。"])
    lines.extend(["", "## 6. 论文可使用结论", "", *data["conclusions"], "",
        "结论仅适用于当前 40 题样本、冻结证据与评分标准。对等级数据计算均值仅用于描述，配对检验针对差值的秩分布；评分的有序性不自动证明等距。Wilcoxon 的位置解释依赖差值分布对称性，渐近 p-value 在非零差很少时需谨慎解读。未检验评审者间一致性，也不能外推到所有高校政策问题。",
        "输入哈希与分析方法详见 statistics_manifest.json。原始评分、blind key 和模型回答保持不变。", ""])
    return "\n".join(lines)


def render_latex(data: dict[str, Any]) -> tuple[str, str]:
    # Fragments only; the thesis supplies document class, Chinese support and booktabs.
    descriptive_lines = ["% Requires booktabs in the parent document. Generated from validated scores.",
        r"\begin{table}[htbp]", r"\centering", r"\caption{DeepSeek 双模型人工评价（95\% CI）}", r"\label{tab:deepseek-manual-evaluation}",
        r"\begin{tabular}{lccc}", r"\toprule", r"指标 & DeepSeek-Flash & DeepSeek-V4-Pro & CI 方法 \\", r"\midrule"]
    f, v = data["descriptive_statistics"]
    for metric, name in zip(ORDINAL, ("正确性", "忠实性", "完整性")):
        def cell(s):
            return f"${s[metric+'_mean']:.3f} \\pm {s[metric+'_std']:.3f}$ [{s[metric+'_ci_low']:.3f}, {s[metric+'_ci_high']:.3f}]"
        descriptive_lines.append(f"{name} & {cell(f)} & {cell(v)} & Bootstrap \\\\")
    def hallucination_cell(s):
        return f"{s['hallucination_rate']:.3f} [{s['hallucination_ci_low']:.3f}, {s['hallucination_ci_high']:.3f}]"
    descriptive_lines += [f"幻觉发生率 & {hallucination_cell(f)} & {hallucination_cell(v)} & Wilson \\\\",
        r"\bottomrule", r"\end{tabular}", f"\\par\\small $N={f['N']}$ pairs; ordinal scores: mean $\\pm$ sample std.", r"\end{table}", ""]
    significance_lines = ["% Requires booktabs in the parent document. N/A means test not applicable.",
        r"\begin{table}[htbp]", r"\centering", r"\caption{DeepSeek 双模型配对统计检验}", r"\label{tab:deepseek-manual-significance}",
        r"\begin{tabular}{llccc}", r"\toprule", r"指标 & 检验 & 原始 $p$ & Holm $p$ & Rank-biserial \\", r"\midrule"]
    for t, name in zip(data["statistical_tests"]["ordinal"], ("正确性", "忠实性", "完整性")):
        effect = "N/A" if t["effect_size"] is None else f"{t['effect_size']:.3f}"
        significance_lines.append(f"{name} & Wilcoxon & {fmt_p(t['raw_p'])} & {fmt_p(t['adjusted_p'])} & {effect} \\\\")
    significance_lines += [f"幻觉 & Exact McNemar & {fmt_p(data['statistical_tests']['hallucination']['p_value'])} & -- & -- \\\\",
        r"\bottomrule", r"\end{tabular}", r"\par\small Positive rank-biserial favors Flash; hallucination is reported separately.", r"\end{table}", ""]
    return "\n".join(descriptive_lines), "\n".join(significance_lines)


def guide_text(experiment_id: str) -> str:
    return f"""# DeepSeek 人工盲评说明

只填写 deepseek_cross_model_manual_review.csv 的评分与 review_note，不修改模型标签、问题、Context 或回答。

| 指标 | 0 | 1 | 2 |
|---|---|---|---|
| correctness | 核心结论错误 | 部分正确，有遗漏或轻微错误 | 正确 |
| faithfulness | 明显脱离证据 | 部分有证据支持 | 关键内容均有证据支持 |
| completeness | 明显缺失关键内容 | 部分完整 | 基本完整 |
| hallucination | 无明显幻觉 | 存在无证据支持的事实性内容 | 不允许 |

review_note 可空，四个评分必须完整，不能用空值表示 0。评分需为整数值；程序允许 1.0 等等值数字，但拒绝小数分数、NaN、无穷大和超范围值。

共 80 行、每行 4 分，需填写 320 个评分。评分过程中禁止打开 deepseek_cross_model_blind_key.json，直到所有 80 条评分全部完成。也不要打开包含真实模型标签的 per_question、summary、解盲表和统计报告。Evidence 只查看当前行 context/evidence；不要通过回答风格猜测模型身份。

在项目根目录运行（推荐使用已有 .venv）：

```powershell
.\\.venv\\Scripts\\python.exe -m src.experiments.analyze_deepseek_manual_review --experiment-id {experiment_id} --validate-only
```

校验模式不读取 blind key，缺分时正常退出并生成 manual_review_validation.csv 和 manual_review_validation.json。

全部评分完成后：

```powershell
.\\.venv\\Scripts\\python.exe -m src.experiments.analyze_deepseek_manual_review --experiment-id {experiment_id}
```

正式模式缺分退出码为 2，结构错误退出码为 1，只有评分和配对均通过才解盲并生成正式 CSV/Markdown/JSON/LaTeX 结果。输入文件永不写入。所有输出位于当前实验目录；statistics_manifest.json 记录输入哈希，评分改动后需重跑统计，以新 manifest 判断结果版本。

LaTeX 文件仅为 table/tabular 片段，父论文需提供 booktabs 与中文支持；用 \\input{{}} 插入。统计程序不调用 API、不下载依赖、不改论文、不登记假评分。
"""


def analyze(experiment_id: str, validate_only: bool = False) -> tuple[int, dict[str, Any]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", experiment_id):
        raise ReviewError("invalid experiment_id")
    root = EXPERIMENT_ROOT.resolve()
    directory = (root / experiment_id).resolve()
    if directory.parent != root or not directory.is_dir():
        raise ReviewError("selected experiment directory is missing or outside the permitted root")
    paths = {name: directory / name for name in INPUT_NAMES}
    for name, path in paths.items():
        if path.resolve().parent != directory or not path.is_file():
            raise ReviewError(f"required input missing or outside experiment: {name}")
    # Snapshot manual bytes before parsing, without opening the blind mapping.
    manual_hash = file_sha256(paths[INPUT_NAMES[0]])
    fields, rows = read_csv(paths[INPUT_NAMES[0]])
    evidence_field = validate_structure(fields, rows, experiment_id)
    validation = validate_scores(rows)
    validation.update(experiment_id=experiment_id, manual_csv_sha256=manual_hash,
        generated_at=datetime.now(timezone.utc).isoformat(), evidence_column=evidence_field)

    def write(name: str, content: str):
        target = directory / name
        if target.resolve().parent != directory or name in INPUT_NAMES:
            raise ReviewError("output would escape experiment directory or overwrite an input")
        atomic_write_text(target, content)

    def write_csv(name: str, data: list[dict[str, Any]], columns: list[str] | None = None):
        target = directory / name
        if target.resolve().parent != directory or name in INPUT_NAMES:
            raise ReviewError("unsafe output destination")
        # Existing helper expects a header union; supplied schema preserves an empty validation CSV.
        atomic_write_csv(target, data, fieldnames=columns)

    def assert_inputs_unchanged(hashes):
        if any(file_sha256(paths[name]) != digest for name, digest in hashes.items()):
            raise ReviewError("source changed during analysis; output not accepted, rerun after edits stop")

    write_csv("manual_review_validation.csv", validation["issues"], ["csv_line", "question_id", "anonymous_model", "field", "value", "issue"])
    write("manual_review_validation.json", json_text(validation))
    write("deepseek_manual_review_guide.md", guide_text(experiment_id))
    assert_inputs_unchanged({INPUT_NAMES[0]: manual_hash})
    if validate_only or not validation["valid"]:
        return (0 if validate_only else 2), {k: v for k, v in validation.items() if k != "issues"}

    source_hashes = {name: file_sha256(path) for name, path in paths.items()}
    if source_hashes[INPUT_NAMES[0]] != manual_hash:
        raise ReviewError("manual CSV changed after validation")
    mapping = read_blind_mapping(paths[INPUT_NAMES[1]])
    unblinded = unblind(rows, mapping, evidence_field)
    paired = build_pairs(unblinded)
    automatic_fields, automatic_rows = read_csv(paths[INPUT_NAMES[2]])
    joined, skipped = join_automatic(unblinded, automatic_fields, automatic_rows)
    # Keep available prompt hashes in the unblinded export, without leaking auto labels into blind inputs.
    prompts = {(r["question_id"], r["requested_model"]): r.get("prompt_sha256") for r in joined}
    for r in unblinded:
        if prompts[(r["question_id"], r["requested_model"])]:
            r["prompt_sha256"] = prompts[(r["question_id"], r["requested_model"])]
    summaries, distributions = descriptive(paired)
    tests = [{"metric": metric, **signed_rank_test([p[f"delta_{metric}"] for p in paired])} for metric in ORDINAL]
    for test, adjusted in zip(tests, holm_adjust([t["raw_p"] for t in tests])):
        test["adjusted_p"] = adjusted
        test["significant_at_0_05"] = adjusted < 0.05 if adjusted is not None else None
    mcnemar = exact_mcnemar(paired)
    correlations_result = correlations(joined)
    metadata = {"experiment_id": experiment_id, "models": list(MODELS), "n_questions": 40, "n_pairs": 40,
        "evaluation_schema": {k: {"allowed": list(range(2 if k == "hallucination" else 3)), "better": "lower" if k == "hallucination" else "higher"} for k in SCORES},
        "descriptive_statistics": summaries, "score_distributions": distributions,
        "confidence_intervals": {"methods": {k: METHODS[k] for k in ("bootstrap", "hallucination_ci")},
            "values": [{"requested_model": r["requested_model"], **{k: v for k, v in r.items() if "ci_" in k}} for r in summaries]},
        "statistical_tests": {"ordinal": tests, "hallucination": mcnemar},
        "effect_sizes": [{k: t[k] for k in ("metric", "effect_size_name", "effect_size", "mean_paired_difference", "flash_better", "tie", "pro_better")} for t in tests],
        "multiple_comparison_method": "Holm-Bonferroni, fixed family of 3 ordinal tests",
        "methods": METHODS, "correlations": correlations_result, "skipped_automatic_columns": skipped,
        "error_analysis": error_examples(paired, joined), "conclusions": conclusions(tests, mcnemar),
        "generated_at": datetime.now(timezone.utc).isoformat(), "source_files": {name: str(path) for name, path in paths.items()},
        "source_file_sha256": source_hashes, "analysis_script_sha256": file_sha256(Path(__file__)),
        "dependency_versions": {"numpy": np.__version__, "scipy": scipy.__version__}, "api_calls": 0,
        "original_inputs_modified": False, "status": "completed"}
    assert_inputs_unchanged(source_hashes)
    write_csv("deepseek_cross_model_manual_review_unblinded.csv", unblinded)
    write_csv("deepseek_cross_model_paired_manual_scores.csv", paired)
    write_csv("deepseek_manual_evaluation_summary.csv", summaries)
    write_csv("deepseek_manual_score_distributions.csv", distributions)
    write_csv("deepseek_manual_significance_tests.csv", tests)
    write_csv("deepseek_hallucination_mcnemar.csv", [mcnemar])
    write_csv("deepseek_auto_manual_joined.csv", joined)
    write_csv("deepseek_auto_manual_correlation.csv", correlations_result)
    write("deepseek_manual_evaluation_report.md", render_report(metadata))
    table, significance_table = render_latex(metadata)
    write("deepseek_manual_evaluation_table.tex", table)
    write("deepseek_significance_table.tex", significance_table)
    write("deepseek_manual_statistics.json", json_text(metadata))
    assert_inputs_unchanged(source_hashes)
    outputs = ["deepseek_cross_model_manual_review_unblinded.csv", "deepseek_cross_model_paired_manual_scores.csv",
        "deepseek_manual_evaluation_summary.csv", "deepseek_manual_score_distributions.csv", "deepseek_manual_significance_tests.csv",
        "deepseek_hallucination_mcnemar.csv", "deepseek_auto_manual_joined.csv", "deepseek_auto_manual_correlation.csv",
        "deepseek_manual_evaluation_report.md", "deepseek_manual_evaluation_table.tex", "deepseek_significance_table.tex", "deepseek_manual_statistics.json"]
    write("statistics_manifest.json", json_text({k: metadata[k] for k in ("experiment_id", "status", "generated_at", "source_files", "source_file_sha256", "analysis_script_sha256", "methods", "dependency_versions", "api_calls", "original_inputs_modified")} | {
        "output_file_sha256": {name: file_sha256(directory / name) for name in outputs}}))
    return 0, {"status": "completed", "experiment_id": experiment_id, "n_pairs": 40, "api_calls": 0, "output_directory": str(directory)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        code, report = analyze(args.experiment_id, args.validate_only)
    except (ReviewError, OSError) as exc:
        print(json_text({"status": "FAIL FAST", "error": str(exc), "api_calls": 0}), end="")
        return 1
    print(json_text(report), end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
