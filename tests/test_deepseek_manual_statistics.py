"""Synthetic review fixtures only; never fill the real experiment's scores."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import socket
from pathlib import Path
from urllib import request

import numpy as np
import pytest

from src.experiments import analyze_deepseek_manual_review as analysis


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline statistics must not call any network/API")
    monkeypatch.setattr(request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def fixture_rows(experiment_id="synthetic_review"):
    rows = []
    for index in range(40):
        for label_index, label in enumerate(("Model A", "Model B")):
            context = f"synthetic context {index}"
            rows.append({"experiment_id": experiment_id, "question_id": f"SYNTHETIC_{index:03d}",
                "anonymous_model": label, "question": f"synthetic question {index}", "context": context,
                "context_sha256": hashlib.sha256(context.encode()).hexdigest(), "answer": f"synthetic answer {index} {label_index}",
                "correctness": str((index + label_index) % 3), "faithfulness": str((index * 2 + label_index) % 3),
                "completeness": str((index + 2 * label_index) % 3), "hallucination": str(int(index % (5 + label_index) == 0)),
                "review_note": ""})
    return rows


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path / "experiments"
    directory = root / "synthetic_review"
    directory.mkdir(parents=True)
    monkeypatch.setattr(analysis, "EXPERIMENT_ROOT", root)
    rows = fixture_rows()
    mapping = {"Model A": analysis.MODELS[0], "Model B": analysis.MODELS[1]}
    save_csv(directory / analysis.INPUT_NAMES[0], rows)
    (directory / analysis.INPUT_NAMES[1]).write_text(json.dumps({"seed": 42, "mapping": mapping}), encoding="utf-8")
    automatic = []
    for row in rows:
        automatic.append({"experiment_id": directory.name, "question_id": row["question_id"],
            "requested_model": mapping[row["anonymous_model"]], "answer": row["answer"],
            "context_sha256": row["context_sha256"], "prompt_sha256": "same-prompt-hash-for-the-pair",
            "keyword_overlap_ratio": str(int(row["correctness"]) / 2),
            "evidence_overlap_ratio": str(int(row["faithfulness"]) / 2),
            "auto_quality_label": "good" if row["correctness"] == "2" else "needs_review"})
    save_csv(directory / analysis.INPUT_NAMES[2], automatic)
    return directory, rows, mapping, automatic


def test_80_valid_rows_pass(experiment):
    directory, rows, _mapping, _auto = experiment
    evidence_field = analysis.validate_structure(list(rows[0]), rows, directory.name)
    assert evidence_field == "context"
    assert analysis.validate_scores(rows)["valid"]


@pytest.mark.parametrize("field,value,expected", [("correctness", "", "missing"), ("faithfulness", "NaN", "missing"),
    ("completeness", "abc", "invalid_string"), ("correctness", "3", "out_of_range_or_noninteger"),
    ("hallucination", "2", "out_of_range_or_noninteger"), ("correctness", "1.5", "out_of_range_or_noninteger"),
    ("correctness", "Infinity", "nonfinite")])
def test_invalid_scores_detected(field, value, expected):
    rows = fixture_rows()
    rows[0][field] = value
    result = analysis.validate_scores(rows)
    assert not result["valid"]
    assert result["issue_count"] == 1
    assert result["issues"][0]["issue"] == expected
    assert result["issues"][0]["question_id"] == rows[0]["question_id"]


def test_empty_scores_320_and_no_blind_read(experiment, monkeypatch):
    directory, rows, _mapping, _auto = experiment
    for row in rows:
        for score in analysis.SCORES:
            row[score] = ""
    save_csv(directory / analysis.INPUT_NAMES[0], rows)
    before = (directory / analysis.INPUT_NAMES[0]).read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("blind key must not be decoded before scoring completes")
    monkeypatch.setattr(analysis, "read_blind_mapping", forbidden)
    code, result = analysis.analyze(directory.name, validate_only=True)
    assert code == 0
    assert result["missing_count"] == 320
    assert result["status"] == "manual review incomplete"
    assert result["blind_key_read"] is False
    assert analysis.analyze(directory.name)[0] == 2
    assert (directory / analysis.INPUT_NAMES[0]).read_bytes() == before
    assert not (directory / "deepseek_manual_evaluation_summary.csv").exists()
    assert not list(directory.glob("*.tex"))
    assert not (directory / "statistics_manifest.json").exists()
    assert not (directory / "deepseek_cross_model_manual_review_unblinded.csv").exists()
    fields, validation = analysis.read_csv(directory / "manual_review_validation.csv")
    assert len(validation) == 320
    assert "value" in fields


def test_complete_validate_only_stays_blind(experiment, monkeypatch):
    directory, _rows, _mapping, _auto = experiment
    def forbidden(*args):
        raise AssertionError("validate-only never decodes the key")
    monkeypatch.setattr(analysis, "read_blind_mapping", forbidden)
    assert analysis.analyze(directory.name, True)[0] == 0
    assert not (directory / "deepseek_manual_statistics.json").exists()


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra_model", "question", "context"])
def test_structural_anomalies_fail_fast(mutation):
    rows = fixture_rows()
    if mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "missing":
        rows.pop()
    elif mutation == "extra_model":
        rows[0]["anonymous_model"] = "Model C"
    elif mutation == "question":
        rows[1]["question"] = "different question"
    else:
        rows[1]["context"] = "modified evidence"
    with pytest.raises(analysis.ReviewError):
        analysis.validate_structure(list(rows[0]), rows, "synthetic_review")


def test_unblinding_mapping_and_delta(experiment):
    directory, rows, mapping, _auto = experiment
    assert analysis.read_blind_mapping(directory / analysis.INPUT_NAMES[1]) == mapping
    unblinded = analysis.unblind(rows, mapping, "context")
    assert unblinded[0]["requested_model"] == analysis.MODELS[0]
    assert unblinded[1]["requested_model"] == analysis.MODELS[1]
    pairs = analysis.build_pairs(unblinded)
    assert len(pairs) == 40
    assert pairs[0]["delta_correctness"] == -1
    assert pairs[0]["delta_faithfulness"] == -1
    assert pairs[0]["delta_completeness"] == -2
    assert pairs[0]["flash_correctness"] == 0
    assert pairs[0]["pro_correctness"] == 1
    assert all(p["delta_hallucination"] == p["flash_hallucination"] - p["pro_hallucination"] for p in pairs)


@pytest.mark.parametrize("mapping", [{"Model A": "deepseek-flash", "Model B": "deepseek-flash"},
    {"Model A": "deepseek-flash"}, {"Model A": "other", "Model B": "deepseek-v4-pro"}])
def test_invalid_blind_mapping_fails(experiment, mapping):
    directory, _rows, _mapping, _auto = experiment
    key = directory / analysis.INPUT_NAMES[1]
    key.write_text(json.dumps({"mapping": mapping}))
    with pytest.raises(analysis.ReviewError):
        analysis.read_blind_mapping(key)


def test_duplicate_json_key_rejected(experiment):
    directory, _rows, _mapping, _auto = experiment
    key = directory / analysis.INPUT_NAMES[1]
    key.write_text('{"mapping":{"Model A":"deepseek-flash","Model A":"deepseek-v4-pro","Model B":"deepseek-flash"}}')
    with pytest.raises(analysis.ReviewError, match="duplicate"):
        analysis.read_blind_mapping(key)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "question", "extra_model"])
def test_strict_real_model_pairs(mutation):
    rows = analysis.unblind(fixture_rows(), {"Model A": analysis.MODELS[0], "Model B": analysis.MODELS[1]}, "context")
    if mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "missing":
        rows.pop()
    elif mutation == "question":
        rows[1]["question"] = "mismatch"
    else:
        rows[1]["requested_model"] = "unexpected-model"
    with pytest.raises(analysis.ReviewError):
        analysis.build_pairs(rows)


def test_bootstrap_reproducible_and_constant_ci():
    values = np.array([0, 1, 2] * 13 + [1])
    assert analysis.bootstrap_mean_ci(values) == analysis.bootstrap_mean_ci(values)
    assert analysis.bootstrap_mean_ci([2] * 40) == (2, 2)
    with pytest.raises(analysis.ReviewError):
        analysis.bootstrap_mean_ci(values, iterations=9999)


def test_wilson_independent_reference():
    # Hand-calculated 95% Wilson interval for 0/40: upper=z^2/(n+z^2).
    low, high = analysis.wilson_interval(0, 40)
    z = 1.959963984540054
    assert low == pytest.approx(0, abs=1e-15)
    assert high == pytest.approx(z * z / (40 + z * z))
    low, high = analysis.wilson_interval(40, 40)
    assert high == pytest.approx(1)
    assert low == pytest.approx(40 / (40 + z * z))


def test_wilcoxon_ties_effect_and_p_independent_reference():
    result = analysis.signed_rank_test([2, 1, 1, -1, 0])
    # Nonzero absolute ranks: 4,2,2,2; W+=8,W-=2; variance=sum(rank^2)/4=7.
    assert result["status"] == "ok"
    assert result["effect_size"] == pytest.approx(0.6)
    assert result["statistic"] == 2
    assert result["raw_p"] == pytest.approx(math.erfc(3 / math.sqrt(14)))
    assert result["mean_paired_difference"] == pytest.approx(0.6)
    assert result["flash_better"] == 3 and result["tie"] == 1 and result["pro_better"] == 1


def test_all_ties_not_applicable():
    result = analysis.signed_rank_test([0] * 40)
    assert result["status"] == "test_not_applicable"
    assert result["raw_p"] is None
    assert result["effect_size"] is None
    assert result["tie"] == 40


def test_exact_mcnemar_independent_reference():
    paired = [{"flash_hallucination": 0, "pro_hallucination": 1}] + [{"flash_hallucination": 1, "pro_hallucination": 0}] * 9
    result = analysis.exact_mcnemar(paired)
    assert result["b"] == 1 and result["c"] == 9
    assert result["p_value"] == pytest.approx(22 / 1024)
    assert result["statistic"] == 1
    assert result["method"] == "exact_binomial_two_sided"
    no_discordance = analysis.exact_mcnemar([{"flash_hallucination": 0, "pro_hallucination": 0}] * 40)
    assert no_discordance["p_value"] == 1
    assert no_discordance["discordant_pairs"] == 0


def test_holm_reference_and_missing_values():
    assert analysis.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    values = analysis.holm_adjust([None, 0.01, 0.03])
    assert values[0] is None
    assert values[1:] == pytest.approx([0.03, 0.06])
    assert analysis.holm_adjust([None, None, None]) == [None] * 3


def test_source_sha_deterministic(tmp_path):
    path = tmp_path / "test.txt"
    path.write_bytes(b"abc")
    assert analysis.file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert analysis.file_sha256(path) == analysis.file_sha256(path)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "changed_answer", "changed_context_hash", "changed_prompt_hash"])
def test_automatic_join_integrity(experiment, mutation):
    directory, rows, mapping, automatic = experiment
    manual = analysis.unblind(rows, mapping, "context")
    if mutation == "duplicate":
        automatic[1] = copy.deepcopy(automatic[0])
    elif mutation == "missing":
        automatic.pop()
    elif mutation == "changed_answer":
        automatic[0]["answer"] = "modified output"
    elif mutation == "changed_context_hash":
        automatic[0]["context_sha256"] = "different"
    else:
        automatic[0]["prompt_sha256"] = "different"
    with pytest.raises(analysis.ReviewError):
        analysis.join_automatic(manual, list(automatic[0]), automatic)


def test_correlations_missing_and_constant_columns(experiment):
    directory, rows, mapping, automatic = experiment
    for row in automatic:
        del row["evidence_overlap_ratio"]
        row["keyword_overlap_ratio"] = "0.5"
    joined, skipped = analysis.join_automatic(analysis.unblind(rows, mapping, "context"), list(automatic[0]), automatic)
    assert skipped == ["evidence_overlap_ratio"]
    result = analysis.correlations(joined)
    assert result[0]["status"] == "test_not_applicable"
    assert result[1]["status"] == "skipped_missing_column"
    assert result[0]["spearman_rho"] is None


def test_complete_pipeline_exports_offline_and_preserves_inputs(experiment):
    directory, _rows, _mapping, _auto = experiment
    before = {name: (directory / name).read_bytes() for name in analysis.INPUT_NAMES}
    outside = set(directory.parent.parent.rglob('*')) - set(directory.rglob('*'))
    code, result = analysis.analyze(directory.name)
    assert code == 0 and result["n_pairs"] == 40
    assert result["api_calls"] == 0
    assert all((directory / name).read_bytes() == original for name, original in before.items())
    assert outside == set(directory.parent.parent.rglob('*')) - set(directory.rglob('*'))
    required = ["deepseek_cross_model_manual_review_unblinded.csv", "deepseek_cross_model_paired_manual_scores.csv",
        "deepseek_manual_evaluation_summary.csv", "deepseek_manual_significance_tests.csv", "deepseek_hallucination_mcnemar.csv",
        "deepseek_auto_manual_correlation.csv", "deepseek_manual_evaluation_report.md", "deepseek_manual_evaluation_table.tex",
        "deepseek_significance_table.tex", "deepseek_manual_statistics.json", "statistics_manifest.json"]
    assert all((directory / name).exists() for name in required)
    data = json.loads((directory / "deepseek_manual_statistics.json").read_text(encoding="utf-8"))
    assert data["n_pairs"] == 40 and data["n_questions"] == 40
    assert data["models"] == list(analysis.MODELS)
    assert data["methods"]["bootstrap"]["iterations"] == 10000
    assert data["methods"]["bootstrap"]["seed"] == 42
    assert data["source_file_sha256"] == {name: hashlib.sha256(original).hexdigest() for name, original in before.items()}
    assert data["analysis_script_sha256"] == analysis.file_sha256(Path(analysis.__file__))
    assert all(len(examples) <= 3 for examples in data["error_analysis"].values())
    assert data["correlations"][0]["spearman_rho"] == pytest.approx(1)
    for name in ("deepseek_manual_evaluation_table.tex", "deepseek_significance_table.tex"):
        latex = (directory / name).read_text(encoding="utf-8")
        assert r"\begin{table}" in latex and r"\begin{tabular}" in latex
        assert r"\documentclass" not in latex and r"\begin{document}" not in latex
        assert r"\toprule" in latex and r"\bottomrule" in latex
    first_statistics = data
    assert analysis.analyze(directory.name)[0] == 0
    second_statistics = json.loads((directory / "deepseek_manual_statistics.json").read_text(encoding="utf-8"))
    first_statistics.pop("generated_at")
    second_statistics.pop("generated_at")
    assert first_statistics == second_statistics


def test_cli_exit_codes_and_safe_id(experiment, capsys):
    directory, rows, _mapping, _auto = experiment
    rows[0]["correctness"] = ""
    save_csv(directory / analysis.INPUT_NAMES[0], rows)
    assert analysis.main(["--experiment-id", directory.name, "--validate-only"]) == 0
    assert analysis.main(["--experiment-id", directory.name]) == 2
    assert analysis.main(["--experiment-id", "../escape", "--validate-only"]) == 1
    output = capsys.readouterr().out
    assert 'manual review incomplete' in output
    assert '"mapping"' not in output
