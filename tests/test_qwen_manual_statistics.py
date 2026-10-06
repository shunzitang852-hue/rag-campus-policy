"""Synthetic, offline tests; formal human scores are never written by tests."""
import copy
import socket

import pytest

from src.experiments import analyze_qwen_manual_review as q


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("network prohibited")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def rows():
    return [{"question_id": f"SYNTHETIC_{i:03}", "requested_model": m, "anonymous_model": "Synthetic " + m,
        "question": str(i), "evidence": str(i), "context_sha256": str(i),
        "prompt_sha256": str(i), "answer": str(i),
        **{k: 2 if k != "hallucination" else 0 for k in q.SCORE_FIELDS}}
        for i in range(40) for m in q.MODELS]


def test_pair_direction_and_binary_orientation():
    source = rows()
    source[1]["correctness"] = 0
    source[1]["hallucination"] = 1
    paired = q.build_pairs(source)
    assert paired[0]["delta_correctness"] == 2
    assert paired[0]["delta_hallucination"] == -1
    tests, binary = q.significance(paired)
    assert tests[0]["qwen_better_count"] == 1
    assert tests[0]["flash_better_count"] == 0
    assert tests[0]["effect_size"] == 1
    assert binary["flash_only_count"] == 1
    assert binary["qwen_only_count"] == 0
    assert binary["p_value"] == 1


def test_degenerate_tests_preserve_null():
    tests, binary = q.significance(q.build_pairs(rows()))
    for r in tests:
        assert r["status"] == "test_not_applicable"
        assert r["raw_p"] is r["adjusted_p"] is r["effect_size"] is None
        assert r["tie_count"] == 40
    assert binary["discordant_pairs"] == 0
    assert binary["p_value"] == 1


def test_exact_mcnemar_known_case():
    source = rows()
    for i in range(6):
        source[2*i]["hallucination"] = 1
    _, binary = q.significance(q.build_pairs(source))
    assert binary["qwen_only_count"] == 6
    assert binary["flash_only_count"] == 0
    assert binary["p_value"] == pytest.approx(0.03125)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "hash"])
def test_pair_integrity(problem):
    source = rows()
    if problem == "duplicate": source[1] = copy.deepcopy(source[0])
    elif problem == "missing": source.pop()
    else: source[1]["prompt_sha256"] = "changed"
    with pytest.raises(q.shared.ReviewError): q.build_pairs(source)


def test_bootstrap_reproducible_and_common_subset_reuses_scores():
    source = rows()
    source[0]["correctness"] = 1
    paired = q.build_pairs(source)
    before = copy.deepcopy(paired)
    assert q.descriptive(paired) == q.descriptive(paired)
    summary, _ = q.descriptive(paired[1:])
    assert summary[0]["N"] == 39
    assert summary[0]["correctness_mean"] == 2
    assert paired == before


def test_spearman_constant_and_nonconstant():
    joined = [{"requested_model": m, "keyword_overlap_ratio": i/39,
        "evidence_overlap_ratio": i/39, "auto_quality_ordinal": i % 3,
        "correctness": 2, "faithfulness": i % 3, "completeness": i % 3}
        for i in range(40) for m in q.MODELS]
    results = q.correlations(joined)
    assert len(results) == 27
    constant = [r for r in results if r["manual_metric"] == "correctness"]
    assert len(constant) == 9
    assert all(r["spearman_rho"] is None and r["p_value"] is None for r in constant)
    assert all(r["notes"] == "rho undefined due to constant manual scores" for r in constant)
    matched = [r for r in results if r["automatic_metric"] == "auto_quality_label" and r["manual_metric"] == "faithfulness"]
    assert all(r["spearman_rho"] == pytest.approx(1) for r in matched)


def test_invalid_scores_block_before_key(monkeypatch, tmp_path):
    source = rows()
    for row in source:
        for field in q.SCORE_FIELDS:
            row[field] = str(row[field])
    source[0]["correctness"] = ""
    monkeypatch.setattr(q, "load_source", lambda *a, **k: (b"", [], source, "evidence"))
    def deny_key(*a, **k):
        raise AssertionError("blind key accessed before validation")
    monkeypatch.setattr(q.shared, "read_blind_mapping", deny_key)
    with pytest.raises(q.shared.ReviewError, match="manual validation failed"):
        q.analyze(tmp_path)


def test_latex_fragment_structure_and_escape():
    tex = q.latex_table("caption", "tab:test", ["model", "p"], [["qwen_test", "NA"]])
    assert "qwen\\_test" in tex
    for env in ("table", "tabular"):
        assert tex.count("\\begin{"+env+"}") == tex.count("\\end{"+env+"}") == 1
