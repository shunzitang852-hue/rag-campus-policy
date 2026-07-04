import json
from pathlib import Path

from src.retrieval.dense_index import load_jsonl
from src.retrieval.real_hybrid import (
    evaluate_hybrid_strategy,
    fuse_real_rrf,
    fuse_real_weighted_score,
    normalize_scores,
)


REPO_ROOT = Path(r"F:\GraduationThesis\rag-campus-policy")
CONFIG_PATH = REPO_ROOT / "configs" / "retrieval_hybrid_real_embedding.yaml"
REAL_DENSE_EVALUATION_DIR = Path(r"F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding\evaluation")
BM25_EVALUATION_DIR = Path(r"F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval\evaluation")
HASH_HYBRID_EVALUATION_DIR = Path(r"F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval\evaluation")
OUTPUT_ROOT = Path(r"F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\hybrid_retrieval_real_embedding")
REPORT_PATH = OUTPUT_ROOT / "evaluation" / "real_hybrid_evaluation_report.json"
QA_JSONL = Path(r"F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl")


def test_real_hybrid_config_loadable():
    text = CONFIG_PATH.read_text(encoding="utf-8")

    assert "real_dense_weight: 0.5" in text
    assert "bm25_weight: 0.5" in text
    assert "rrf" in text
    assert "k: 60" in text


def test_real_embedding_dense_result_files_are_readable():
    for strategy in ["fixed_length", "article_based", "structure_aware"]:
        rows = load_jsonl(REAL_DENSE_EVALUATION_DIR / f"real_embedding_dense_search_results_{strategy}.jsonl")
        assert rows
        assert rows[0]["question_id"]
        assert "score" in rows[0]


def test_bm25_result_files_are_readable():
    for strategy in ["fixed_length", "article_based", "structure_aware"]:
        rows = load_jsonl(BM25_EVALUATION_DIR / f"bm25_search_results_{strategy}.jsonl")
        assert rows
        assert rows[0]["question_id"]
        assert "score" in rows[0]


def test_weighted_score_fusion_returns_real_dense_score_field():
    question = {
        "question_id": "q1",
        "question": "demo",
        "related_chunk_ids_fixed_length": ["c2"],
    }
    dense_rows = [
        {"rank": 1, "retrieved_chunk_id": "c1", "retrieved_document_title": "doc", "score": 0.2},
        {"rank": 2, "retrieved_chunk_id": "c2", "retrieved_document_title": "doc", "score": 0.1},
    ]
    bm25_rows = [{"rank": 1, "retrieved_chunk_id": "c2", "retrieved_document_title": "doc", "score": 9.0}]

    results = fuse_real_weighted_score(question, "fixed_length", dense_rows, bm25_rows)

    assert results
    assert "real_dense_score" in results[0]
    assert "dense_score" not in results[0]
    assert "hybrid_score" in results[0]


def test_rrf_fusion_returns_real_dense_score_field():
    question = {
        "question_id": "q1",
        "question": "demo",
        "related_chunk_ids_fixed_length": ["c2"],
    }
    dense_rows = [{"rank": 1, "retrieved_chunk_id": "c1", "retrieved_document_title": "doc", "score": 0.2}]
    bm25_rows = [{"rank": 1, "retrieved_chunk_id": "c2", "retrieved_document_title": "doc", "score": 9.0}]

    results = fuse_real_rrf(question, "fixed_length", dense_rows, bm25_rows)

    assert results
    assert "real_dense_score" in results[0]
    assert results[0]["fusion_method"] == "rrf"


def test_normalize_scores_handles_equal_scores():
    assert normalize_scores([4.0, 4.0, 4.0]) == [0.0, 0.0, 0.0]


def test_evaluation_functions_compute_recall_and_mrr():
    questions = [{"question_id": "q1", "question_type": "fact_lookup", "difficulty": "easy"}]
    rows = [
        {"question_id": "q1", "rank": 1, "is_relevant": False, "hybrid_score": 1.0},
        {"question_id": "q1", "rank": 2, "is_relevant": True, "hybrid_score": 0.9},
    ]
    metrics = evaluate_hybrid_strategy("fixed_length", "weighted_score", rows, questions)

    assert metrics["Recall@1"] == 0
    assert metrics["Recall@3"] == 1
    assert metrics["MRR@10"] == 0.5


def test_real_hybrid_report_contains_three_strategies_two_methods_and_40_questions():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert set(report["metrics_by_strategy"]) == {"fixed_length", "article_based", "structure_aware"}
    for metrics_by_method in report["metrics_by_strategy"].values():
        assert set(metrics_by_method) == {"weighted_score", "rrf"}
        for metrics in metrics_by_method.values():
            assert metrics["question_count"] == 40
            assert "Recall@5" in metrics
            assert "MRR@10" in metrics


def test_qa_testset_has_40_questions():
    rows = load_jsonl(QA_JSONL)

    assert len(rows) == 40


def test_old_hash_hybrid_output_directory_is_not_overwritten():
    old_report = HASH_HYBRID_EVALUATION_DIR / "hybrid_evaluation_report.json"
    new_report = REPORT_PATH

    assert old_report.exists()
    assert new_report.exists()
    assert old_report != new_report
