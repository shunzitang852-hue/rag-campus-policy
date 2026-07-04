import json
from pathlib import Path

from src.retrieval.bm25 import build_index, search, tokenize
from src.retrieval.dense_index import load_jsonl
from src.retrieval.evaluation import evaluate_strategy


REPO_ROOT = Path(r"F:\GraduationThesis\rag-campus-policy")
CONFIG_PATH = REPO_ROOT / "configs" / "retrieval_bm25.yaml"
CHUNK_ROOT = Path(r"F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks")
QA_JSONL = Path(r"F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl")
OUTPUT_ROOT = Path(r"F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\bm25_retrieval")
REPORT_PATH = OUTPUT_ROOT / "evaluation" / "bm25_evaluation_report.json"


def test_bm25_config_loadable():
    text = CONFIG_PATH.read_text(encoding="utf-8")

    assert "cjk_char_bigram" in text
    assert "k1: 1.5" in text
    assert "b: 0.75" in text


def test_three_chunk_files_are_readable():
    for strategy in ["fixed_length", "article_based", "structure_aware"]:
        rows = load_jsonl(CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl")
        assert rows
        assert rows[0]["chunk_id"]
        assert rows[0]["chunk_text"]


def test_qa_testset_readable_and_has_40_questions():
    rows = load_jsonl(QA_JSONL)

    assert len(rows) == 40
    assert all(row["question"] for row in rows)


def test_tokenization_outputs_nonempty_tokens():
    tokens = tokenize("学生如何申请国家奖学金？")

    assert tokens
    assert "学" in tokens
    assert "学生" in tokens


def test_bm25_search_returns_nonempty_results():
    chunks = load_jsonl(CHUNK_ROOT / "hdu_ies_student_handbook_chunks_article_based.jsonl")[:30]
    index = build_index(chunks)
    results = search("学生申请国家奖学金需要符合什么条件？", index, top_k=10)

    assert results
    assert results[0]["retrieved_chunk_id"]
    assert "score" in results[0]


def test_evaluation_functions_compute_recall_and_mrr():
    questions = [{"question_id": "q1", "question_type": "fact_lookup", "difficulty": "easy"}]
    rows = [
        {"question_id": "q1", "rank": 1, "is_relevant": False, "score": 1.0},
        {"question_id": "q1", "rank": 2, "is_relevant": True, "score": 0.9},
    ]
    metrics = evaluate_strategy("demo", rows, questions, [1, 3, 5, 10])

    assert metrics["Recall@1"] == 0
    assert metrics["Recall@3"] == 1
    assert metrics["MRR@10"] == 0.5


def test_bm25_evaluation_report_contains_three_strategies_and_40_questions():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert set(report["metrics_by_strategy"]) == {"fixed_length", "article_based", "structure_aware"}
    for metrics in report["metrics_by_strategy"].values():
        assert metrics["question_count"] == 40
        assert "Recall@5" in metrics
        assert "MRR@10" in metrics
