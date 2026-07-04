import importlib.util
import json
from pathlib import Path

from scripts.build_real_embedding_dense_retrieval import load_simple_yaml
from scripts.evaluate_real_embedding_dense_retrieval import search_with_provider, write_report_only
from src.retrieval.dense_index import load_jsonl
from src.retrieval.embedding_providers import (
    DeterministicHashProvider,
    LocalModelPlaceholderProvider,
    ProviderUnavailable,
    create_provider,
    encode_batches,
    validate_same_dimension,
)
from src.retrieval.evaluation import annotate_results, evaluate_strategy


REPO_ROOT = Path(r"F:\GraduationThesis\rag-campus-policy")
CONFIG_PATH = REPO_ROOT / "configs" / "retrieval_dense_real_embedding.yaml"
CHUNK_ROOT = Path(r"F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks")
QA_JSONL = Path(r"F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\qa_testset\hdu_ies_student_handbook_qa_testset_v1.jsonl")
OUTPUT_ROOT = Path(r"F:\GraduationThesis\experiment_outputs\hdu_ies\student_handbook\dense_retrieval_real_embedding")


def test_real_embedding_config_loadable():
    config = load_simple_yaml(CONFIG_PATH)

    assert config["embedding_provider"] == "sentence_transformers"
    assert config["model_name"] == "BAAI/bge-small-zh-v1.5"
    assert set(config["strategies"]) == {"fixed_length", "article_based", "structure_aware"}
    assert {"deterministic_hash", "sentence_transformers", "local_model_placeholder"} <= set(config["supported_providers"])


def test_deterministic_hash_provider_still_available():
    provider = create_provider({"embedding_provider": "deterministic_hash", "embedding_dimension": "384"})
    vectors = provider.encode(["policy question", "policy answer"])

    assert provider.metadata()["provider"] == "deterministic_hash"
    assert len(vectors) == 2
    assert {len(vector) for vector in vectors} == {384}


def test_provider_interface_and_dimension_validation_exist():
    provider = DeterministicHashProvider(dimension=16)
    vectors, dimension = encode_batches(provider, ["a", "b", "c"], batch_size=2)

    assert dimension == 16
    assert len(vectors) == 3
    assert validate_same_dimension(vectors) == 16


def test_provider_output_dimension_is_consistent():
    provider = DeterministicHashProvider(dimension=32)
    vectors = provider.encode(["学生管理", "奖学金申请", "考试纪律"])

    assert validate_same_dimension(vectors) == 32


def test_three_chunk_files_are_readable():
    for strategy in ["fixed_length", "article_based", "structure_aware"]:
        rows = load_jsonl(CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl")
        assert rows
        assert rows[0]["chunk_id"]
        assert rows[0]["chunk_text"]


def test_qa_testset_readable():
    rows = load_jsonl(QA_JSONL)

    assert len(rows) == 40
    assert all(row["question"] for row in rows)


def test_evaluation_functions_are_reusable_for_real_embedding_rows():
    question = {
        "question_id": "q1",
        "question": "demo",
        "question_type": "fact_lookup",
        "difficulty": "easy",
        "related_chunk_ids_fixed_length": ["c2"],
    }
    results = [
        {"rank": 1, "retrieved_chunk_id": "c1", "retrieved_document_title": "doc", "score": 0.4},
        {"rank": 2, "retrieved_chunk_id": "c2", "retrieved_document_title": "doc", "score": 0.3},
    ]
    rows = annotate_results(question, "fixed_length", results)
    metrics = evaluate_strategy("fixed_length", rows, [question], [1, 3, 5, 10])

    assert metrics["Recall@1"] == 0
    assert metrics["Recall@3"] == 1
    assert metrics["MRR@10"] == 0.5


def test_search_with_provider_uses_only_question_vector():
    provider = DeterministicHashProvider(dimension=32)
    chunks = [
        {"chunk_id": "c1", "document_title": "doc", "chunk_text": "学生 奖学金", "embedding": provider.encode(["学生 奖学金"])[0]},
        {"chunk_id": "c2", "document_title": "doc", "chunk_text": "考试 纪律", "embedding": provider.encode(["考试 纪律"])[0]},
    ]

    results = search_with_provider("学生奖学金", chunks, provider, top_k=2)

    assert len(results) == 2
    assert results[0]["retrieved_chunk_id"]
    assert "score" in results[0]


def test_unavailable_real_provider_fallback_report_is_controlled(tmp_path):
    report = write_report_only(
        tmp_path,
        "sentence-transformers is not installed in the active Python environment",
        {"embedding_provider": "sentence_transformers", "model_name": "BAAI/bge-small-zh-v1.5"},
    )

    report_path = tmp_path / "evaluation" / "real_embedding_dense_evaluation_report.json"
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model_ran"] is False
    assert saved["model_ran"] is False
    assert saved["metrics_by_strategy"] == {}


def test_sentence_transformers_provider_availability_matches_environment():
    available = importlib.util.find_spec("sentence_transformers") is not None
    if available:
        assert available
    else:
        try:
            create_provider({"embedding_provider": "sentence_transformers", "model_name": "BAAI/bge-small-zh-v1.5"})
        except ProviderUnavailable as exc:
            assert "sentence-transformers" in str(exc)
        else:
            raise AssertionError("sentence_transformers provider should be unavailable")


def test_local_model_placeholder_is_explicitly_unavailable():
    provider = LocalModelPlaceholderProvider(Path(r"F:\GraduationThesis\models\missing"))

    try:
        provider.encode(["demo"])
    except ProviderUnavailable as exc:
        assert "local_model_placeholder" in str(exc)
    else:
        raise AssertionError("placeholder provider should not silently generate vectors")
