import json
import subprocess
from pathlib import Path

from src.chunking.chunker import (
    article_based_chunks,
    fixed_length_chunks,
    generate_chunks,
    stable_chunk_id,
    structure_aware_chunks,
)
from src.chunking.config import get_strategy_config
from src.chunking.validation import chunk_stats
from src.preprocessing.policy_splitter import EXPECTED_RAW_SHA256, sha256_file


ARTICLES = Path(r"F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\hdu_ies_student_handbook_articles_clean.jsonl")
CHUNK_ROOT = Path(r"F:\GraduationThesis\datasets\processed\hdu_ies\student_handbook\chunks")
RAW_PATH = Path(r"F:\GraduationThesis\datasets\raw\hdu_ies\student_handbook\hdu_ies_student_handbook_source.txt")
CLEANED_PATH = Path(r"F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\cleaned\hdu_ies_student_handbook_cleaned.txt")


def load_records(limit=None):
    rows = []
    with ARTICLES.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def test_fixed_length_basic_and_overlap():
    config = get_strategy_config("fixed_length")
    records = load_records(80)
    chunks = fixed_length_chunks(records, config)

    assert chunks
    assert all(chunk.chunk_char_count <= config.chunk_size for chunk in chunks)
    assert any(chunk.metadata.get("split_start_offset", 0) > 0 for chunk in chunks)


def test_chinese_sentence_boundary_cut():
    config = get_strategy_config("fixed_length")
    records = load_records(30)
    chunk = fixed_length_chunks(records, config)[0]

    assert chunk.chunk_text[-1] in set(config.sentence_boundary_chars) or chunk.chunk_char_count == config.chunk_size


def test_article_based_keeps_policy_boundary_and_metadata():
    config = get_strategy_config("article_based")
    records = load_records(120)
    chunks = article_based_chunks(records, config)

    assert chunks
    assert all(len({rid.split("_R")[0] for rid in chunk.metadata["source_record_ids"]}) == 1 for chunk in chunks)
    assert all(chunk.policy_title for chunk in chunks)


def test_structure_aware_chunks_are_deterministic():
    config = get_strategy_config("structure_aware")
    records = load_records(200)
    first = [chunk.to_dict() for chunk in structure_aware_chunks(records, config)]
    second = [chunk.to_dict() for chunk in structure_aware_chunks(records, config)]

    assert first == second


def test_stable_chunk_id_uses_no_random_uuid():
    one = stable_chunk_id("fixed_length", "doc", "policy", 1, "测试文本", ["r1"])
    two = stable_chunk_id("fixed_length", "doc", "policy", 1, "测试文本", ["r1"])

    assert one == two
    assert len(one) < 80


def test_empty_text_records_produce_no_chunks():
    config = get_strategy_config("fixed_length")
    record = load_records(1)[0]
    record["content_clean"] = ""
    record["content_raw"] = ""

    assert generate_chunks([record], config) == []


def test_chunk_coverage_and_duplicate_coverage_are_separate():
    config = get_strategy_config("article_based")
    records = load_records(300)
    chunks = [chunk.to_dict() for chunk in article_based_chunks(records, config)]
    stats = chunk_stats(chunks, records, config.__dict__)

    assert stats["body_coverage_ratio"] == 1.0
    assert "duplicate_coverage_ratio" in stats


def test_generated_chunk_jsonl_parseable_for_all_strategies():
    required_fields = {
        "chunk_id",
        "strategy",
        "document_id",
        "document_title",
        "chunk_index",
        "chunk_text",
        "char_count",
        "start_char",
        "end_char",
        "source_file",
        "metadata",
    }
    for strategy in ["fixed_length", "article_based", "structure_aware"]:
        path = CHUNK_ROOT / f"hdu_ies_student_handbook_chunks_{strategy}.jsonl"
        assert path.exists()
        with path.open("r", encoding="utf-8") as handle:
            first = json.loads(next(handle))
        assert required_fields <= set(first)
        assert first["strategy"] == strategy
        assert first["chunk_id"]
        assert first["chunk_text"].strip()
        assert Path(first["source_file"]) == CLEANED_PATH
        assert first["metadata"]["cleaned_source_path"] == str(CLEANED_PATH)
        assert "policy_title" in first["metadata"]


def test_raw_file_hash_unchanged_after_chunking():
    assert sha256_file(RAW_PATH) == EXPECTED_RAW_SHA256


def test_chunk_report_acceptance_metrics_pass():
    report = json.loads((CHUNK_ROOT / "hdu_ies_student_handbook_chunking_report.json").read_text(encoding="utf-8"))

    assert report["input_file"] == str(CLEANED_PATH)
    for stats in report["strategies"].values():
        assert stats["empty_chunk_count"] == 0
        assert stats["covered_policy_count"] == 32
        assert stats["missing_policy_count"] == 0
        assert stats["max_chars"] <= 900


def test_chunk_outputs_are_not_tracked_by_git():
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=r"F:\GraduationThesis\rag-campus-policy",
        text=True,
        capture_output=True,
        check=True,
    )
    assert "datasets/processed/hdu_ies/student_handbook/chunks" not in result.stdout.replace("\\", "/")
