import csv
import json
from pathlib import Path


REPO_ROOT = Path(r"F:\GraduationThesis\rag-campus-policy")
INTERIM_ROOT = Path(r"F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook")
EVALUATION_ROOT = Path(r"F:\GraduationThesis\datasets\evaluation\hdu_ies\manual_review\parsing_stage")
RAW_PATH = Path(r"F:\GraduationThesis\datasets\raw\hdu_ies\student_handbook\hdu_ies_student_handbook_source.txt")
EXPECTED_HASH = "83937E33D6E10780DF6BEDEBFB9FD26A2D5D6DE940CF85E8D3F13E3D32165189"
OFFICIAL_TITLE = "基于RAG技术的高校政策知识问答系统构建与检索策略优化研究"
WRONG_TITLE = OFFICIAL_TITLE + "——以杭州电子科技大学信息工程学院为例"


def test_coverage_gap_report_has_no_uncovered_policy_body():
    report = json.loads((INTERIM_ROOT / "coverage_gap_report.json").read_text(encoding="utf-8"))

    assert report["uncovered_policy_body_character_count"] == 0
    assert report["uncovered_segments"]


def test_unclassified_analysis_covers_all_189_records():
    with (INTERIM_ROOT / "unclassified_text_analysis.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 189
    assert all(row["secondary_classification"] for row in rows)


def test_possible_table_review_covers_all_43_records():
    with (INTERIM_ROOT / "possible_table_review.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 43
    assert all(row["recommended_representation"] for row in rows)


def test_manual_review_package_files_exist_for_seven_key_policies():
    files = list(EVALUATION_ROOT.glob("*_manual_review.md"))

    assert len(files) == 7
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "制度边界：" in text
        assert "是否允许进入Chunk阶段：" in text


def test_official_title_is_preserved_and_wrong_title_absent():
    text = "\n".join(
        path.read_text(encoding="utf-8-sig", errors="ignore")
        for path in list(REPO_ROOT.glob("*.md")) + list((REPO_ROOT / "docs").rglob("*.md")) + list((REPO_ROOT / "configs").glob("*.yaml"))
    )

    assert OFFICIAL_TITLE in text
    assert WRONG_TITLE not in text


def test_no_chunk_embedding_or_vector_artifacts_exist():
    forbidden = []
    for pattern in ["*embedding*", "*.sqlite3", "*.faiss", "*.index", "*.parquet"]:
        forbidden.extend(Path(r"F:\GraduationThesis\datasets").rglob(pattern))

    assert not forbidden
