import json
from pathlib import Path

from src.preprocessing.models import SplitPolicy
from src.preprocessing.structure_parser import parse_policy


def test_parse_policy_detects_chapter_article_clause_and_items():
    policy = SplitPolicy(
        document_id="hdu_ies_student_handbook",
        policy_id="TEST_001",
        sequence=1,
        policy_title_original="测试办法",
        policy_title_canonical="测试办法",
        title_match_status="exact",
        issuer="杭州电子科技大学信息工程学院",
        document_number="信工测〔2026〕1号",
        policy_year="2026",
        revision_status="试行",
        policy_category="测试",
        source_start_line=100,
        source_end_line=106,
        raw_text="\n".join(
            [
                "测试办法",
                "第一章 总则",
                "第一条 为规范测试，制定本办法。",
                "（一）保留款项。",
                "1. 保留数字子项。",
                "普通段落。",
            ]
        ),
        verification_status="pending_independent_verification",
        notes="",
    )

    records, anomalies = parse_policy(policy)
    statuses = [record.parse_status for record in records]
    assert "chapter_heading" in statuses
    assert "article" in statuses
    assert "clause" in statuses
    assert "item" in statuses
    assert all(record.source_start_line <= record.source_end_line for record in records)
    assert records[2].article_number == "第一条"


def test_parse_policy_keeps_unclassified_text():
    policy = SplitPolicy(
        document_id="hdu_ies_student_handbook",
        policy_id="TEST_002",
        sequence=2,
        policy_title_original="测试办法",
        policy_title_canonical="测试办法",
        title_match_status="exact",
        issuer="",
        document_number="",
        policy_year="",
        revision_status="",
        policy_category="",
        source_start_line=1,
        source_end_line=3,
        raw_text="测试办法\n没有编号的说明\n另一段说明",
        verification_status="pending_independent_verification",
        notes="",
    )

    records, anomalies = parse_policy(policy)
    assert len(records) == 3
    assert any(record.parse_status == "unclassified" for record in records)
    assert any(anomaly.anomaly_type == "unclassified_text" for anomaly in anomalies)


def test_generated_articles_jsonl_is_parseable_if_present():
    path = Path(r"F:\GraduationThesis\datasets\interim\hdu_ies\student_handbook\hdu_ies_student_handbook_articles.jsonl")
    if not path.exists():
        return
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert row["record_id"] not in seen
            seen.add(row["record_id"])
            assert row["policy_id"]
            assert row["content_raw"]
            assert row["source_start_line"] <= row["source_end_line"]
