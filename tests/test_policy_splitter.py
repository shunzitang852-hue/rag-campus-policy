from pathlib import Path

from src.preprocessing.policy_splitter import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_RAW_PATH,
    EXPECTED_RAW_SHA256,
    find_title_match,
    find_body_search_start,
    read_catalog,
    read_raw_lines,
    safe_filename,
    sha256_file,
    split_policies,
)


def test_raw_hash_matches_expected():
    assert sha256_file(DEFAULT_RAW_PATH) == EXPECTED_RAW_SHA256


def test_catalog_has_32_entries():
    catalog = read_catalog(DEFAULT_CATALOG_PATH)
    assert len(catalog) == 32
    assert [entry.sequence for entry in catalog] == list(range(1, 33))


def test_split_real_handbook_has_unique_continuous_nonempty_policies():
    lines = read_raw_lines(DEFAULT_RAW_PATH)
    catalog = read_catalog(DEFAULT_CATALOG_PATH)
    policies, anomalies = split_policies(lines, catalog)

    assert len(policies) == 32
    assert len({policy.policy_id for policy in policies}) == 32
    assert [policy.sequence for policy in policies] == list(range(1, 33))
    assert all(policy.raw_text.strip() for policy in policies)
    assert all(policy.source_start_line <= policy.source_end_line for policy in policies)
    assert all(policies[index].source_end_line < policies[index + 1].source_start_line for index in range(31))
    assert any(anomaly.anomaly_type == "title_mismatch" for anomaly in anomalies)


def test_controlled_fuzzy_title_is_recorded_for解除处分():
    lines = read_raw_lines(DEFAULT_RAW_PATH)
    catalog = read_catalog(DEFAULT_CATALOG_PATH)
    entry = [item for item in catalog if item.sequence == 14][0]
    index, original, status, basis = find_title_match(lines, entry, find_body_search_start(lines, catalog))

    assert index is not None
    assert status == "fuzzy"
    assert "学生校解除处分管理办法" in original
    assert basis.startswith("controlled_marker")


def test_safe_filename_removes_illegal_characters():
    assert safe_filename('001:测试/文件*名?.txt') == "001_测试_文件_名_.txt"
