"""Generate acceptance-review artifacts for the student handbook parsing stage."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.policy_splitter import DEFAULT_INTERIM_ROOT, DEFAULT_RAW_PATH, DATASETS_ROOT, safe_filename
from src.preprocessing.structure_parser import load_jsonl
from src.preprocessing.text_cleaner import DEFAULT_PROCESSED_ROOT


REVIEW_SEED = 20260703
MANUAL_REVIEW_TITLES = [
    "学生学籍管理实施细则",
    "本科毕业生学士学位授予细则",
    "学生考试违规、作弊的认定办法",
    "学生纪律处分实施细则",
    "学生解除处分管理办法",
    "关于学生参加毕业设计（论文）的若干规定",
    "学生公寓管理办法（暂行）",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_outputs(interim_root: Path, processed_root: Path) -> dict[str, Any]:
    return {
        "policies": load_jsonl(interim_root / "hdu_ies_student_handbook_policies.jsonl"),
        "articles": load_jsonl(interim_root / "hdu_ies_student_handbook_articles.jsonl"),
        "anomalies": load_jsonl(interim_root / "parse_anomalies.jsonl"),
        "clean_articles": load_jsonl(processed_root / "hdu_ies_student_handbook_articles_clean.jsonl"),
        "stats_path": interim_root / "parsing_statistics.json",
    }


def line_number_map(rows: list[dict[str, Any]]) -> set[int]:
    covered: set[int] = set()
    for row in rows:
        covered.update(range(int(row["source_start_line"]), int(row["source_end_line"]) + 1))
    return covered


def classify_uncovered_line(line_no: int, text: str, first_policy_line: int, last_toc_line: int) -> tuple[str, str, bool]:
    stripped = text.strip()
    if not stripped:
        return "blank_or_formatting", "空行或排版留白，未进入结构化正文记录。", False
    if line_no <= last_toc_line and ("目 录" in stripped or "\t" in text or re.search(r"\d+\s*$", stripped)):
        return "table_of_contents", "制度目录区域，不归入制度正文。", True
    if line_no < last_toc_line and "目 录" not in stripped:
        return "preface", "制度正文开始前的前言或说明文字。", True
    if line_no < first_policy_line:
        return "separator", "目录与第一项制度之间的分隔或留白。", False
    return "policy_body", "位于制度正文范围内但未进入结构化记录，需视为解析缺陷。", True


def generate_coverage_gap_report(raw_path: Path, policies: list[dict[str, Any]], articles: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    raw_keepends = raw_path.read_text(encoding="utf-8").splitlines(keepends=True)
    covered_lines = line_number_map(articles)
    first_policy_line = min(int(row["source_start_line"]) for row in policies)
    last_policy_line = max(int(row["source_end_line"]) for row in policies)
    last_toc_line = first_policy_line - 1

    segments: list[dict[str, Any]] = []
    uncovered_policy_body_chars = 0
    uncovered_chars = 0
    for line_no, raw_line in enumerate(raw_keepends, start=1):
        text = raw_line.rstrip("\r\n")
        if line_no in covered_lines:
            continue
        classification, reason, requires_review = classify_uncovered_line(line_no, text, first_policy_line, last_toc_line)
        if classification == "policy_body" and text.strip():
            uncovered_policy_body_chars += len(raw_line)
        uncovered_chars += len(raw_line)
        segments.append(
            {
                "start_line": line_no,
                "end_line": line_no,
                "text": text,
                "classification": classification,
                "reason": reason,
                "requires_manual_review": requires_review,
            }
        )

    raw_char_count = len(raw_path.read_text(encoding="utf-8"))
    payload = {
        "raw_character_count": raw_char_count,
        "covered_character_count": raw_char_count - uncovered_chars,
        "uncovered_character_count": uncovered_chars,
        "coverage_ratio": (raw_char_count - uncovered_chars) / raw_char_count if raw_char_count else 0,
        "uncovered_policy_body_character_count": uncovered_policy_body_chars,
        "uncovered_segments": segments,
    }
    write_json(output_path, payload)
    return payload


def classify_unclassified(text: str) -> tuple[str, str, bool, bool, str]:
    stripped = text.strip()
    if re.search(r"令第\d+号|〔\d{4}〕|\[\d{4}\]|\(\d{4}", stripped):
        return "policy_basis", "保留为制度依据或文号说明行", False, True, "疑似文号、依据或发布说明。"
    if re.search(r"\d{4}年\d{1,2}月\d{1,2}日|自.*施行|负责解释|同时废止", stripped):
        return "signature_or_date", "保留为施行日期、解释权或废止说明", False, True, "包含日期、施行或解释说明。"
    if re.search(r"如下|下列|条件|标准|范围|对象|办法|规定", stripped):
        return "policy_preamble", "保留为制度前言或引导段", False, True, "制度说明、适用范围或列表引导语。"
    if re.search(r"\d+元|%|％|比例|名额|学分|个月|日内|期限", stripped):
        return "unnumbered_paragraph", "保留为无编号政策段落", False, True, "含金额、比例、期限或数量，需人工关注。"
    if re.match(r"^[，。、；;：:）)]", stripped) or stripped.endswith(("，", "、")):
        return "continuation_line", "可考虑归并到上一结构记录", True, True, "疑似排版断行或续行。"
    if len(stripped) <= 12 and re.search(r"表|附件|附录", stripped):
        return "table_caption", "保留并人工核验表格标题", False, True, "疑似表格或附件标题。"
    if stripped.startswith(("注", "说明", "备注")):
        return "supplementary_note", "保留为补充说明", False, True, "补充说明。"
    if stripped:
        return "unnumbered_paragraph", "保留为合法无编号段落", False, True, "未发现必须修改解析器的编号模式。"
    return "unknown", "保留并人工核验", False, True, "空或无法判断。"


def generate_unclassified_analysis(anomalies: list[dict[str, Any]], policy_map: dict[str, dict[str, Any]], output_path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    for index, anomaly in enumerate([a for a in anomalies if a["anomaly_type"] == "unclassified_text"], start=1):
        policy = policy_map.get(anomaly["policy_id"], {})
        classification, action, parser_change, manual_review, notes = classify_unclassified(anomaly["original_text"])
        rows.append(
            {
                "record_id": f"UNCLASSIFIED_{index:03d}",
                "policy_id": anomaly["policy_id"],
                "policy_title": policy.get("policy_title_canonical", ""),
                "source_start_line": anomaly["source_start_line"],
                "source_end_line": anomaly["source_end_line"],
                "original_text": anomaly["original_text"],
                "secondary_classification": classification,
                "suggested_action": action,
                "requires_parser_change": str(parser_change).lower(),
                "requires_manual_review": str(manual_review).lower(),
                "notes": notes,
            }
        )
    fields = [
        "record_id",
        "policy_id",
        "policy_title",
        "source_start_line",
        "source_end_line",
        "original_text",
        "secondary_classification",
        "suggested_action",
        "requires_parser_change",
        "requires_manual_review",
        "notes",
    ]
    write_csv(output_path, rows, fields)
    return rows, Counter(row["secondary_classification"] for row in rows)


def review_possible_table(text: str) -> dict[str, Any]:
    columns = [part for part in re.split(r"\t+|\s{2,}", text.strip()) if part]
    contains_amount = bool(re.search(r"\d+(\.\d+)?\s*元|万元|金额|奖学金", text))
    contains_percentage = bool(re.search(r"\d+(\.\d+)?\s*[%％]|比例|百分", text))
    contains_condition = bool(re.search(r"条件|符合|申请|评定|要求|不得|可以|应当", text))
    contains_date = bool(re.search(r"\d{4}年|\d+月|\d+日|期限|日内|个月|学年|学期", text))
    looks_table = len(columns) >= 2 or sum([contains_amount, contains_percentage, contains_condition, contains_date]) >= 2
    return {
        "detected_columns": len(columns) if columns else 1,
        "detected_rows": 1,
        "contains_amount": str(contains_amount).lower(),
        "contains_percentage": str(contains_percentage).lower(),
        "contains_condition": str(contains_condition).lower(),
        "contains_date_or_deadline": str(contains_date).lower(),
        "recommended_representation": "table_text_block" if looks_table else "single_text_line",
        "requires_parser_change": "false",
        "notes": "保留原文；本轮不强制恢复 Markdown 表格。" if looks_table else "更像排版空白触发的普通文本，保留原文并人工抽查。",
    }


def generate_possible_table_review(anomalies: list[dict[str, Any]], policy_map: dict[str, dict[str, Any]], output_path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    for index, anomaly in enumerate([a for a in anomalies if a["anomaly_type"] == "possible_table"], start=1):
        policy = policy_map.get(anomaly["policy_id"], {})
        review = review_possible_table(anomaly["original_text"])
        rows.append(
            {
                "table_candidate_id": f"TABLE_{index:03d}",
                "policy_id": anomaly["policy_id"],
                "policy_title": policy.get("policy_title_canonical", ""),
                "source_start_line": anomaly["source_start_line"],
                "source_end_line": anomaly["source_end_line"],
                "raw_text": anomaly["original_text"],
                "detected_columns": review["detected_columns"],
                "detected_rows": review["detected_rows"],
                "contains_amount": review["contains_amount"],
                "contains_percentage": review["contains_percentage"],
                "contains_condition": review["contains_condition"],
                "contains_date_or_deadline": review["contains_date_or_deadline"],
                "current_parse_result": "possible_table",
                "recommended_representation": review["recommended_representation"],
                "requires_parser_change": review["requires_parser_change"],
                "requires_manual_review": "true",
                "notes": review["notes"],
            }
        )
    fields = [
        "table_candidate_id",
        "policy_id",
        "policy_title",
        "source_start_line",
        "source_end_line",
        "raw_text",
        "detected_columns",
        "detected_rows",
        "contains_amount",
        "contains_percentage",
        "contains_condition",
        "contains_date_or_deadline",
        "current_parse_result",
        "recommended_representation",
        "requires_parser_change",
        "requires_manual_review",
        "notes",
    ]
    write_csv(output_path, rows, fields)
    return rows, Counter(row["policy_id"] for row in rows)


def group_by_policy(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["policy_id"]].append(row)
    return grouped


def sample_middle_articles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    articles = [row for row in rows if row["parse_status"] == "article"]
    if len(articles) <= 3:
        return articles
    mid = len(articles) // 2
    return articles[max(0, mid - 1) : min(len(articles), mid + 2)]


def find_multi_clause_article(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if row["parse_status"] == "article":
            if sum(1 for item in current if item["parse_status"] == "clause") >= 2:
                return current
            current = [row]
        elif current:
            current.append(row)
    if sum(1 for item in current if item["parse_status"] == "clause") >= 2:
        return current
    return []


def find_numeric_evidence(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if re.search(r"\d+元|%|％|\d{4}年|\d+个月|\d+日内|期限", row["content_raw"]):
            return row
    return None


def make_manual_review_packages(
    policies: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    clean_articles: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    article_map = group_by_policy(articles)
    clean_map = {row["record_id"]: row for row in clean_articles}
    anomaly_counts = Counter(row["policy_id"] for row in anomalies)
    table_counts = Counter(row["policy_id"] for row in table_rows)
    created: list[Path] = []

    for policy in policies:
        if not any(title in policy["policy_title_canonical"] for title in MANUAL_REVIEW_TITLES):
            continue
        rows = article_map.get(policy["policy_id"], [])
        chapters = [f"{row['chapter_number']} {row['chapter_title']}".strip() for row in rows if row["parse_status"] == "chapter_heading"]
        article_count = sum(1 for row in rows if row["parse_status"] == "article")
        clause_count = sum(1 for row in rows if row["parse_status"] == "clause")
        item_count = sum(1 for row in rows if row["parse_status"] in {"item", "cn_item"})
        unnumbered_count = sum(1 for row in rows if row["parse_status"] in {"paragraph", "unclassified"})
        middle = sample_middle_articles(rows)
        multi_clause = find_multi_clause_article(rows)
        numeric = find_numeric_evidence(rows)
        start_preview = "\n".join(policy["raw_text"].splitlines()[:8])
        end_preview = "\n".join(policy["raw_text"].splitlines()[-8:])
        compare_rows = rows[:3]

        content = [
            f"# {policy['policy_title_canonical']} 解析人工核验包",
            "",
            f"- policy_id: `{policy['policy_id']}`",
            f"- 制度规范标题: {policy['policy_title_canonical']}",
            f"- 制度原始标题: {policy['policy_title_original']}",
            f"- 制度分类: {policy['policy_category']}",
            f"- 制度起止行号: {policy['source_start_line']} - {policy['source_end_line']}",
            f"- 条款数量: {article_count}",
            f"- 款项数量: {clause_count}",
            f"- 子项数量: {item_count}",
            f"- 无编号段落数量: {unnumbered_count}",
            f"- 异常数量: {anomaly_counts.get(policy['policy_id'], 0)}",
            f"- 表格候选数量: {table_counts.get(policy['policy_id'], 0)}",
            "",
            "## 制度开头原文",
            "",
            "```text",
            start_preview,
            "```",
            "",
            "## 制度结尾原文",
            "",
            "```text",
            end_preview,
            "```",
            "",
            "## 章节列表",
            "",
        ]
        content.extend(f"- {item}" for item in chapters[:30])
        content.extend(["", "## 随机抽取3条中间条款", ""])
        for row in middle:
            content.extend(["```text", row["content_raw"], "```", ""])
        content.extend(["## 至少1条含多个款项的条款", ""])
        if multi_clause:
            content.extend(["```text", "\n".join(row["content_raw"] for row in multi_clause[:10]), "```", ""])
        else:
            content.append("未在自动解析结果中定位到含多个款项的条款，请人工核验。")
        content.extend(["", "## 含金额、比例、日期或期限的内容", ""])
        if numeric:
            content.extend(["```text", numeric["content_raw"], "```", ""])
        else:
            content.append("未自动定位到金额、比例、日期或期限内容，请人工核验。")
        content.extend(["", "## 原文和清洗文本对照", ""])
        for row in compare_rows:
            clean = clean_map.get(row["record_id"], {})
            content.extend(
                [
                    f"### {row['record_id']}",
                    "",
                    "原文：",
                    "",
                    "```text",
                    row["content_raw"],
                    "```",
                    "",
                    "清洗文本：",
                    "",
                    "```text",
                    clean.get("content_clean", ""),
                    "```",
                    "",
                ]
            )
        content.extend(
            [
                "## 人工核验填写区",
                "",
                "制度边界：通过 / 不通过",
                "",
                "标题：通过 / 不通过",
                "",
                "章节结构：通过 / 不通过",
                "",
                "条款结构：通过 / 不通过",
                "",
                "款项归属：通过 / 不通过",
                "",
                "数字与比例：通过 / 不通过",
                "",
                "表格语义：通过 / 不通过 / 不适用",
                "",
                "存在串文：是 / 否",
                "",
                "存在漏文：是 / 否",
                "",
                "是否允许进入Chunk阶段：是 / 否 / 修正后允许",
                "",
                "问题说明：",
                "",
                "人工核验人：",
                "",
                "核验日期：",
                "",
            ]
        )
        path = output_dir / f"{policy['sequence']:03d}_{safe_filename(policy['policy_title_canonical'])}_manual_review.md"
        path.write_text("\n".join(content), encoding="utf-8", newline="\n")
        created.append(path)
    return created


def write_coverage_markdown(path: Path, gap_report: dict[str, Any]) -> None:
    counts = Counter(segment["classification"] for segment in gap_report["uncovered_segments"])
    lines = [
        "# 学生手册解析覆盖率缺口报告",
        "",
        f"- raw 字符数: {gap_report['raw_character_count']}",
        f"- covered 字符数: {gap_report['covered_character_count']}",
        f"- uncovered 字符数: {gap_report['uncovered_character_count']}",
        f"- coverage_ratio: {gap_report['coverage_ratio']}",
        f"- uncovered_policy_body_character_count: {gap_report['uncovered_policy_body_character_count']}",
        "",
        "## 缺口分类统计",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
    lines.extend(["", "## 缺口明细", ""])
    for segment in gap_report["uncovered_segments"][:200]:
        lines.extend(
            [
                f"### line {segment['start_line']}",
                "",
                f"- classification: `{segment['classification']}`",
                f"- reason: {segment['reason']}",
                f"- requires_manual_review: {segment['requires_manual_review']}",
                "",
                "```text",
                segment["text"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_anomaly_review(
    path: Path,
    anomalies: list[dict[str, Any]],
    unclassified_rows: list[dict[str, Any]],
    unclassified_counts: Counter[str],
    table_rows: list[dict[str, Any]],
    table_policy_counts: Counter[str],
    policy_map: dict[str, dict[str, Any]],
    gap_report: dict[str, Any],
) -> None:
    rng = random.Random(REVIEW_SEED)
    sample_pool = list(unclassified_rows)
    sample = rng.sample(sample_pool, min(30, len(sample_pool)))
    anomaly_counts = Counter(row["anomaly_type"] for row in anomalies)
    lines = [
        "# 学生手册解析异常核验汇总",
        "",
        f"- 异常总数: {len(anomalies)}",
        f"- 固定随机种子: {REVIEW_SEED}",
        "",
        "## 异常类型数量",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(anomaly_counts.items()))
    lines.extend(["", "## unclassified_text 二次分类统计", ""])
    lines.extend(f"- {key}: {value}" for key, value in sorted(unclassified_counts.items()))
    lines.extend(
        [
            f"- parse_error: {unclassified_counts.get('parse_error', 0)}",
            f"- unknown: {unclassified_counts.get('unknown', 0)}",
            "",
            "## possible_table 按制度分布",
            "",
        ]
    )
    for policy_id, count in sorted(table_policy_counts.items()):
        title = policy_map.get(policy_id, {}).get("policy_title_canonical", "")
        lines.append(f"- {policy_id} {title}: {count}")
    lines.extend(["", "## unclassified_text 抽样30条", ""])
    for row in sample:
        lines.extend(
            [
                f"### {row['record_id']} `{row['secondary_classification']}`",
                "",
                f"- policy_id: {row['policy_id']}",
                f"- title: {row['policy_title']}",
                f"- line: {row['source_start_line']}",
                "",
                "```text",
                row["original_text"],
                "```",
                "",
            ]
        )
    lines.extend(["", "## possible_table 全量索引", ""])
    for row in table_rows:
        lines.append(f"- {row['table_candidate_id']} {row['policy_id']} line {row['source_start_line']}: {row['recommended_representation']}")
    lines.extend(["", "## title_mismatch 详情", ""])
    for row in anomalies:
        if row["anomaly_type"] == "title_mismatch":
            lines.extend(["```text", row["description"], "```", ""])
    lines.extend(["## missing_document_number 详情", ""])
    for row in anomalies:
        if row["anomaly_type"] == "missing_document_number":
            lines.extend(["```text", row["description"], "```", ""])
    lines.extend(
        [
            "## 覆盖率缺口详情",
            "",
            f"- uncovered_character_count: {gap_report['uncovered_character_count']}",
            f"- uncovered_policy_body_character_count: {gap_report['uncovered_policy_body_character_count']}",
            "",
            "## 建议人工核验顺序",
            "",
            "1. 学生解除处分管理办法",
            "2. 学生公寓管理办法（暂行）",
            "3. possible_table 候选较多的制度",
            "4. unclassified_text 中含金额、比例、期限或施行日期的记录",
            "5. 7项重点制度人工核验包",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    outputs = load_outputs(args.interim_root, args.processed_root)
    policies = outputs["policies"]
    articles = outputs["articles"]
    clean_articles = outputs["clean_articles"]
    anomalies = outputs["anomalies"]
    policy_map = {row["policy_id"]: row for row in policies}

    gap_report = generate_coverage_gap_report(
        args.raw_path,
        policies,
        articles,
        args.interim_root / "coverage_gap_report.json",
    )
    write_coverage_markdown(args.repo_root / "docs" / "data" / "hdu_ies_student_handbook_coverage_gap_report.md", gap_report)

    unclassified_rows, unclassified_counts = generate_unclassified_analysis(
        anomalies,
        policy_map,
        args.interim_root / "unclassified_text_analysis.csv",
    )
    table_rows, table_policy_counts = generate_possible_table_review(
        anomalies,
        policy_map,
        args.interim_root / "possible_table_review.csv",
    )
    manual_files = make_manual_review_packages(
        policies,
        articles,
        clean_articles,
        anomalies,
        table_rows,
        DATASETS_ROOT / "evaluation" / "hdu_ies" / "manual_review" / "parsing_stage",
    )
    write_anomaly_review(
        args.repo_root / "docs" / "data" / "hdu_ies_student_handbook_anomaly_review.md",
        anomalies,
        unclassified_rows,
        unclassified_counts,
        table_rows,
        table_policy_counts,
        policy_map,
        gap_report,
    )

    stats_path = outputs["stats_path"]
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats.update(
        {
            "coverage_gap_report": {
                "uncovered_character_count": gap_report["uncovered_character_count"],
                "uncovered_policy_body_character_count": gap_report["uncovered_policy_body_character_count"],
                "coverage_ratio": gap_report["coverage_ratio"],
            },
            "unclassified_secondary_counts": dict(sorted(unclassified_counts.items())),
            "unclassified_parse_error_count": unclassified_counts.get("parse_error", 0),
            "unclassified_unknown_count": unclassified_counts.get("unknown", 0),
            "possible_table_policy_counts": dict(sorted(table_policy_counts.items())),
            "manual_review_package_count": len(manual_files),
        }
    )
    write_json(stats_path, stats)

    print(f"coverage_gap_report={args.interim_root / 'coverage_gap_report.json'}")
    print(f"unclassified_rows={len(unclassified_rows)}")
    print(f"possible_table_rows={len(table_rows)}")
    print(f"manual_review_files={len(manual_files)}")
    print(f"uncovered_policy_body_character_count={gap_report['uncovered_policy_body_character_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
