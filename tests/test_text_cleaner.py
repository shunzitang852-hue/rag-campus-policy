from src.preprocessing.text_cleaner import clean_record, clean_text


def test_clean_text_is_conservative_for_policy_tokens():
    raw = "信工学〔2023〕9号　\n学费1000元，占比20%，自2023年9月1日起施行。   \n\n\n"
    cleaned, actions = clean_text(raw)

    assert "信工学〔2023〕9号" in cleaned
    assert "1000元" in cleaned
    assert "20%" in cleaned
    assert "2023年9月1日" in cleaned
    assert "normalize_full_width_spaces" in actions
    assert "strip_trailing_spaces" in actions
    assert "collapse_excess_blank_lines" in actions


def test_clean_record_keeps_raw_and_adds_clean_fields():
    record = {"content_raw": "第一条　测试内容。   "}
    cleaned = clean_record(record)

    assert cleaned["content_raw"] == "第一条　测试内容。   "
    assert cleaned["content_clean"] == "第一条 测试内容。"
    assert cleaned["cleaning_actions"]
