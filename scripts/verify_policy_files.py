"""Verify local policy PDF files without parsing full policy text.

The script is intentionally local-only: it does not download files, perform OCR,
write extracted text, or modify the original PDFs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PDF_HEADER = b"%PDF-"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_text_layer(reader: Any, max_pages: int = 3) -> str:
    texts: list[str] = []
    for page in list(reader.pages[:max_pages]):
        try:
            text = page.extract_text() or ""
        except Exception:
            return "unknown"
        if text.strip():
            texts.append(text.strip())
    return "true" if texts else "false"


def inspect_pdf(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file() if path.exists() else False,
        "extension": path.suffix.lower(),
        "size_bytes": None,
        "sha256": None,
        "pdf_header": "unknown",
        "page_count": "unknown",
        "is_encrypted": "unknown",
        "has_text_layer": "unknown",
        "errors": [],
    }

    if not path.exists():
        result["errors"].append("file_not_found")
        return result
    if not path.is_file():
        result["errors"].append("not_a_file")
        return result

    result["size_bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)

    try:
        with path.open("rb") as handle:
            result["pdf_header"] = "true" if handle.read(len(PDF_HEADER)) == PDF_HEADER else "false"
    except OSError as exc:
        result["errors"].append(f"header_read_error: {exc}")
        return result

    if result["pdf_header"] != "true":
        result["errors"].append("invalid_pdf_header")
        return result

    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        result["errors"].append("pypdf_not_available")
        return result

    try:
        reader = PdfReader(str(path))
        result["is_encrypted"] = bool(reader.is_encrypted)
        if reader.is_encrypted:
            result["errors"].append("encrypted_pdf")
            return result
        result["page_count"] = len(reader.pages)
        result["has_text_layer"] = detect_text_layer(reader)
    except Exception as exc:
        result["errors"].append(f"pdf_read_error: {exc}")

    return result


def iter_pdf_targets(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*.pdf") if p.is_file())
    return [path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify local policy PDF files.")
    parser.add_argument("path", help="PDF file path or directory containing PDFs.")
    args = parser.parse_args(argv)

    target = Path(args.path)
    results = [inspect_pdf(path) for path in iter_pdf_targets(target)]
    print(json.dumps(results, ensure_ascii=False, indent=2))

    has_errors = any(item["errors"] for item in results)
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
