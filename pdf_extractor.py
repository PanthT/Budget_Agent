"""Parse expense transactions from a bank/credit-card statement PDF.

Uses pdfplumber to extract tables and selectable page text deterministically
(no OCR), returning clean transaction records: {date, description, category,
amount}.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pdfplumber

# Tokens that indicate the transaction is a debit (expense) vs. an amount field.
_AMOUNT_RE = re.compile(r"-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?")

# Negative amounts / parentheses like (12.34) both mean an expense/debit.
_MONEY_RE = re.compile(r"\(\s*-?[\d.,]+\s*\)|-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})")


def _to_number(s: str | None) -> float | None:
    if not s:
        return None
    s = s.strip().replace(" ", "")
    if not s:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    if s.startswith("-"):
        negative = True
        s = s[1:]
    if "," in s and "." in s:
        # Assume European-style 1.234,56 -> 1234.56
        if s.rindex(",") > s.rindex("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        val = float(s)
    except ValueError:
        return None
    return -val if negative else val


def _looks_like_date(text: str) -> bool:
    t = text.strip()
    # Matches: 2024-01-15, 01/15/2024, 01/15, 15-Jan-24, Jan 15 2024
    return bool(re.match(
        r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
        r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
        r"|\d{1,2}[-/.]\d{1,2}"
        r"|\d{1,2}[ -][A-Za-z]{3}[ -]?\d{0,4}"
        r"|[A-Za-z]{3}[ -]\d{1,2},?[ -]?\d{0,4})$",
        t,
        re.IGNORECASE,
    ))


def extract_transactions(pdf_path: str | Path) -> list[dict]:
    """Return a list of transaction dicts parsed from the statement PDF."""
    transactions: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_transactions: list[dict] = []
            for table in page.extract_tables():
                page_transactions.extend(_rows_from_table(table))

            # Some statement pages have no detected table even when earlier
            # pages did, so decide whether to use text extraction per page.
            if not page_transactions:
                text = page.extract_text() or ""
                page_transactions.extend(_rows_from_text(text))

            transactions.extend(page_transactions)

    return transactions


def _rows_from_table(table) -> list[dict]:
    rows: list[dict] = []
    for raw in table:
        cells = [(c or "").strip() for c in raw]
        if not any(cells):
            continue
        joined = " | ".join(cells)
        # Skip headers
        if _looks_like_header(joined):
            continue

        # Try to find date + amount in the row
        date = next((c for c in cells if _looks_like_date(c)), None)
        amounts = [_to_number(c) for c in cells if _valid_amount(c)]
        amounts = [a for a in amounts if a is not None]

        if date and amounts:
            desc = " ".join(c for c in cells if c and c != date and not _valid_amount(c)).strip()
            # pick the largest-magnitude amount as the transaction value
            amount = max(amounts, key=abs)
            rows.append({
                "date": date,
                "description": desc or "Unknown",
                "category": "uncategorized",
                "amount": amount,
            })
    return rows


def _rows_from_text(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _looks_like_header(line):
            continue
        # Date at the start, amount at the end:  "01/15  Whole Foods   -45.20"
        mdate = re.match(r"^([\d.?/.\-][A-Za-z0-9 ,/.-]*?)\s{1,}(.+?)\s+(-?[\d.,]+)$", line)
        if not mdate:
            continue
        date_candidate, desc_candidate, amt_candidate = mdate.groups()
        if not _looks_like_date(date_candidate):
            continue
        amount = _to_number(amt_candidate)
        if amount is None:
            continue
        rows.append({
            "date": date_candidate,
            "description": desc_candidate.strip(),
            "category": "uncategorized",
            "amount": amount,
        })
    return rows


def _valid_amount(cell: str) -> bool:
    return bool(_MONEY_RE.fullmatch(cell.strip()))


def _looks_like_header(line: str) -> bool:
    lowered = line.lower()
    for token in ("date", "description", "amount", "balance", "transaction",
                  "details", "credit", "debit", "posted", "ref", "memo"):
        if token in lowered:
            return True
    return False


def to_json(transactions: list[dict]) -> str:
    return json.dumps(transactions, indent=2)