"""Builds the LangChain agent that analyzes pre-parsed transactions."""
from __future__ import annotations

import json
import argparse
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

if __package__:
    from . import csv_extractor, pdf_extractor
    from .config import get_chat_model
else:
    import csv_extractor
    import pdf_extractor
    from config import get_chat_model

from langchain.agents import create_agent
from langchain_core.tools import tool


@tool("budget_report", description=(
    "Calculate the exact total expenses and count of expense transactions "
    "from a list of transaction objects. Pass the transaction list directly "
    "as the transactions argument, not as a JSON-encoded string. Amounts are "
    "negative for expenses. This tool does not read source documents."
))
def budget_report(transactions: list[dict]) -> str:
    total = Decimal("0")
    count = 0
    for index, transaction in enumerate(transactions):
        try:
            amount = Decimal(str(transaction["amount"]))
        except (KeyError, InvalidOperation, TypeError) as exc:
            raise ValueError(
                f"Transaction {index + 1} has a missing or invalid amount."
            ) from exc
        if not amount.is_finite():
            raise ValueError(f"Transaction {index + 1} has a non-finite amount.")
        if amount < 0:
            total += abs(amount)
            count += 1

    return json.dumps({
        "total_expenses": str(total.quantize(Decimal("0.01"))),
        "transaction_count": count,
        "note": "Use these calculated values exactly; do not recalculate them.",
    }, indent=2)


def build_agent():
    model = get_chat_model()
    agent = create_agent(
        model=model,
        tools=[budget_report],
        system_prompt=(
            "You are a personal budgeting assistant. Your job is to help the "
            "user understand and improve their spending. The user message "
            "contains transactions already extracted deterministically from "
            "the source document; never try to read or re-parse a file.\n\n"
            "Workflow:\n"
            "1. Call budget_report with the transactions list directly and "
            "use its returned total and count exactly; do not recalculate them.\n"
            "2. Categorize only negative-amount transactions as spending. "
            "Positive amounts are inflows and must not be presented as expenses.\n"
            "3. Group expenses into clear spending categories such as "
            "Groceries, Dining, Transport, Housing, Utilities, Entertainment, "
            "Shopping, Health, and Other. Sum each category.\n"
            "4. Give a concise summary of total spending and the top "
            "categories.\n"
            "5. Offer practical saving suggestions, but do not assume a "
            "monthly budget or make unsupported savings estimates unless the "
            "transaction period is clear. Do not label purchases necessary or "
            "unnecessary based only on merchant names.\n"
            "Be friendly, specific, and use actual numbers from the data."
        ),
        debug=False,
    )
    return agent


def analyze_file(file_path: str | Path):
    """Parse a PDF or CSV locally, then send normalized transactions to the agent."""
    transactions = load_transactions(file_path)
    transactions_json = json.dumps(transactions, ensure_ascii=False)
    return build_agent().invoke({"messages": [
        {"role": "user", "content": (
            "Analyze these already-extracted transactions and give me "
            "a full budget report and saving recommendations based on them."
            f"\nTransactions JSON:\n{transactions_json}"
        )}
    ]})


def load_transactions(file_path: str | Path) -> list[dict]:
    """Parse a supported statement locally and return normalized transactions."""
    file_path = Path(file_path)
    file_type = file_path.suffix.lower()
    if file_type == ".csv":
        transactions = csv_extractor.extract_transactions(file_path)
    elif file_type == ".pdf":
        transactions = pdf_extractor.extract_transactions(file_path)
    else:
        raise ValueError("file must have a .pdf or .csv extension")

    if not transactions:
        raise ValueError(
            "no transactions could be parsed; check that this is a supported "
            "statement file (PDFs must contain selectable text)"
        )

    return transactions


def _transaction_context(file_path: str | Path, transactions: list[dict]) -> str:
    transactions_json = json.dumps(transactions, ensure_ascii=False)
    return (
        f"Transactions were extracted locally from {Path(file_path).name}. "
        "Use this data for the user's budgeting questions. Do not attempt to "
        "read or re-parse the source file.\n"
        f"Transactions JSON:\n{transactions_json}"
    )


def _document_path_in_message(message: str) -> Path | None:
    match = re.search(
        r"""(?:"([^"\n]+\.(?:pdf|csv))"|'([^'\n]+\.(?:pdf|csv))'|([^\s"'<>]+\.(?:pdf|csv)))""",
        message,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return Path(next(group for group in match.groups() if group is not None)).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with the local budgeting agent using terminal input."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Optional PDF or CSV statement path to analyze at startup.",
    )
    args = parser.parse_args()

    agent = build_agent()
    messages: list[dict] = []

    def ask_agent(user_text: str, file_path: str | None = None) -> None:
        nonlocal messages
        if file_path is not None:
            transactions = load_transactions(file_path)
            user_text = f"{user_text}\n\n{_transaction_context(file_path, transactions)}"
        messages.append({"role": "user", "content": user_text})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print(getattr(messages[-1], "content", messages[-1]))

    if args.file:
        try:
            ask_agent("Please analyze this statement and summarize my spending.", args.file)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

    print(
        "Budget agent ready. Chat normally; include a PDF or CSV path in your "
        "message to analyze it. Press Ctrl-D or Ctrl-C to exit."
    )
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        file_path = _document_path_in_message(user_text)
        try:
            ask_agent(user_text, str(file_path) if file_path else None)
        except (OSError, ValueError) as exc:
            print(f"Could not analyze the statement: {exc}")


if __name__ == "__main__":
    main()