"""Builds the LangChain agent that analyzes pre-parsed transactions."""
from __future__ import annotations

import json
import argparse
import shlex
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
    "Calculate the total expenses and number of expense transactions from an "
    "already-parsed JSON list. Amounts are negative for expenses. This tool "
    "does not read or parse source documents."
))
def budget_report(transactions_json: str) -> str:
    try:
        transactions = json.loads(transactions_json)
    except json.JSONDecodeError as exc:
        return f"Could not parse transactions JSON: {exc}"

    if isinstance(transactions, dict):
        transactions = transactions.get("transactions", [])

    total = sum(float(t.get("amount", 0)) for t in transactions if float(t.get("amount", 0)) < 0)
    count = len([t for t in transactions if float(t.get("amount", 0)) < 0])

    return json.dumps({
        "total_expenses": round(abs(total), 2),
        "transaction_count": count,
        "note": "LLM will categorize spending and give the budget plan.",
        "transactions": transactions,
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
            "1. Use budget_report to calculate the total expenses and "
            "transaction count from the provided transactions JSON.\n"
            "2. Group the transactions into clear spending categories such as "
            "Groceries, Dining, Transport, Housing, Utilities, Entertainment, "
            "Shopping, Health, and Other. Sum each category.\n"
            "3. Give a concise summary of total spending and the top "
            "categories.\n"
            "4. Then give a personalized budget recommendation: suggest "
            "reasonable monthly limits per category, flag the largest "
            "unnecessary expenses, and list 3-5 actionable ways the user can "
            "save money.\n"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with the local budgeting agent; optionally load a PDF or CSV."
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Optional statement path (.pdf or .csv) to load at startup.",
    )
    args = parser.parse_args()

    agent = build_agent()
    messages: list[dict] = []

    def load_file(file_path: str) -> None:
        nonlocal messages
        transactions = load_transactions(file_path)
        messages = [{
            "role": "user",
            "content": _transaction_context(file_path, transactions),
        }]
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        last = messages[-1]
        print(getattr(last, "content", last))
        print(f"\nLoaded {len(transactions)} transactions from {Path(file_path).name}.")

    if args.file:
        try:
            load_file(args.file)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

    print("Budget agent ready. Use /load <path.pdf|path.csv>, /help, or /quit.")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text == "/quit":
            break
        if user_text == "/help":
            print("Commands: /load <path.pdf|path.csv>, /help, /quit")
            continue
        if user_text.startswith("/load"):
            try:
                command = shlex.split(user_text)
                if len(command) != 2:
                    raise ValueError("Usage: /load <path.pdf|path.csv>")
                load_file(command[1])
            except (OSError, ValueError) as exc:
                print(f"Could not load statement: {exc}")
            continue

        messages.append({"role": "user", "content": user_text})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        last = messages[-1]
        print(f"Agent: {getattr(last, 'content', last)}")


if __name__ == "__main__":
    main()