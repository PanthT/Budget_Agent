"""Builds the LangChain budgeting agent.

The agent has tools to (1) extract transactions from an expense PDF,
(2) extract transactions from an expense CSV, and
(3) analyze/categorize them into a personalized budget recommendation.
"""
from __future__ import annotations

import json

from langchain.agents import create_agent
from langchain_core.tools import tool

if __package__:
    from . import csv_extractor, pdf_extractor
    from .config import get_chat_model
else:
    import csv_extractor
    import pdf_extractor
    from config import get_chat_model


@tool("load_expenses", description=(
    "Extract expenses from a searchable (text-based) bank/credit-card statement PDF. "
    "Image-only scanned PDFs are not supported because this tool does not run OCR. "
    "Takes the file path to the PDF. Returns a JSON list of transactions "
    "with fields: date, description, category, amount (negative = money spent)."
))
def load_expenses(pdf_path: str) -> str:
    transactions = pdf_extractor.extract_transactions(pdf_path)
    if not transactions:
        return json.dumps(
            {"error": "No transactions could be parsed. "
                      "The file may not be a statement or uses an unsupported layout."}
        )
    return pdf_extractor.to_json(transactions)


@tool("load_csv_expenses", description=(
    "Extract the list of expenses from a bank/credit-card statement CSV file. "
    "Takes the file path to the CSV. Returns a JSON list of transactions "
    "with fields: date, description, category, amount (negative = money spent). "
    "Supports various CSV formats including common banking/export formats."
))
def load_csv_expenses(csv_path: str) -> str:
    try:
        transactions = csv_extractor.extract_transactions(csv_path)
        if not transactions:
            return json.dumps(
                {"error": "No transactions could be parsed. "
                          "The file may not be a valid CSV statement or uses an unsupported layout."}
            )
        return csv_extractor.to_json(transactions)
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to parse CSV file: {exc}"}
        )


@tool("budget_report", description=(
    "Given a JSON list of transactions (from load_expenses or load_csv_expenses), categorize the "
    "spending and produce a personal budget recommendation with concrete "
    "amounts and money-saving suggestions. Takes the transactions JSON string."
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
        tools=[load_expenses, load_csv_expenses, budget_report],
        system_prompt=(
            "You are a personal budgeting assistant. Your job is to help the "
            "user understand and improve their spending.\n\n"
            "Workflow:\n"
            "1. Use load_expenses for searchable PDF statements or "
            "load_csv_expenses for CSV statements to get the transactions.\n"
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