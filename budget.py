"""Budget Agent CLI entry point.

Usage:
    python budget.py path/to/expenses.pdf
    python budget.py path/to/expenses.csv
"""
import argparse
import sys
from pathlib import Path

if __package__:
    from .agent import build_agent
else:
    from agent import build_agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze an expenses PDF or CSV and get a personalized budget plan."
    )
    parser.add_argument("file", help="Path to the expense/bank-statement PDF or CSV file.")
    args = parser.parse_args()

    file_path = Path(args.file)
    file_type = file_path.suffix.lower()
    if file_type not in {".pdf", ".csv"}:
        parser.error("file must have a .pdf or .csv extension")

    agent = build_agent()
    print(f"Analyzing your expenses from {args.file}...\n")

    if file_type == ".csv":
        tool_name = "load_csv_expenses"
        display_type = "CSV"
    else:
        tool_name = "load_expenses"
        display_type = "PDF"
    
    response = agent.invoke({"messages": [
        {"role": "user", "content": (
            f"Use the {tool_name} tool to load my expenses from the "
            f"{display_type} at '{args.file}', then give me "
            f"a full budget report and saving recommendations based on them."
        )}
    ]})

    # Print the final assistant answer (last AI message without tool calls).
    last = response["messages"][-1]
    if getattr(last, "content", None):
        print(last.content)
    else:
        print(response)


if __name__ == "__main__":
    sys.exit(main())