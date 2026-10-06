"""Budget Agent CLI entry point.

Parses statement files locally before sending normalized transactions to the
LLM for budgeting analysis.

Usage:
    python budget.py path/to/expenses.pdf
    python budget.py path/to/expenses.csv
"""
import argparse
import sys
from pathlib import Path

if __package__:
    from .agent import analyze_file
else:
    from agent import analyze_file


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

    print(f"Analyzing your expenses from {args.file}...\n")
    try:
        response = analyze_file(file_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    # Print the final assistant answer (last AI message without tool calls).
    last = response["messages"][-1]
    if getattr(last, "content", None):
        print(last.content)
    else:
        print(response)


if __name__ == "__main__":
    sys.exit(main())