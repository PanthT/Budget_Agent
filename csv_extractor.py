"""Parse expense transactions from a bank/credit-card statement CSV.

Uses the csv module to extract transaction records from CSV files.
Returns clean transaction records: {date, description, category, amount}.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _normalize_date(date_str: str) -> str:
    """Normalize date string to a standard format."""
    if not date_str:
        return ""
    
    date_str = date_str.strip()
    
    if ' ' in date_str:
        date_str = date_str.split()[0]
    
    date_str = date_str.replace('/', '-').replace('.', '-')
    
    return date_str


def _normalize_amount(amount_str: str) -> float:
    """Normalize amount string to float, handling various formats."""
    if not amount_str:
        return 0.0
    
    amount_str = amount_str.strip()
    
    for symbol in ['$', '€', '£', '¥', 'USD', 'EUR', 'GBP', 'JPY']:
        amount_str = amount_str.replace(symbol, '')
    
    is_negative = False
    if amount_str.startswith('(') and amount_str.endswith(')'):
        is_negative = True
        amount_str = amount_str[1:-1]
    
    if amount_str.startswith('-'):
        is_negative = not is_negative
        amount_str = amount_str[1:]
    
    amount_str = amount_str.replace(',', '').replace(' ', '')
    
    try:
        value = float(amount_str)
    except ValueError:
        import re
        numbers = re.findall(r'-?\d+(?:\.\d+)?', amount_str)
        if numbers:
            value = float(numbers[0])
        else:
            return 0.0
    
    return -value if is_negative else value


def _infer_category(description: str) -> str:
    """Infer category from transaction description."""
    description = description.lower()
    
    if any(keyword in description for keyword in [
        'restaurant', 'food', 'groceries', 'coffee', 'lunch', 'dinner', 
        'breakfast', 'snack', 'supermarket', 'foodmart', 'starbucks',
        'mcdonald', 'burger king', 'subway', 'chipotle'
    ]):
        return 'Dining'
    
    elif any(keyword in description for keyword in [
        'gas', 'fuel', 'parking', 'transit', 'uber', 'lyft', 'taxi',
        'bus', 'train', 'metro', 'gas station', 'shell', 'exxon'
    ]):
        return 'Transportation'
    
    elif any(keyword in description for keyword in [
        'rent', 'mortgage', 'maintenance', 'hoa', 'property tax',
        ' landlord', 'apartment', 'housing'
    ]):
        return 'Housing'
    
    elif any(keyword in description for keyword in [
        'electricity', 'water', 'gas', 'internet', 'wifi', 'phone',
        'mobile', 'cable', 'utility', 'power'
    ]):
        return 'Utilities'
    
    elif any(keyword in description for keyword in [
        'amazon', 'walmart', 'target', 'best buy', 'apple store',
        'shopping', 'retail', 'online', 'store', 'mall'
    ]):
        return 'Shopping'
    
    elif any(keyword in description for keyword in [
        'movie', 'cinema', 'netflix', 'spotify', 'apple music',
        'entertainment', 'concert', 'theater', 'game', 'hobby'
    ]):
        return 'Entertainment'
    
    elif any(keyword in description for keyword in [
        'hospital', 'medical', 'pharmacy', 'doctor', 'dentist',
        'health', 'prescription', 'insurance'
    ]):
        return 'Health'
    
    elif any(keyword in description for keyword in [
        'hotel', 'flight', 'airline', 'hotel', 'resort', 'vacation',
        'travel', 'tour'
    ]):
        return 'Travel'
    
    elif any(keyword in description for keyword in [
        'deposit', 'paycheck', 'salary', 'income', 'refund', 'transfer in',
        'direct deposit', 'wage'
    ]):
        return 'Income'
    
    else:
        return 'Other'


def extract_transactions(csv_path: str | Path) -> list[dict[str, Any]]:
    """Return a list of transaction dicts parsed from the statement CSV."""
    transactions: list[dict[str, Any]] = []
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read first 1024 bytes to detect CSV structure
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        sample = f.read(1024)
        f.seek(0)
        
        try:
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
        except:
            dialect = None
            has_header = True
    
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f) if has_header else csv.reader(f, dialect or 'excel')
        
        if has_header:
            fieldnames = reader.fieldnames or []
        else:
            fieldnames = ['date', 'description', 'amount', 'category']
        
        for row in reader:
            if has_header:
                transaction = _parse_csv_row(row, fieldnames)
            else:
                transaction = _parse_positional_row(row)
            
            if transaction:
                transactions.append(transaction)
    
    return transactions


def _parse_csv_row(row: dict, fieldnames: list[str]) -> dict[str, Any] | None:
    """Parse a CSV row with header to extract transaction."""
    transaction = {
        'date': '',
        'description': '',
        'category': 'uncategorized',
        'amount': 0.0
    }
    
    date_keys = ['date', 'transaction_date', 'posted_date', 'date posted']
    desc_keys = ['description', 'desc', 'merchant', 'payee', 'transaction_details']
    amount_keys = ['amount', 'transaction_amount', 'debit', 'credit', 'amount_credited']
    
    for i, fieldname in enumerate(fieldnames):
        if isinstance(row, dict):
            value = row.get(fieldname, '')
        else:
            value = row[i] if i < len(row) else ''
        
        fieldname_lower = fieldname.lower()
        
        if any(key in fieldname_lower for key in date_keys):
            transaction['date'] = _normalize_date(value)
        
        elif any(key in fieldname_lower for key in desc_keys):
            transaction['description'] = str(value).strip()
        
        elif any(key in fieldname_lower for key in amount_keys):
            if 'debit' in fieldname_lower or 'credit' in fieldname_lower:
                transaction['amount'] = _normalize_amount(value)
            else:
                transaction['amount'] = _normalize_amount(value)
    
    if not transaction['date'] or not transaction['description']:
        try:
            if len(row) >= 3:
                if not transaction['date']:
                    transaction['date'] = _normalize_date(str(row[0] if not isinstance(row, dict) else row.get(fieldnames[0] if fieldnames else '')))
                if not transaction['description']:
                    transaction['description'] = str(row[1] if not isinstance(row, dict) else row.get(fieldnames[1] if len(fieldnames) > 1 else '')).strip()
                if not transaction['amount']:
                    transaction['amount'] = _normalize_amount(str(row[2] if not isinstance(row, dict) else row.get(fieldnames[2] if len(fieldnames) > 2 else '')))
        except (IndexError, AttributeError):
            pass
    
    if transaction['date'] and transaction['description']:
        transaction['category'] = _infer_category(transaction['description'])
        return transaction
    
    return None


def _parse_positional_row(row: list[str]) -> dict[str, Any] | None:
    """Parse a CSV row without headers, using positional arguments."""
    if len(row) < 2:
        return None
    
    date_str = _normalize_date(row[0])
    amount_str = row[2] if len(row) > 2 else (row[1] if len(row) > 1 else '')
    
    if len(row) >= 3:
        description = row[1]
        amount_str = row[2]
    elif len(row) == 2:
        if _normalize_amount(row[1]) != 0:
            description = ''
            amount_str = row[1]
        else:
            description = row[1]
            amount_str = '0'
    else:
        return None
    
    if not date_str:
        return None
    
    amount = _normalize_amount(amount_str)
    
    transaction = {
        'date': date_str,
        'description': description.strip(),
        'category': _infer_category(description),
        'amount': amount
    }
    
    return transaction


def to_json(transactions: list[dict]) -> str:
    """Convert transactions list to JSON string."""
    return json.dumps(transactions, indent=2)