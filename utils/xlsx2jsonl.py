#!/usr/bin/env python3
"""
Convert Excel file to JSONL format
"""
import pandas as pd
import json
import sys
from pathlib import Path


def xlsx_to_jsonl(xlsx_file, jsonl_file=None):
    """
    Convert an Excel file to JSONL format
    
    Args:
        xlsx_file: Path to the input Excel file
        jsonl_file: Path to the output JSONL file (optional, defaults to same name with .jsonl extension)
    """
    # Read Excel file
    print(f"Reading Excel file: {xlsx_file}")
    df = pd.read_excel(xlsx_file)
    
    # If no output file specified, use the same name with .jsonl extension
    if jsonl_file is None:
        jsonl_file = Path(xlsx_file).with_suffix('.jsonl')
    
    print(f"Converting {len(df)} rows to JSONL format...")
    
    # Convert to JSONL
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            # Convert row to dictionary and handle NaN values
            row_dict = row.to_dict()
            # Replace NaN with None for proper JSON serialization
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
            # Write as JSON line
            f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
            
            # Print progress every 1000 rows
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} rows...")
    
    print(f"✓ Successfully converted to: {jsonl_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python xlsx_to_jsonl.py <input.xlsx> [output.jsonl]")
        sys.exit(1)
    
    xlsx_file = sys.argv[1]
    jsonl_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    xlsx_to_jsonl(xlsx_file, jsonl_file)

