#!/usr/bin/env python3
"""
Convert CSV file to JSONL format
"""
import pandas as pd
import json
import sys
from pathlib import Path


def csv_to_jsonl(csv_file, jsonl_file):
    """
    Convert a CSV file to JSONL format
    
    Args:
        csv_file: Path to the input CSV file
        jsonl_file: Path to the output JSONL file
    """
    # Read CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
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
    if len(sys.argv) < 3:
        print("Usage: python csv2jsonl.py <input.csv> <output.jsonl>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    jsonl_file = sys.argv[2]
    
    csv_to_jsonl(csv_file, jsonl_file)
