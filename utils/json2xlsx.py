import pandas as pd
import argparse
import json
import re

def clean_text_for_excel(text):
    """Clean text to remove illegal characters for Excel worksheets."""
    if not isinstance(text, str):
        return text
    
    # Remove control characters except for common whitespace
    # Keep spaces, tabs, and newlines but convert them to spaces
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Replace tabs and newlines with spaces
    text = re.sub(r'[\t\n\r]+', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

def clean_data_for_excel(data):
    """Recursively clean all string values in the data structure."""
    if isinstance(data, dict):
        return {key: clean_data_for_excel(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_excel(item) for item in data]
    elif isinstance(data, str):
        return clean_text_for_excel(data)
    else:
        return data
def jsonl_to_excel(input_file, output_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_data = json.loads(line)
                # Clean the data to remove illegal characters for Excel
                cleaned_data = clean_data_for_excel(json_data)
                data.append(cleaned_data)
            except json.JSONDecodeError:
                print(f"Warning: skip invalid line: {line}")
    
    df = pd.DataFrame(data)
    
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Successfully saved data to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a JSONL file to an Excel file."
    )
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        required=True,
        help="Path to the excel file."
    )
    args = parser.parse_args()

    jsonl_to_excel(args.input, args.output)