import json
import argparse


def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(data, file_path):
    """Append data to an existing JSONL file."""
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract id and author from first JSONL file and append to second JSONL file."
    )
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        required=True,
        help="Path to the input JSONL file (source file)."
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        required=True,
        help="Path to the output JSONL file (target file to append to)."
    )
    args = parser.parse_args()
    
    # Load data from input file
    input_data = load_jsonl(args.input)
    print(f"Loaded {len(input_data)} records from {args.input}")
    
    # Extract id and author from each record
    extracted_data = []
    for item in input_data:
        if 'id' in item and 'author' in item:
            extracted_data.append({
                'id': item['id'],
                'author': item['author']
            })
        else:
            print(f"WARNING: Record missing 'id' or 'author' field: {item}")
    
    # Append to output file
    append_jsonl(extracted_data, args.output)
    print(f"Appended {len(extracted_data)} records to {args.output}")
