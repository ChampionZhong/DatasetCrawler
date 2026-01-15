import pickle
import json
import os
from datetime import datetime, timedelta  
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


def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + '\n')


def append_jsonl(data, file_path):
    """Append data to an existing JSONL file."""
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + '\n')


def load_object_from_pickle(filepath):
    """
    Load a Python object from a pickle file.
    
    Args:
        filepath (str): Path to the pickle file to load
        
    Returns:
        The Python object stored in the pickle file
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        Exception: If there's an error during unpickling
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist")
    
    try:
        print(f"Loading object from {filepath}...")
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print("Object loaded successfully")
        return obj
    except Exception as e:
        print(f"Error loading object: {e}")
        raise


def deduplicate_latest_info(latest_info: list, main_info: list) -> list:
    """
    Deduplicates a list of new dataset info against a main list based on the 'id' field.

    This function is efficient for large lists as it first creates a set of existing IDs 
    for quick lookups.

    Args:
        latest_info (list): A list of dictionaries, where each dictionary is a new dataset's info.
        main_info (list): A list of dictionaries representing the main data storage.

    Returns:
        list: A new list containing only the items from latest_info that are not
                present in main_info (based on 'id').
    """
    # Create a set of all existing IDs from the main storage for fast O(1) lookups
    existing_ids = {item['id'] for item in main_info}
    
    # Use a list comprehension to build a new list of items
    # whose IDs are not in the set of existing IDs.
    new_unique_items = [
        item for item in latest_info if item['id'] not in existing_ids
    ]
    
    return new_unique_items

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process dataset metadata and extract information."
    )
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        required=True,
        help="Path to the input pickle file."
    )
    parser.add_argument(
        '--output', 
        '-o',
        type=str,
        required=True,
        help="Path to the output JSONL file."
    )
    parser.add_argument(
        '--main_info',
        '-m',
        default=None,
        type=str,
        help="Path to the main info JSONL file."
    )
    parser.add_argument(
        '--start_date',
        default=str((datetime.today() - timedelta(days=10)).date()),
        type=str,
        help="Start date to filter depends on date. (格式: YYYY-MM-DD)"
    )
    parser.add_argument(
        '--end_date',
        default=str((datetime.today() - timedelta(days=3)).date()),
        type=str,
        help="End date to filter depends on date. (格式: YYYY-MM-DD)"
    )
    parser.add_argument(
        '--history_info',
        default="./data/weekly/base/base.jsonl",
        type=str,
        help="Path to the history info JSONL file."
    )
    args = parser.parse_args()
    
    # Load metadata from pickle file
    metadata = load_object_from_pickle(args.input)
    history_info = load_jsonl(args.history_info)
    
    # Build a set of (id, author) tuples from history_info for fast lookup
    history_set = set()
    for item in history_info:
        if 'id' in item and 'author' in item:
            history_set.add((item['id'], item['author']))
    
    # Debug: Check if metadata is loaded correctly
    if metadata is None:
        print("WARNING: metadata is None!")
    elif isinstance(metadata, list):
        print(f"Loaded {len(metadata)} dataset records from pickle file")
    else:
        print(f"WARNING: metadata is not a list, type: {type(metadata)}")
    
    cnt = 0
    all_extracted_info = []
    # domain_list = ["math","code","reasoning"] # check title, tags, description
    domain_list = ["code", "math"]
    size_list = ['1K<n<10K','10K<n<100K','100K<n<1M','1M<n<10M']
    language_list = ['en','code']
    collect_date = datetime.today().date()
    # NOTE(zzp): Set the date range for filtering datasets
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    print(f"start_date: {start_date}")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    print(f"end_date: {end_date}")
    
    # Ensure start_date >= end_date for proper range checking
    if start_date < end_date:
        print(f"WARNING: start_date ({start_date}) < end_date ({end_date}), swapping them")
        start_date, end_date = end_date, start_date
        print(f"After swap - start_date: {start_date}, end_date: {end_date}")

    processed_count = 0
    for info in metadata:
        processed_count += 1
        if processed_count % 10000 == 0:
            print(f"Processed {processed_count} datasets...")
        if info.private or info.disabled:
            continue

        # check tag info. Focus on modality, size_categories, task_categories, arxiv
        if info.tags:
            # Transform tags list into a dictionary
            tag_dict = {}
            for tag in info.tags:
                # Check if the tag has a category (contains a colon)
                if ':' in tag:
                    # Split the tag into category and value
                    category, value = tag.split(':', 1)
                    category = category.strip()
                    value = value.strip()
                    
                    # Handle duplicate keys by converting to list
                    if category in tag_dict:
                        # If the value is already a list, append to it
                        if isinstance(tag_dict[category], list):
                            tag_dict[category].append(value)
                        # Otherwise, convert to list with both values
                        else:
                            tag_dict[category] = [tag_dict[category], value]
                    else:
                        tag_dict[category] = value
                else:
                    # For tags without a category, use the tag itself as both key and value
                    tag_dict[tag] = tag
            
            collect = False

            # create a boolean for each condition
            modality_text_only = False
            publication = False # arxiv, doi
            benchmark = False
            language = False
            card_data = False
            downloads = False
            target_domain = False
            time_range = False
            size_check = False
            history_not_in = False
            early_than_start_date = False
            late_than_end_date = False

            # Check if current dataset's id and author are in history
            current_id = info.id.split('/')[-1]  # Extract dataset name from full path
            current_author = info.author
            if (current_id, current_author) not in history_set:
                history_not_in = True

            if "benchmark" in tag_dict:
                benchmark = True

            language = True
            if "language" in tag_dict:
                if tag_dict['language'] is not None:
                    if isinstance(tag_dict['language'], list):
                        for lan in tag_dict['language']:
                            if lan not in language_list:
                                language = False
                                break
                    else:
                        if tag_dict['language'] not in language_list:
                            language = False
            
            if info.cardData:
                card_data = True
            
            # Check if created_at is within the date range (end_date <= created_at <= start_date)
            # Note: start_date is typically today, end_date is typically N days ago
            if info.created_at and end_date <= info.created_at.date() <= start_date:
                time_range = True
            
            if info.created_at and info.created_at.date() <= start_date:
                early_than_start_date = True
            
            if info.created_at and info.created_at.date() >= end_date:
                late_than_end_date = True

            if info.downloads>10:
                downloads = True
            
            if "size_categories" in tag_dict:
                if tag_dict['size_categories'] in size_list:
                    size_check = True
            
            # check which domain element is in title and tags, store the domain info
            all_domains = []
            for domain in domain_list:
                if domain in info.id.lower():
                    all_domains.append(domain)
                
                # Check if domain is a substring of any tag key
                for tag in tag_dict.keys():
                    if domain in tag.lower() and domain not in all_domains:
                        all_domains.append(domain)
            
            if len(all_domains) > 0:
                target_domain = True

            # collect = modality_text_only and likes and card_data and downloads and target_domain
            # collect = modality_text_only and target_domain and benchmark
            # collect = modality_text_only and time_range and downloads and size_check and card_data and language
            collect = downloads and language and card_data and size_check and history_not_in and early_than_start_date

            description = None
            if hasattr(info, "description") and info.description is not None:
                description = info.description
            elif hasattr(info, "cardData") and info.cardData is not None:
                description = info.cardData.get("description", None)
            
            if collect:
                # Convert datetime objects to timezone unaware
                created_at = info.created_at.date() if info.created_at else None
                last_modified = info.last_modified.date() if info.last_modified else None

                dataset_meta = {
                    "id": info.id.split('/')[-1],
                    "author": info.author,
                    "downloads": info.downloads,
                    "likes": info.likes,
                    "created_at": created_at,   
                    "last_modified": last_modified,
                    "collect_date": collect_date,
                    "description": description,
                    "language": tag_dict.get("language"),
                    "pretty_name": info.cardData.get("pretty_name"),
                    "size_categories": tag_dict.get("size_categories"),
                    "arxiv": tag_dict.get("arxiv"),
                    "tags": tag_dict,
                }

                all_extracted_info.append(dataset_meta)

    # Deduplicate the extracted dataset info against the main list
    if args.main_info:
        main_info = load_jsonl(args.main_info)
        deduplicated_info = deduplicate_latest_info(all_extracted_info, main_info)
    else:
        deduplicated_info = all_extracted_info

    # Save the extracted dataset info to a JSONL file, converting dates to strings
    save_jsonl(deduplicated_info, args.output)
    print(f"Saved {len(deduplicated_info)} dataset records to {args.output}")