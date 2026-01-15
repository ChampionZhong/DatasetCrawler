from huggingface_hub import list_datasets
import os
import pickle
from datetime import datetime
import argparse


def save_metadata_to_pickle(metadata, filename=None):
    """
    Save metadata to a pickle file.
    
    Args:
        metadata: The metadata list to save
        filename: The filename to save, if None, will use a timestamped default name
    """
    if filename is None:
        # Create output directory (if not exists)
        os.makedirs('output', exist_ok=True)
        # Use timestamp to define file name 
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/hf_datasets_metadata_{timestamp}.pkl"
    
    # save meta data to pickle file
    print(f"Saving metadata to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved successfully to {filename}")
    
    # return filepath
    return filename


def fetch_all_dataset_metadata(limit=None):
    print("Fetching list of all datasets from Hugging Face Hub...")
    # Get latest 20000 datasets
    if limit is None:
        all_datasets_info = list(list_datasets(full=True, sort="created_at", direction=-1))
    else:
        all_datasets_info = list(list_datasets(full=True, sort="created_at", direction=-1, limit=limit))
    # all_datasets_info = list(list_datasets(full=True, sort="created_at", direction=-1))

    return all_datasets_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch all dataset metadata from Hugging Face Hub and save to a pickle file."
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        required=True,
        help="Path to the output pickle file."
    )
    parser.add_argument(
        '--limit',
        '-l',
        type=int,
        # default=20000,
        help="Limit the number of datasets to fetch."
    )
    args = parser.parse_args()
    try:
        limit = args.limit
        metadata_list = fetch_all_dataset_metadata(limit)
    except:
        limit = None
        metadata_list = fetch_all_dataset_metadata()
    save_metadata_to_pickle(metadata_list, args.output)


