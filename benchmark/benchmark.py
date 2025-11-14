import json
import os
import argparse
import asyncio
from openai import AsyncOpenAI
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError
import aiofiles
from typing import Dict, List, Optional, Any
import time
from pathlib import Path


# Load API configuration from config/api.json
def load_api_config():
    """Load API configuration from config/api.json file."""
    config_path = Path(__file__).parent.parent / "config" / "api.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Normalize base_url: OpenAI client expects base_url to end with /v1/, not /v1/chat/completions
    base_url = config.get("base_url", "")
    if base_url.endswith("/chat/completions"):
        base_url = base_url.replace("/chat/completions", "")
    if not base_url.endswith("/"):
        base_url += "/"
    
    return {
        "base_url": base_url,
        "api_key": config.get("api_key", ""),
        "model": config.get("model", "gpt-4o-mini-2025-07-16"),
        "temperature": config.get("temperature", 0.1),
    }


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


async def save_jsonl_async(data: List[Dict], file_path: str):
    """Save data to a JSONL file asynchronously."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            await f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def create_classification_prompt(dataset_info: Dict) -> str:
    """
    Create a prompt for the LLM to classify whether a dataset is a benchmark
    and assign domain tags.

    Args:
        dataset_info: Dictionary containing dataset information

    Returns:
        Formatted prompt string
    """
    # Extract key information from the dataset
    dataset_id = dataset_info.get("id", "Unknown")
    author = dataset_info.get("author", "Unknown")
    description = dataset_info.get("description", "No description available")
    tags = dataset_info.get("tags", {})
    pretty_name = dataset_info.get("pretty_name", "")
    arxiv = dataset_info.get("arxiv", "")

    # Truncate description if too long
    if description and len(description) > 2000:
        description = description[:2000] + "..."

    prompt = f"""You are an expert in AI datasets and benchmarks. Your task is to analyze a dataset and determine:
    1. Whether it is a benchmark dataset (used for evaluating/testing models)
    2. Which domain(s) it belongs to

    Dataset Information:
    - ID: {dataset_id}
    - Author: {author}
    - Pretty Name: {pretty_name}
    - ArXiv: {arxiv}
    - Description: {description}
    - Tags: {json.dumps(tags, indent=2)}

    ====== ALLOWED DOMAIN TAGS (STRICT LIST) ======
    You MUST choose ONLY from these exact domain tags:
    
    1. "code" - Programming, code generation, code understanding, software engineering
    2. "math" - Mathematics, mathematical reasoning, arithmetic, algebra, geometry, calculus
    3. "science" - Natural sciences (physics, chemistry, biology), scientific reasoning
    4. "medical" - Medical diagnosis, healthcare, clinical data, biomedical research
    5. "finance" - Financial analysis, stock prediction, economic data
    6. "law" - Legal documents, law, regulations, judicial reasoning
    7. "agent" - AI agents, interaction, tool use, planning, multi-step reasoning, decision making, robotics
    8. "multi-modal" - Datasets involving multiple modalities (text, image, audio, video)
    9. "general" - General-purpose benchmarks, NLP tasks, text classification, QA, reasoning, etc.
    10. "other" - Benchmarks that don't belong to any of the above categories
    
    ====== CRITICAL RULES ======
    - You MUST use ONLY these exact strings: "code", "math", "science", "medical", "finance", "law", "agent", "multi-modal", "general", "other"
    - DO NOT create new tags like "robotics", "biology", "NLP", "computer-vision", etc.
    - Use "science" for physics/chemistry/biology topics
    - Use "agent" for robotics/planning/tool-use
    - Use "general" for NLP/text/reasoning tasks
    - Use "other" if none of the above fit well

    Instructions:
    1. Carefully analyze whether this dataset is a BENCHMARK (used for evaluation/testing) or just a regular dataset (used for training/fine-tuning).
    2. Benchmarks typically have characteristics like:
        - Used for evaluation or testing purposes
        - Contains test/validation sets with ground truth
        - Designed to measure model performance
        - Often has leaderboards or published results
        - Names often include words like "benchmark", "eval", "test", "challenge"

    3. If it IS a benchmark, assign one or more domain tags from the STRICT LIST above.
    4. Multiple domains can be assigned if the benchmark spans multiple areas.

    Response Format (JSON only):
    {{
        "is_benchmark": true/false,
        "reasoning": "Brief explanation of why this is or isn't a benchmark",
        "domains": ["code", "math", "science", "medical", "finance", "law", "agent", "multi-modal", "general", or "other" ONLY],
        "confidence": "high/medium/low"
    }}

    REMINDER: Only use the 10 exact domain tags listed above. No other values are allowed in the "domains" field."""

    return prompt


# Define valid domain tags as a constant
VALID_DOMAIN_TAGS = {
    "code",
    "math", 
    "science",
    "medical",
    "finance",
    "law",
    "agent",
    "multi-modal",
    "general",
    "other"
}


def validate_and_filter_domains(domains: List[str]) -> List[str]:
    """
    Validate and filter domain tags to only include valid ones.
    
    Args:
        domains: List of domain tags from LLM response
        
    Returns:
        Filtered list containing only valid domain tags
    """
    if not domains:
        return ["other"]  # Default to "other" if no domains provided
    
    valid_domains = [d for d in domains if d in VALID_DOMAIN_TAGS]
    
    # If all domains were invalid, default to "other"
    if not valid_domains:
        print(f"Warning: Invalid domains detected: {domains}. Defaulting to 'other'")
        return ["other"]
    
    # Log any invalid domains that were filtered out
    invalid_domains = [d for d in domains if d not in VALID_DOMAIN_TAGS]
    if invalid_domains:
        print(f"Warning: Filtered out invalid domains: {invalid_domains}")
    
    return valid_domains


def create_default_classification(dataset_id: str, error_msg: str) -> Dict[str, Any]:
    """Create a default classification result for failed requests."""
    return {
        "is_benchmark": False,
        "reasoning": f"Classification failed due to error: {error_msg}",
        "domains": [],
        "confidence": "low"
    }


async def classify_dataset_with_llm_async(
    client: AsyncOpenAI, 
    dataset_info: Dict, 
    model: str = "gpt-4o", 
    temperature: float = 0.1,
    max_retries: int = 3,
    semaphore: asyncio.Semaphore = None
) -> Optional[Dict]:
    """
    Classify a single dataset using OpenAI API asynchronously.

    Args:
        client: AsyncOpenAI client instance
        dataset_info: Dictionary containing dataset information
        model: OpenAI model to use
        temperature: Temperature for LLM generation
        max_retries: Maximum number of retry attempts
        semaphore: Semaphore for rate limiting

    Returns:
        Classification result dictionary or None if failed
    """
    dataset_id = dataset_info.get("id", "unknown")
    prompt = create_classification_prompt(dataset_info)
    
    async def _classify():
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in AI datasets and benchmarks. You MUST only use the exact domain tags provided in the instructions: code, math, science, medical, finance, law, agent, multi-modal, general, other. Always respond with valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )

                result = json.loads(response.choices[0].message.content)
                
                # Validate and filter domains
                if "domains" in result and isinstance(result["domains"], list):
                    result["domains"] = validate_and_filter_domains(result["domains"])
                
                return result

            except json.JSONDecodeError as e:
                print(f"JSON decode error for dataset {dataset_id}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                return create_default_classification(dataset_id, f"JSON decode error: {str(e)}")

            except APIConnectionError as e:
                print(f"Connection error for dataset {dataset_id} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return create_default_classification(dataset_id, f"Connection error: {str(e)}")

            except APITimeoutError as e:
                print(f"Timeout error for dataset {dataset_id} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return create_default_classification(dataset_id, f"Timeout error: {str(e)}")

            except RateLimitError as e:
                print(f"Rate limit error for dataset {dataset_id} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                return create_default_classification(dataset_id, f"Rate limit error: {str(e)}")

            except APIError as e:
                print(f"API error for dataset {dataset_id} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (2 ** attempt))
                    continue
                return create_default_classification(dataset_id, f"API error: {str(e)}")

            except Exception as e:
                print(f"Unexpected error for dataset {dataset_id} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                return create_default_classification(dataset_id, f"Unexpected error: {str(e)}")

        return create_default_classification(dataset_id, "Maximum retries exceeded")
    
    if semaphore:
        async with semaphore:
            return await _classify()
    else:
        return await _classify()


async def process_single_dataset(
    client: AsyncOpenAI,
    dataset: Dict[str, Any],
    model: str,
    temperature: float,
    max_retries: int,
    semaphore: asyncio.Semaphore,
    idx: int,
    total: int
) -> Optional[Dict[str, Any]]:
    """
    Process a single dataset and return it if it's a benchmark.
    
    Args:
        client: AsyncOpenAI client
        dataset: Dataset information dictionary
        model: Model name to use
        temperature: Temperature for LLM generation
        max_retries: Maximum retry attempts
        semaphore: Semaphore for rate limiting
        idx: Current index (for logging)
        total: Total number of datasets (for logging)
    
    Returns:
        Dataset with classification info if it's a benchmark, None otherwise
    """
    dataset_id = dataset.get("id", f"unknown_{idx}")
    print(f"Processing {idx}/{total}: id={dataset_id}")
    
    try:
        classification = await classify_dataset_with_llm_async(
            client, dataset, model, temperature, max_retries, semaphore
        )
        
        if classification and classification.get("is_benchmark", False):
            # Keep all original information and add classification results
            dataset["benchmark_reasoning"] = classification.get("reasoning", "")
            dataset["domains"] = classification.get("domains", [])
            dataset["classification_confidence"] = classification.get("confidence", "unknown")
            return dataset
        
        return None
        
    except Exception as e:
        print(f"Error processing dataset {dataset_id}: {e}")
        return None


async def process_datasets_async(
    input_file: str,
    output_file: str,
    base_url: str,
    api_key: str,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_concurrent: int = 100,
    start_index: int = 0,
    limit: Optional[int] = None,
    max_retries: int = 3,
):
    """
    Process all datasets from input JSONL file and save benchmark classifications asynchronously.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        base_url: Base URL for OpenAI API
        api_key: OpenAI API key
        model: OpenAI model to use
        temperature: Temperature for LLM generation
        max_concurrent: Maximum concurrent requests
        start_index: Start processing from this index
        limit: Maximum number of datasets to process
        max_retries: Maximum retry attempts per dataset
    """
    # Initialize AsyncOpenAI client
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # Load input data
    print(f"Loading data from {input_file}...")
    all_datasets = load_jsonl(input_file)
    print(f"Loaded {len(all_datasets)} datasets")

    # Apply limit if specified
    if limit:
        end_index = min(start_index + limit, len(all_datasets))
        datasets_to_process = all_datasets[start_index:end_index]
    else:
        datasets_to_process = all_datasets[start_index:]

    print(f"Processing {len(datasets_to_process)} datasets (from index {start_index})")
    print(f"Max concurrent requests: {max_concurrent}, Max retries: {max_retries}, Temperature: {temperature}")

    # Check if output file exists and load existing results
    existing_ids = set()
    if os.path.exists(output_file):
        print(f"Loading existing results from {output_file}...")
        existing_results = load_jsonl(output_file)
        existing_ids = {item["id"] for item in existing_results}
        print(f"Found {len(existing_results)} existing benchmarks")

    # Filter out already processed datasets
    datasets_to_process = [d for d in datasets_to_process if d["id"] not in existing_ids]
    print(f"After filtering, {len(datasets_to_process)} datasets need processing")

    if not datasets_to_process:
        print("No new datasets to process!")
        return

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all datasets
    tasks = [
        process_single_dataset(
            client, dataset, model, temperature, max_retries, semaphore, idx + 1, len(datasets_to_process)
        )
        for idx, dataset in enumerate(datasets_to_process)
    ]

    # Execute all tasks concurrently
    print("Starting concurrent processing...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None results and exceptions
    benchmarks = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i+1} failed with exception: {result}")
        elif result is not None:
            benchmarks.append(result)

    # Load existing results and append new benchmarks
    all_benchmarks = []
    if os.path.exists(output_file):
        all_benchmarks = load_jsonl(output_file)
    
    all_benchmarks.extend(benchmarks)

    # Save all benchmarks
    await save_jsonl_async(all_benchmarks, output_file)

    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"New benchmarks identified: {len(benchmarks)}")
    print(f"Total benchmarks in output file: {len(all_benchmarks)}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    # Print domain statistics
    if all_benchmarks:
        domain_counts = {}
        for result in all_benchmarks:
            for domain in result.get("domains", []):
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        print("\nDomain Distribution:")
        for domain, count in sorted(
            domain_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {domain}: {count}")


async def main():
    parser = argparse.ArgumentParser(
        description="Classify datasets as benchmarks and assign domain tags using LLM with concurrent processing."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the input JSONL file containing dataset information.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to the output JSONL file for benchmark results.",
    )
    # Load default values from config file
    api_config = load_api_config()
    
    parser.add_argument(
        "--base_url",
        "-u",
        type=str,
        default=api_config["base_url"],
        help=f"Base URL for OpenAI API (default: from config/api.json)",
    )
    parser.add_argument(
        "--api_key",
        "-k",
        type=str,
        default=api_config["api_key"],
        help="OpenAI API key (default: from config/api.json)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=api_config["model"],
        help=f"OpenAI model to use (default: from config/api.json)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=api_config["temperature"],
        help=f"Temperature for LLM generation (default: from config/api.json)",
    )
    parser.add_argument(
        "--max_concurrent",
        "-c",
        type=int,
        default=100,
        help="Maximum number of concurrent requests (default: 100)",
    )
    parser.add_argument(
        "--start_index",
        "-s",
        type=int,
        default=0,
        help="Start processing from this index (default: 0)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of datasets to process (default: all)",
    )
    parser.add_argument(
        "--max_retries",
        "-r",
        type=int,
        default=3,
        help="Maximum number of retry attempts per dataset (default: 3)",
    )

    args = parser.parse_args()

    # Process datasets asynchronously
    await process_datasets_async(
        input_file=args.input,
        output_file=args.output,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        start_index=args.start_index,
        limit=args.limit,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    asyncio.run(main())
