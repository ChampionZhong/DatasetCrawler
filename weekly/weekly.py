import re
import json
import asyncio
from openai import AsyncOpenAI
import argparse
from typing import List, Dict, Any
import aiofiles
import time
from pathlib import Path
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError


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
        "model": config.get("model", "gpt-5.2"),
    }


api_config = load_api_config()
client = AsyncOpenAI(
    base_url=api_config["base_url"],
    api_key=api_config["api_key"],
)

system_prompt = "You are an expert AI assistant specializing in classifying Hugging Face datasets. Your task is to analyze the provided metadata of text datasets and categorize them into one of the predefined categories."

prompt = """
# Task:
Based on the JSON object containing the dataset metadata, classify the dataset into **one or more** of the categories listed below. Your classification must be based *only* on the information available in the metadata. A single dataset can be assigned multiple categories if its content spans multiple domains (e.g., a dataset containing mathematical problems described in Python code could be classified as both 'Math Word Problems' and 'Programming/Code').

# Classification Categories:
You MUST use one or more of the following categories for each dataset:
1.  **Math Word Problems**: Datasets containing mathematical problems, formulas, proofs, or requiring mathematical reasoning. Keywords: math, algebra, geometry, calculus, problem-solving, MWP.
2.  **Programming/Code**: Datasets containing source code, code generation tasks, code completion, bug fixing, or programming-related text. Keywords: code, programming, software, python, java, algorithm, commit, bug.
3.  **Science**: Datasets from scientific domains like physics, chemistry, biology, astronomy, or general scientific literature/papers. Excludes medicine. Keywords: science, biology, physics, chemistry, arxiv, paper, research, experiment.
4.  **Medical/Clinical Text**: Datasets related to healthcare, medicine, clinical trials, patient records, medical literature, or biology in a medical context. Keywords: medical, clinical, health, patient, disease, PubMed, doctor, hospital.
5.  **Legal Text Analysis**: Datasets containing legal documents, court cases, laws, regulations, or legal arguments. Keywords: legal, law, court, case, statute, regulation, agreement.
6.  **Finance/Economics**: Datasets related to finance, economics, stock markets, financial reports, or economic analysis. Keywords: finance, economic, stock, market, trade, company, report.
7.  **Other**: Use this category *only* if the dataset does not clearly fit into any of the other six categories. If a dataset fits multiple categories (e.g., Math and Code), you MUST assign all relevant categories instead of 'Other'. This category is for datasets with clearly non-overlapping content like general knowledge, conversations, literature, news, etc.

# Decision Logic:
To make your decision, analyze the metadata fields in the following order of importance:
1.  **`id` and `pretty_name`**: These often contain direct keywords (e.g., "code", "math", "legal").
2.  **`description`**: This is the most valuable field. Read it carefully for the dataset's purpose, source, and content.
3.  **`tags`**: Look for custom tags that might indicate the domain.
4.  **`arxiv`**: If an arXiv ID is present, it strongly suggests a "Science" or "Math" category.
5.  **Multi-Category Logic**: If the metadata suggests clear relevance to **more than one category**, you MUST include all applicable categories in your output.
6.  **Insufficient Information**: If the information is truly insufficient to place the dataset into any of categories 1-6, classify it as '7. Other'.

# Output Format:
Your output MUST be a single, valid JSON object for each input dataset. The object MUST contain the following three fields:
- `id`: The original `id` of the dataset (string).
- `classifications`: A JSON list `[]` containing **one or more** classification objects. Each object in this list MUST contain:
    - `category_id`: The integer ID of the category (1-7).
    - `category_name`: The name of the category (string).
- `reasoning`: A single, brief explanation for your classification decisions. If multiple categories are assigned, this reasoning MUST justify all of them, citing the evidence from the metadata.

# Example:
## Input:
{
    "id": "math-problems-for-python-coders",
    "author": "SciCoder",
    "description": "This dataset contains a collection of algebra and geometry problems intended to be solved programmatically. Each problem includes a natural language description and a corresponding Python solution stub.",
    "tags": ["modality:text", "task:question-answering", "language:en"]
}
## Output in a JSONL format:
{
    "id": "math-problems-for-python-coders",
    "classifications": [
        {"category_id": 1, "category_name": "Math Word Problems"},
        {"category_id": 2, "category_name": "Programming/Code"}
    ],
    "reasoning": "The 'id' contains 'math' and 'coders', and the 'description' explicitly mentions 'algebra and geometry problems' to be solved with 'Python solution stubs', justifying both the Math and Code categories."
}
# Start Classification:
Now, please classify the following dataset metadata based on all the rules and the format specified above.
{info}
"""


def create_default_classification(dataset_id: str, error_msg: str) -> Dict[str, Any]:
    """Create a default classification result for failed requests."""
    return {
        "id": dataset_id,
        "classifications": [{"category_id": 7, "category_name": "Other"}],
        "reasoning": f"Classification failed due to error: {error_msg}",
    }


def validate_classifications_format(classifications: List[Dict[str, Any]]) -> bool:
    """
    Validate that the classifications list contains valid classification objects.

    Args:
        classifications: List of classification dictionaries

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(classifications, list):
        return False

    for classification in classifications:
        if not isinstance(classification, dict):
            return False
        if "category_id" not in classification or "category_name" not in classification:
            return False
        if not isinstance(classification["category_id"], int):
            return False
        if not isinstance(classification["category_name"], str):
            return False

    return True


# Start of Selection
def parse_output(raw_output: str, dataset_id: str = "unknown") -> Dict[str, Any]:
    """
    Parse the raw output from LLM with enhanced error handling.

    Args:
        raw_output: Raw output from the LLM
        dataset_id: Dataset ID for error reporting

    Returns:
        Parsed classification result

    Raises:
        ValueError: If parsing fails completely
    """
    try:
        # Step 1: Remove Markdown code block
        code_block_pattern = r"```(?:json)?(.*?)```"
        match = re.search(code_block_pattern, raw_output, re.DOTALL)
        content = match.group(1).strip() if match else raw_output.strip()

        # Step 2: Handle empty or whitespace-only content
        if not content:
            raise ValueError("Empty content after removing markdown blocks")

        # Step 3: Try to parse JSON
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to fix common JSON format issues
            content = content.replace(
                "'", '"'
            )  # Replace single quotes with double quotes
            content = re.sub(r",\s*}", "}", content)  # Remove trailing commas
            content = re.sub(r",\s*]", "]", content)  # Remove trailing commas in arrays

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON content even after fixes: {e}")

        # Step 4: Validate and normalize the parsed output
        if isinstance(parsed, dict):
            # Check if it has the required fields
            if "id" in parsed and "classifications" in parsed and "reasoning" in parsed:
                # Validate the classifications format
                if not validate_classifications_format(parsed["classifications"]):
                    raise ValueError(
                        f"Invalid classifications format: {parsed['classifications']}"
                    )
                return parsed
            else:
                raise ValueError(f"Missing required fields in parsed output: {parsed}")

        # If it's a single-element list containing an object, return that object
        if (
            isinstance(parsed, list)
            and len(parsed) == 1
            and isinstance(parsed[0], dict)
        ):
            result = parsed[0]
            if "id" in result and "classifications" in result and "reasoning" in result:
                # Validate the classifications format
                if not validate_classifications_format(result["classifications"]):
                    raise ValueError(
                        f"Invalid classifications format: {result['classifications']}"
                    )
                return result
            else:
                raise ValueError(f"Missing required fields in parsed output: {result}")

        # If it's a list with multiple elements, find the one matching our dataset
        if isinstance(parsed, list) and len(parsed) > 1:
            for item in parsed:
                if isinstance(item, dict) and item.get("id") == dataset_id:
                    if "classifications" in item and "reasoning" in item:
                        # Validate the classifications format
                        if not validate_classifications_format(item["classifications"]):
                            raise ValueError(
                                f"Invalid classifications format: {item['classifications']}"
                            )
                        return item
            # If no match found, take the first valid item
            for item in parsed:
                if (
                    isinstance(item, dict)
                    and "id" in item
                    and "classifications" in item
                    and "reasoning" in item
                ):
                    # Validate the classifications format
                    if not validate_classifications_format(item["classifications"]):
                        raise ValueError(
                            f"Invalid classifications format: {item['classifications']}"
                        )
                    return item

        raise ValueError(f"Parsed output format is invalid: {type(parsed)}")

    except Exception as e:
        raise ValueError(f"Failed to parse output for dataset {dataset_id}: {str(e)}")


async def llm_classification_async(
    info: Dict[str, Any],
    prompt: str,
    temperature: float = 0.7,
    semaphore: asyncio.Semaphore = None,
    max_retries: int = 3,
):
    """
    Async version of the classification function with robust error handling.

    Args:
        info: a dictionary containing the dataset metadata.
        prompt: the prompt template
        temperature: temperature for the LLM
        semaphore: semaphore for rate limiting
        max_retries: maximum number of retry attempts

    Returns:
        a dictionary containing the classification result.
    """
    if semaphore:
        async with semaphore:
            return await _llm_classification_async(
                info, prompt, temperature, max_retries
            )
    else:
        return await _llm_classification_async(info, prompt, temperature, max_retries)


async def _llm_classification_async(
    info: Dict[str, Any], prompt: str, temperature: float = 0.1, max_retries: int = 3
):
    """
    Internal async function for LLM classification with retry logic and error handling.
    """
    dataset_id = info.get("id", "unknown")
    formatted_prompt = prompt.replace("{info}", json.dumps(info))

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=api_config["model"],
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": formatted_prompt}],
                    },
                ],
                temperature=temperature,
            )

            # Extract and parse the response
            raw_output = response.choices[0].message.content.strip()
            parsed_output = parse_output(raw_output, dataset_id)

            # Validate that the output matches the expected dataset ID
            if parsed_output.get("id") != dataset_id:
                print(
                    f"Warning: Dataset ID mismatch for {dataset_id}. Got: {parsed_output.get('id')}"
                )
                # Fix the ID if it's wrong
                parsed_output["id"] = dataset_id

            return parsed_output

        except APIConnectionError as e:
            print(
                f"Connection error for dataset {dataset_id} (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                # Exponential backoff for connection errors
                wait_time = 2**attempt
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return create_default_classification(
                    dataset_id,
                    f"Connection error after {max_retries} attempts: {str(e)}",
                )

        except APITimeoutError as e:
            print(
                f"Timeout error for dataset {dataset_id} (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                # Exponential backoff for timeout errors
                wait_time = 2**attempt
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return create_default_classification(
                    dataset_id, f"Timeout error after {max_retries} attempts: {str(e)}"
                )

        except RateLimitError as e:
            print(
                f"Rate limit error for dataset {dataset_id} (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                # Longer wait for rate limit errors
                wait_time = 5 * (2**attempt)
                print(f"Rate limited. Waiting {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return create_default_classification(
                    dataset_id,
                    f"Rate limit error after {max_retries} attempts: {str(e)}",
                )

        except APIError as e:
            print(
                f"API error for dataset {dataset_id} (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                # Short wait for API errors
                wait_time = 1 * (2**attempt)
                print(f"API error. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return create_default_classification(
                    dataset_id, f"API error after {max_retries} attempts: {str(e)}"
                )

        except ValueError as e:
            # Output format errors - these are usually not retryable
            print(f"Output format error for dataset {dataset_id}: {e}")
            if attempt < max_retries - 1:
                # Short wait before retry for format errors
                wait_time = 1
                print(f"Retrying with different temperature...")
                await asyncio.sleep(wait_time)
                # Slightly adjust temperature to potentially get different output format
                temperature = max(0.1, temperature - 0.1)
            else:
                return create_default_classification(
                    dataset_id,
                    f"Output format error after {max_retries} attempts: {str(e)}",
                )

        except Exception as e:
            print(
                f"Unexpected error for dataset {dataset_id} (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                # Exponential backoff for unexpected errors
                wait_time = 2**attempt
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return create_default_classification(
                    dataset_id,
                    f"Unexpected error after {max_retries} attempts: {str(e)}",
                )

    # This should never be reached, but just in case
    return create_default_classification(dataset_id, "Maximum retries exceeded")


async def process_dataset_batch(
    datasets: List[Dict[str, Any]],
    prompt: str,
    max_concurrent: int = 5,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Process a batch of datasets concurrently with enhanced error handling.

    Args:
        datasets: List of dataset metadata dictionaries
        prompt: The prompt template
        max_concurrent: Maximum number of concurrent requests
        temperature: Temperature for the LLM
        max_retries: Maximum number of retry attempts for each dataset

    Returns:
        List of processed datasets with classification results
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_dataset(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        dataset_id = item.get("id", f"unknown_{idx}")
        print(f"Processing {idx}/{len(datasets)}: id={dataset_id}")

        try:
            classification = await llm_classification_async(
                item, prompt, temperature, semaphore, max_retries
            )

            if classification and isinstance(classification, dict):
                # Validate classification structure
                if (
                    "classifications" in classification
                    and "reasoning" in classification
                ):
                    classifications_data = classification["classifications"]
                    if all(
                        isinstance(c, dict)
                        and "category_name" in c
                        and "category_id" in c
                        for c in classifications_data
                    ):
                        # Store all classifications
                        item["domain_names"] = [
                            c["category_name"] for c in classifications_data
                        ]
                        item["domain_ids"] = [
                            c["category_id"] for c in classifications_data
                        ]
                        item["domain_reasoning"] = classification["reasoning"]

                        # Handle empty classifications
                        if not item["domain_names"] or not item["domain_ids"]:
                            item["domain_names"] = ["Other"]
                            item["domain_ids"] = [7]
                            # item['domain_reasoning'] = 'No valid classifications found, defaulting to Other'
                    else:
                        raise ValueError(
                            "Missing category_name or category_id in classifications"
                        )
                else:
                    raise ValueError("Missing classifications or reasoning in response")
            else:
                raise ValueError("Invalid classification result format")

        except Exception as e:
            print(f"Final error processing dataset {dataset_id}: {e}")
            # Set default values for completely failed cases
            item["domain_names"] = ["Other"]
            item["domain_ids"] = [7]
            item["domain_reasoning"] = f"Processing failed: {str(e)}"

        return item

    # Create tasks for all datasets
    tasks = [process_single_dataset(item, idx + 1) for idx, item in enumerate(datasets)]

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and return valid results
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i+1} failed with exception: {result}")
            # Create a minimal result for failed tasks
            dataset_id = datasets[i].get("id", f"unknown_{i+1}")
            failed_item = datasets[i].copy()
            failed_item["domain_names"] = ["Other"]
            failed_item["domain_ids"] = [7]
            failed_item["domain_reasoning"] = f"Task execution failed: {str(result)}"
            valid_results.append(failed_item)
        else:
            valid_results.append(result)

    return valid_results


def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(data, file_path):
    """Append data to an existing JSONL file."""
    with open(file_path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


async def load_jsonl_async(file_path):
    """Load data from a JSONL file asynchronously."""
    data = []
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        async for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


async def save_jsonl_async(data, file_path):
    """Save data to a JSONL file asynchronously."""
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            await f.write(json.dumps(item, ensure_ascii=False) + "\n")


async def append_jsonl_async(data, file_path):
    """Append data to an existing JSONL file asynchronously."""
    async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
        for item in data:
            await f.write(json.dumps(item, ensure_ascii=False) + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Classify dataset metadata using LLM with concurrent processing."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--main_info",
        "-m",
        default=None,
        type=str,
        help="Path to the main info JSONL file.",
    )
    parser.add_argument(
        "--max_concurrent",
        "-c",
        type=int,
        default=50,
        help="Maximum number of concurrent requests (default: 50)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.1,
        help="Temperature for LLM generation (default: 0.7)",
    )
    parser.add_argument(
        "--max_retries",
        "-r",
        type=int,
        default=3,
        help="Maximum number of retry attempts per dataset (default: 3)",
    )
    args = parser.parse_args()

    # Load input data
    data = load_jsonl(args.input)

    print(
        f"Starting concurrent processing of {len(data)} datasets with max {args.max_concurrent} concurrent requests..."
    )
    print(f"Using temperature: {args.temperature}, max retries: {args.max_retries}")

    # Process datasets concurrently
    results = await process_dataset_batch(
        data,
        prompt,
        max_concurrent=args.max_concurrent,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )

    # Handle main_info if provided
    if args.main_info:
        await append_jsonl_async(results, args.main_info)
        print(f"Appended classification results to {args.main_info}")
    else:
        await save_jsonl_async(results, args.output)
        print(f"Saved classification results to {args.output}")

    print(f"Successfully processed {len(results)} datasets")


if __name__ == "__main__":
    asyncio.run(main())
