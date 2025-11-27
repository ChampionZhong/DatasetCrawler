import json
import asyncio
import argparse
import os
import time
from datetime import datetime
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import logging
from typing import List, Dict, Optional, Set
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
        "temperature": config.get("temperature", 0.3),
    }


# Load domain configuration
def load_domain_config(domain: str) -> Dict:
    """Load domain configuration from config/domains/{domain}.json."""
    config_path = Path(__file__).parent.parent / "config" / "domains" / f"{domain}.json"
    if not config_path.exists():
        raise ValueError(f"Domain configuration not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def append_jsonl(item: Dict, file_path: str):
    """Append a single item to an existing JSONL file."""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


class CheckpointManager:
    """Manage checkpoints for resumable processing."""

    def __init__(self, checkpoint_file: str, auto_save_interval: int = 50):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_file: Path to checkpoint file
            auto_save_interval: Auto-save checkpoint every N processed datasets
        """
        self.checkpoint_file = checkpoint_file
        self.auto_save_interval = auto_save_interval
        self.processed_ids: Set[str] = set()
        self.processed_count = 0
        self.last_save_time = time.time()

    def load(self) -> Set[str]:
        """Load processed dataset IDs from checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.processed_ids = set(data.get("processed_ids", []))
                    logger.info(
                        f"Loaded checkpoint: {len(self.processed_ids)} datasets already processed"
                    )
                    return self.processed_ids
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                return set()
        else:
            logger.info("No checkpoint found, starting fresh")
            return set()

    def save(self, force: bool = False):
        """
        Save checkpoint to file.

        Args:
            force: Force save even if auto_save_interval not reached
        """
        current_time = time.time()
        should_save = (
            force
            or self.processed_count % self.auto_save_interval == 0
            or (current_time - self.last_save_time) > 300  # Save every 5 minutes
        )

        if should_save and self.processed_count > 0:
            try:
                checkpoint_data = {
                    "processed_ids": list(self.processed_ids),
                    "processed_count": self.processed_count,
                    "last_update": datetime.now().isoformat(),
                }
                temp_file = self.checkpoint_file + ".tmp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                os.replace(temp_file, self.checkpoint_file)
                self.last_save_time = current_time
                logger.info(
                    f"Checkpoint saved: {self.processed_count} datasets processed"
                )
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

    def add_processed(self, dataset_id: str):
        """Mark a dataset as processed."""
        self.processed_ids.add(dataset_id)
        self.processed_count += 1
        self.save()

    def is_processed(self, dataset_id: str) -> bool:
        """Check if a dataset has been processed."""
        return dataset_id in self.processed_ids

    def finalize(self):
        """Finalize checkpoint (force save and cleanup if needed)."""
        self.save(force=True)
        logger.info(
            f"Checkpoint finalized: {len(self.processed_ids)} total datasets processed"
        )


class DomainDatasetFilter:
    """Unified filter for domain-specific datasets using OpenAI API."""

    def __init__(
        self,
        domain_config: Dict,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = None,
        max_concurrent: int = 10,
        max_retries: int = 3,
        timeout: Optional[int] = None,
    ):
        """
        Initialize the filter.

        Args:
            domain_config: Domain configuration dictionary
            api_key: OpenAI API key
            base_url: Optional custom base URL for API
            model: Model to use for classification
            temperature: Temperature for LLM generation (uses domain default if None)
            max_concurrent: Maximum number of concurrent API requests
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout in seconds for API requests
        """
        self.domain_config = domain_config
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout
        )
        self.model = model
        self.temperature = temperature or domain_config.get("default_temperature", 0.3)
        self.max_tokens = domain_config.get("default_max_tokens", 500)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.response_format = domain_config.get("response_format", {})
        self.filtering = domain_config.get("filtering", {})

    def _build_prompt(self, dataset_info: Dict) -> str:
        """Build the prompt for the LLM based on domain configuration."""
        dataset_id = dataset_info.get("id", "N/A")
        description = dataset_info.get("description", "N/A")
        pretty_name = dataset_info.get("pretty_name", "N/A")
        
        # Handle tags - can be dict or string
        tags = dataset_info.get("tags", {})
        if isinstance(tags, str):
            tags = {"raw_tags": tags}
        elif not isinstance(tags, dict):
            tags = {}
        
        # Truncate description if needed
        desc_max_len = 800 if "agent_specific" in self.domain_config.get("name", "") else 500
        description = description[:desc_max_len] if description else "N/A"
        
        # Get previous classification info if exists (for agent_specific)
        prev_classification = {}
        if "agent_specific" in self.domain_config.get("name", ""):
            prev_classification = dataset_info.get("agent_classification", {})
            if isinstance(prev_classification, str):
                prev_classification = {}
        
        prev_tags = prev_classification.get("tags", [])
        if isinstance(prev_tags, str):
            prev_tags = [prev_tags]
        elif not isinstance(prev_tags, list):
            prev_tags = []
            
        prev_reasoning = prev_classification.get("reasoning", "N/A")
        
        # Build tags list or categories description
        template_vars = {
            "dataset_id": dataset_id,
            "pretty_name": pretty_name,
            "description": description,
            "tags": json.dumps(tags, ensure_ascii=False)[:300],
            "prev_tags": prev_tags,
            "prev_reasoning": prev_reasoning,
        }
        
        # Add tags list or categories
        if "tags" in self.domain_config:
            tags_list = json.dumps(self.domain_config["tags"], indent=2)
            template_vars["tags_list"] = tags_list
        elif "categories" in self.domain_config:
            categories_desc = "\n".join([
                f"- {key}: {value}" 
                for key, value in self.domain_config["categories"].items()
            ])
            template_vars["categories_desc"] = categories_desc
        
        # Format the prompt template
        prompt_template = self.domain_config.get("user_prompt_template", "")
        try:
            prompt = prompt_template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}, using raw template")
            prompt = prompt_template
        
        return prompt

    async def classify_dataset(self, dataset_info: Dict) -> Optional[Dict]:
        """
        Classify a single dataset using OpenAI API with retry logic.

        Args:
            dataset_info: Dataset information dictionary

        Returns:
            Classification result or None if failed
        """
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    prompt = self._build_prompt(dataset_info)

                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": self.domain_config.get("system_prompt", ""),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"},
                    )

                    result = json.loads(response.choices[0].message.content)

                    # Validate required keys based on response format
                    required_keys = [
                        self.response_format.get("is_related_key", "is_related"),
                        self.response_format.get("confidence_key", "confidence"),
                        self.response_format.get("reasoning_key", "reasoning"),
                    ]
                    
                    # Add tags key if exists
                    if self.response_format.get("tags_key"):
                        required_keys.append(self.response_format["tags_key"])
                    
                    # Add quality_score key for agent_specific
                    if self.response_format.get("quality_score_key"):
                        required_keys.append(self.response_format["quality_score_key"])

                    if not all(key in result for key in required_keys):
                        raise ValueError("Invalid response format from API")

                    return result

                except Exception as e:
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed for dataset {dataset_info.get('id')}: {e}"
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2**attempt)  # Exponential backoff
                    else:
                        logger.error(
                            f"All retries failed for dataset {dataset_info.get('id')}"
                        )
                        return None

    async def process_batch(
        self,
        datasets: List[Dict],
        output_file: str,
        checkpoint_manager: Optional[CheckpointManager] = None,
        min_quality_score: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> tuple:
        """
        Process a batch of datasets concurrently.

        Args:
            datasets: List of dataset dictionaries
            output_file: Path to output file for incremental saving
            checkpoint_manager: Optional checkpoint manager for resumable processing
            min_quality_score: Minimum quality score (overrides config if provided)
            min_confidence: Minimum confidence (overrides config if provided)

        Returns:
            Tuple of (filtered_datasets, stats)
        """
        filtered_datasets = []
        domain_name = self.domain_config.get("name", "domain")
        
        # Use provided values or fall back to config
        min_conf = min_confidence or self.filtering.get("min_confidence", 0.5)
        min_quality = min_quality_score or self.filtering.get("min_quality_score")
        
        stats = {
            "total": len(datasets),
            "related": 0,
            "not_related": 0,
            "failed": 0,
            "skipped": 0,
        }

        # Create tasks for all datasets
        tasks = [self.classify_dataset(dataset) for dataset in datasets]

        # Process with progress bar
        results = []
        for coro in tqdm.as_completed(
            tasks, total=len(tasks), desc=f"Processing {domain_name} datasets"
        ):
            result = await coro
            results.append(result)

        # Combine results with original datasets
        for dataset, result in zip(datasets, results):
            dataset_id = dataset.get("id", "")

            # Check if already processed
            if checkpoint_manager and checkpoint_manager.is_processed(dataset_id):
                stats["skipped"] += 1
                continue

            if result is None:
                stats["failed"] += 1
                if checkpoint_manager:
                    checkpoint_manager.add_processed(dataset_id)
                continue

            # Extract values based on response format
            is_related_key = self.response_format.get("is_related_key", "is_related")
            confidence_key = self.response_format.get("confidence_key", "confidence")
            tags_key = self.response_format.get("tags_key")
            quality_score_key = self.response_format.get("quality_score_key")
            
            is_related = result.get(is_related_key, False)
            confidence = result.get(confidence_key, 0)
            quality_score = result.get(quality_score_key) if quality_score_key else None

            # Apply filtering criteria
            passes_filter = is_related and confidence >= min_conf
            if min_quality is not None and quality_score is not None:
                passes_filter = passes_filter and quality_score >= min_quality

            if passes_filter:
                # Create classified dataset entry
                classified_dataset = dataset.copy()
                
                # Build classification dict
                classification = {
                    "confidence": confidence,
                    "reasoning": result.get(self.response_format.get("reasoning_key", "reasoning"), ""),
                    "classified_at": datetime.now().isoformat(),
                }
                
                # Add tags if available
                if tags_key and tags_key in result:
                    tags = result[tags_key]
                    if isinstance(tags, str):
                        tags = [tags]
                    classification["tags"] = tags
                
                # Add quality score for agent_specific
                if quality_score is not None:
                    classification["quality_score"] = quality_score
                    classification["is_high_quality"] = True
                
                # Add primary category for agent_specific
                if self.response_format.get("primary_category_key"):
                    classification["primary_category"] = result.get(
                        self.response_format["primary_category_key"]
                    )
                
                # Add secondary categories for agent_specific
                if self.response_format.get("secondary_categories_key"):
                    classification["secondary_categories"] = result.get(
                        self.response_format["secondary_categories_key"], []
                    )
                
                # Add data characteristics for agent_specific
                if self.response_format.get("data_characteristics_key"):
                    classification["data_characteristics"] = result.get(
                        self.response_format["data_characteristics_key"], {}
                    )
                
                # Add recommended use cases for agent_specific
                if self.response_format.get("recommended_use_cases_key"):
                    classification["recommended_use_cases"] = result.get(
                        self.response_format["recommended_use_cases_key"], []
                    )
                
                # Add evaluated_by_model for agent_specific
                if "agent_specific" in domain_name:
                    classification["evaluated_by_model"] = self.model
                    classification["evaluated_at"] = datetime.now().isoformat()

                # Store classification
                classification_key = self.filtering.get("classification_key", f"{domain_name}_classification")
                classified_dataset[classification_key] = classification

                # Merge tags with existing tags
                tags_key_in_config = self.filtering.get("tags_key")
                if tags_key_in_config and tags_key and tags_key in result:
                    tags = result[tags_key]
                    if isinstance(tags, str):
                        tags = [tags]
                    if "tags" not in classified_dataset:
                        classified_dataset["tags"] = {}
                    classified_dataset["tags"][tags_key_in_config] = tags

                filtered_datasets.append(classified_dataset)
                stats["related"] += 1

                # Incremental save
                append_jsonl(classified_dataset, output_file)
                
                logger.info(
                    f"✓ Accepted: {dataset_id} (Confidence: {confidence:.2f}"
                    + (f", Quality: {quality_score}/10" if quality_score else "")
                    + ")"
                )
            else:
                stats["not_related"] += 1
                logger.debug(
                    f"✗ Rejected: {dataset_id} (Confidence: {confidence:.2f}"
                    + (f", Quality: {quality_score}/10" if quality_score else "")
                    + ")"
                )

            # Mark as processed
            if checkpoint_manager:
                checkpoint_manager.add_processed(dataset_id)

        return filtered_datasets, stats


async def main_async(args):
    """Main async function."""
    # Load domain configuration
    domain_config = load_domain_config(args.domain)
    logger.info(f"Loaded domain configuration: {domain_config.get('name')}")
    logger.info(f"Description: {domain_config.get('description')}")
    
    # Check if previous classification is required
    if domain_config.get("requires_previous_classification", False):
        logger.info("This domain requires previous classification (e.g., from agent.py)")
    
    # Check API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key must be provided via --api_key or OPENAI_API_KEY environment variable"
        )

    # Load input data
    logger.info(f"Loading data from {args.input}")
    datasets = load_jsonl(args.input)
    logger.info(f"Loaded {len(datasets)} datasets")

    # Create output directory if needed
    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )

    # Initialize checkpoint manager
    checkpoint_manager = None
    if args.enable_checkpoint:
        checkpoint_file = args.checkpoint_file or f"{args.output}.checkpoint.json"
        checkpoint_manager = CheckpointManager(
            checkpoint_file=checkpoint_file,
            auto_save_interval=args.checkpoint_interval,
        )
        checkpoint_manager.load()

        # Filter out already processed datasets
        if checkpoint_manager.processed_ids:
            original_count = len(datasets)
            datasets = [
                d
                for d in datasets
                if d.get("id", "") not in checkpoint_manager.processed_ids
            ]
            logger.info(
                f"Filtered out {original_count - len(datasets)} already processed datasets"
            )
            logger.info(f"Remaining datasets to process: {len(datasets)}")

            if len(datasets) == 0:
                logger.info("All datasets have been processed! Nothing to do.")
                return
    else:
        if os.path.exists(args.output):
            if not args.append:
                logger.info(f"Clearing existing output file: {args.output}")
                os.remove(args.output)

    # Limit number of datasets if specified
    if args.limit and args.limit > 0:
        datasets = datasets[: args.limit]
        logger.info(f"Limited to first {len(datasets)} datasets")

    # Initialize filter
    logger.info(f"Using model: {args.model} for {domain_config.get('name')} classification")
    logger.info(f"Using temperature: {args.temperature}")
    filter_engine = DomainDatasetFilter(
        domain_config=domain_config,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

    # Process datasets in batches
    batch_size = args.batch_size
    all_filtered = []
    total_stats = {
        "total": 0,
        "related": 0,
        "not_related": 0,
        "failed": 0,
        "skipped": 0,
    }

    try:
        for i in range(0, len(datasets), batch_size):
            batch = datasets[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(datasets)-1)//batch_size + 1}"
            )

            filtered, stats = await filter_engine.process_batch(
                batch,
                args.output,
                checkpoint_manager,
                min_quality_score=args.min_quality_score,
                min_confidence=args.min_confidence,
            )
            all_filtered.extend(filtered)

            # Update total stats
            for key in total_stats:
                total_stats[key] += stats[key]

            logger.info(f"Batch stats: {stats}")

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        # Finalize checkpoint
        if checkpoint_manager:
            checkpoint_manager.finalize()

    # Print final statistics
    domain_name = domain_config.get("name", "domain")
    logger.info("=" * 60)
    logger.info(f"{domain_name.upper()} Classification Results:")
    logger.info("=" * 60)
    logger.info(f"Total datasets processed: {total_stats['total']}")
    if total_stats["total"] > 0:
        acceptance_rate = total_stats["related"] / total_stats["total"] * 100
        logger.info(
            f"{domain_name.capitalize()}-related datasets: {total_stats['related']} ({acceptance_rate:.2f}%)"
        )
    else:
        logger.info(f"{domain_name.capitalize()}-related datasets: {total_stats['related']}")
    logger.info(f"Not related: {total_stats['not_related']}")
    logger.info(f"Failed: {total_stats['failed']}")
    logger.info(f"Skipped (already processed): {total_stats['skipped']}")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified domain-specific dataset filtering using OpenAI API."
    )
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        required=True,
        choices=["agent", "agent_specific", "finance", "vqa"],
        help="Domain to filter (agent, agent_specific, finance, vqa).",
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to the output JSONL file for domain-related datasets.",
    )
    
    # Load default values from config file
    api_config = load_api_config()
    
    parser.add_argument(
        "--api_key",
        type=str,
        default=api_config["api_key"],
        help="OpenAI API key (default: from config/api.json, or use OPENAI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=api_config["base_url"],
        help="Custom OpenAI API base URL (default: from config/api.json).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=api_config["model"],
        help="OpenAI model to use (default: from config/api.json).",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=None,
        help="Temperature for LLM generation (default: from domain config).",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=30,
        help="Maximum number of concurrent API requests (default: 30).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Batch size for processing (default: 500).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed requests (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for API requests (optional).",
    )
    parser.add_argument(
        "--min_quality_score",
        type=int,
        default=None,
        help="Minimum quality score (1-10) to accept a dataset (for agent_specific, default: from config).",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=None,
        help="Minimum confidence (0.0-1.0) to accept a dataset (default: from domain config).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of datasets to process (for testing).",
    )
    parser.add_argument(
        "--enable_checkpoint",
        action="store_true",
        default=True,
        help="Enable checkpoint for resumable processing (default: True).",
    )
    parser.add_argument(
        "--no_checkpoint",
        action="store_true",
        help="Disable checkpoint (overwrites --enable_checkpoint).",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=None,
        help="Path to checkpoint file (default: output_file.checkpoint.json).",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save checkpoint every N processed datasets (default: 50).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file instead of clearing it.",
    )

    args = parser.parse_args()

    # Handle no_checkpoint flag
    if args.no_checkpoint:
        args.enable_checkpoint = False

    # Load domain config to get default temperature if not provided
    if args.temperature is None:
        domain_config = load_domain_config(args.domain)
        args.temperature = domain_config.get("default_temperature", 0.3)

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

