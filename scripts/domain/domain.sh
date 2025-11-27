#!/bin/bash

#SBATCH --partition=raise
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --output=domain_%j.out
#SBATCH --error=domain_%j.err

# Unified domain-specific dataset filtering script
# Usage: sbatch domain.sh <domain> [options]
# Domains: agent, agent_specific, finance
# Note: This script should be run from the project root directory

# Check if domain is provided
if [ -z "$1" ]; then
    echo "Error: Domain not specified"
    echo "Usage: sbatch domain.sh <domain>"
    echo "Available domains: agent, agent_specific, finance, vqa"
    exit 1
fi

DOMAIN=$1
shift  # Remove domain from arguments

timestamp=$(date +'%Y-%m-%d')

# Domain-specific configurations
case $DOMAIN in
    agent)
        MAX_CONCURRENT=32
        BATCH_SIZE=500
        MAX_RETRIES=3
        CHECKPOINT_INTERVAL=500
        INPUT_FILE="data/domain/domain_${timestamp}/2_domain_processed_${timestamp}.jsonl"
        OUTPUT_FILE="data/domain/domain_${timestamp}/agent_${timestamp}.jsonl"
        ;;
    agent_specific)
        MAX_CONCURRENT=16
        BATCH_SIZE=64
        MAX_RETRIES=3
        TIMEOUT=180
        MIN_QUALITY=6
        MIN_CONFIDENCE=0.7
        CHECKPOINT_INTERVAL=32
        INPUT_FILE="data/domain/domain_${timestamp}/2_domain_processed_${timestamp}.jsonl"
        OUTPUT_FILE="data/domain/domain_${timestamp}/agent_specific_${timestamp}.jsonl"
        ;;
    finance)
        MAX_CONCURRENT=32
        BATCH_SIZE=500
        MAX_RETRIES=3
        CHECKPOINT_INTERVAL=500
        INPUT_FILE="data/domain/domain_${timestamp}/2_domain_processed_${timestamp}.jsonl"
        OUTPUT_FILE="data/domain/domain_${timestamp}/finance_${timestamp}.jsonl"
        ;;
    vqa)
        MAX_CONCURRENT=32
        BATCH_SIZE=500
        MAX_RETRIES=3
        CHECKPOINT_INTERVAL=500
        INPUT_FILE="data/domain/domain_${timestamp}/2_domain_processed_${timestamp}.jsonl"
        OUTPUT_FILE="data/domain/domain_${timestamp}/vqa_${timestamp}.jsonl"
        ;;
    *)
        echo "Error: Unknown domain: $DOMAIN"
        echo "Available domains: agent, agent_specific, finance, vqa"
        exit 1
        ;;
esac

echo "================================================"
echo "Domain Dataset Filter - $DOMAIN"
echo "================================================"
echo "Timestamp: ${timestamp}"
echo "Domain: ${DOMAIN}"
echo "Input: ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo "Model: (from config/api.json)"
echo "Max Concurrent: ${MAX_CONCURRENT}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max Retries: ${MAX_RETRIES}"
if [ "$DOMAIN" = "agent_specific" ]; then
    echo "Min Quality Score: ${MIN_QUALITY}/10"
    echo "Min Confidence: ${MIN_CONFIDENCE}"
    echo "Timeout: ${TIMEOUT}s"
fi
echo "Checkpoint Interval: ${CHECKPOINT_INTERVAL}"
echo "================================================"

# Check if input file exists
if [ ! -f "${INPUT_FILE}" ]; then
    echo "Error: Input file not found: ${INPUT_FILE}"
    if [ "$DOMAIN" = "agent_specific" ]; then
        echo "Please run 'domain.sh agent' first to generate coarse-filtered data."
    else
        echo "Please run 2_process.py first to generate processed data."
    fi
    exit 1
fi

# Count input datasets
INPUT_COUNT=$(wc -l < "${INPUT_FILE}")
echo "Input datasets: ${INPUT_COUNT}"
echo ""

# Build command
CMD="python domain/domain.py \
    --domain ${DOMAIN} \
    -i \"${INPUT_FILE}\" \
    -o \"${OUTPUT_FILE}\" \
    --max_concurrent ${MAX_CONCURRENT} \
    --batch_size ${BATCH_SIZE} \
    --max_retries ${MAX_RETRIES} \
    --checkpoint_interval ${CHECKPOINT_INTERVAL}"

# Add domain-specific parameters
if [ "$DOMAIN" = "agent_specific" ]; then
    CMD="$CMD --timeout ${TIMEOUT} --min_quality_score ${MIN_QUALITY} --min_confidence ${MIN_CONFIDENCE}"
fi

# Run filtering
# API parameters (model, api_key, base_url, temperature) are loaded from config/api.json
eval $CMD

EXIT_CODE=$?

# Check results
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✓ Processing completed successfully!"
    echo "================================================"
    
    # Convert to Excel
    if [ -f "${OUTPUT_FILE}" ]; then
        OUTPUT_COUNT=$(wc -l < "${OUTPUT_FILE}")
        echo "${DOMAIN}-related datasets: ${OUTPUT_COUNT}"
        if [ ${INPUT_COUNT} -gt 0 ]; then
            ACCEPTANCE_RATE=$(echo "scale=2; ${OUTPUT_COUNT}*100/${INPUT_COUNT}" | bc)
            echo "Acceptance rate: ${ACCEPTANCE_RATE}%"
        fi
        
        echo ""
        echo "Converting to Excel..."
        python utils/json2xlsx.py -i "${OUTPUT_FILE}" -o "${OUTPUT_FILE%.jsonl}.xlsx"
        
        if [ $? -eq 0 ]; then
            echo "✓ Excel file created: ${OUTPUT_FILE%.jsonl}.xlsx"
        fi
    fi
    
    echo ""
    echo "Files created:"
    echo "  - JSONL: ${OUTPUT_FILE}"
    echo "  - Excel: ${OUTPUT_FILE%.jsonl}.xlsx"
    echo "  - Checkpoint: ${OUTPUT_FILE}.checkpoint.json"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "✗ Processing failed or interrupted (Exit code: ${EXIT_CODE})"
    echo "================================================"
    echo "You can resume by running the same command again (checkpoint enabled)"
    echo "Checkpoint file: ${OUTPUT_FILE}.checkpoint.json"
    echo "================================================"
fi

