#!/bin/bash
# bash monitor_progress.sh data/2025-10-27/agent_2025-10-27.jsonl.checkpoint.json
# watch -n 10 bash monitor_progress.sh data/2025-10-27/agent_2025-10-27.jsonl.checkpoint.json
# bash monitor_progress.sh data/2025-10-27/agent_specific_2025-10-27.jsonl.checkpoint.json


# monitor dataset processing progress

if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_file>"
    echo ""
    echo "Example:"
    echo "  $0 data/2025-10-27/agent_2025-10-27.jsonl.checkpoint.json"
    echo "  $0 data/2025-10-27/agent_specific_2025-10-27.jsonl.checkpoint.json"
    echo ""
    echo "Monitor continuously:"
    echo "  watch -n 10 $0 <checkpoint_file>"
    exit 1
fi

CHECKPOINT_FILE=$1
OUTPUT_FILE="${CHECKPOINT_FILE%.checkpoint.json}"

echo "================================================"
echo "Dataset Processing Progress Monitor"
echo "================================================"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# check if checkpoint file exists
if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo "❌ Checkpoint file not found: ${CHECKPOINT_FILE}"
    echo ""
    echo "Possible reasons:"
    echo "  1. The process has not started yet"
    echo "  2. Checkpoint is disabled (--no_checkpoint)"
    echo "  3. File path is incorrect"
    echo ""
    exit 1
fi

# show checkpoint information
echo "📊 Checkpoint Information:"
echo "---"
if command -v jq &> /dev/null; then
    PROCESSED_COUNT=$(jq -r '.processed_count // 0' "${CHECKPOINT_FILE}")
    LAST_UPDATE=$(jq -r '.last_update // "Unknown"' "${CHECKPOINT_FILE}")
    PROCESSED_IDS_COUNT=$(jq -r '.processed_ids | length' "${CHECKPOINT_FILE}")
    
    echo "Processed Count: ${PROCESSED_COUNT}"
    echo "Unique IDs: ${PROCESSED_IDS_COUNT}"
    echo "Last Update: ${LAST_UPDATE}"
else
    echo "⚠️  Install 'jq' for better output: yum install jq / apt install jq"
    echo ""
    cat "${CHECKPOINT_FILE}"
fi

# show output file information
echo ""
echo "📁 Output File Information:"
echo "---"
if [ -f "${OUTPUT_FILE}" ]; then
    LINE_COUNT=$(wc -l < "${OUTPUT_FILE}")
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    echo "Output File: ${OUTPUT_FILE}"
    echo "Filtered Datasets: ${LINE_COUNT}"
    echo "File Size: ${FILE_SIZE}"
    
    if [ ${LINE_COUNT} -gt 0 ]; then
        echo ""
        echo "📝 Last 3 processed datasets:"
        echo "---"
        if command -v jq &> /dev/null; then
            tail -n 3 "${OUTPUT_FILE}" | jq -r '.id' 2>/dev/null
        else
            tail -n 3 "${OUTPUT_FILE}" | grep -o '"id":"[^"]*"' | cut -d'"' -f4
        fi
    fi
else
    echo "Output file not created yet: ${OUTPUT_FILE}"
    echo "Processing may not have started or no datasets passed the filter yet."
fi

echo ""
echo "================================================"
echo "💡 Tips:"
echo "  - Run continuously: watch -n 10 bash $0 ${CHECKPOINT_FILE}"
echo "  - Check logs: tail -f agent_*.err"
echo "  - View results: cat ${OUTPUT_FILE} | jq ."
echo "================================================"

