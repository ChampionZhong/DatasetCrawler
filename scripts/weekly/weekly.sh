#!/bin/bash

#SBATCH --job-name=weekly
#SBATCH --partition=raise
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=weekly_%j.out
#SBATCH --error=weekly_%j.err
#SBATCH --time=24:00:00

timestamp=$(date +'%Y-%m-%d')

mkdir -p data/weekly/weekly_${timestamp}

python 1_crawler.py -o "data/weekly/weekly_${timestamp}/1_weekly_crawler_${timestamp}.pkl" --limit 500000
python 2_process.py -i "data/weekly/weekly_${timestamp}/1_weekly_crawler_${timestamp}.pkl" -o "data/weekly/weekly_${timestamp}/2_weekly_processed_${timestamp}.jsonl" --start_date "2025-12-29" --end_date "2026-01-11"
python weekly/weekly.py -i "data/weekly/weekly_${timestamp}/2_weekly_processed_${timestamp}.jsonl" -o "data/weekly/weekly_${timestamp}/3_weekly_classified_${timestamp}.jsonl"
python 4_add_to_base.py -i "data/weekly/weekly_${timestamp}/3_weekly_classified_${timestamp}.jsonl" -o "data/weekly/base/base.jsonl"
python utils/json2xlsx.py -i "data/weekly/weekly_${timestamp}/3_weekly_classified_${timestamp}.jsonl" -o "data/weekly/weekly_${timestamp}/3_weekly_classified_${timestamp}.xlsx"