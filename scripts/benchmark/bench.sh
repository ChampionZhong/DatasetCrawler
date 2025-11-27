#!/bin/bash

#SBATCH --job-name=benchmark
#SBATCH --partition=raise
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --time=24:00:00

timestamp=$(date +'%Y-%m-%d')
start_date=${timestamp}
end_date=$(date -d "${timestamp} - 60 days" +'%Y-%m-%d')

echo "start_date: ${start_date}"
echo "end_date: ${end_date}"

mkdir -p data/benchmark/benchmark_${timestamp}

python 1_crawler.py -o "data/benchmark/benchmark_${timestamp}/1_benchmark_crawler_${timestamp}.pkl" --limit 70000
python 2_process.py -i "data/benchmark/benchmark_${timestamp}/1_benchmark_crawler_${timestamp}.pkl" -o "data/benchmark/benchmark_${timestamp}/2_benchmark_processed_${timestamp}.jsonl" --start_date "${start_date}" --end_date "${end_date}"
# python benchmark/benchmark.py -i "data/benchmark/benchmark_${timestamp}/2_benchmark_processed_${timestamp}.jsonl" -o "data/benchmark/benchmark_${timestamp}/3_benchmark_classified_${timestamp}.jsonl"
# python utils/jsonl2xlsx.py -i "data/benchmark/benchmark_${timestamp}/3_benchmark_classified_${timestamp}.jsonl" -o "data/benchmark/benchmark_${timestamp}/3_benchmark_classified_${timestamp}.xlsx"