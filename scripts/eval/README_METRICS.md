# Metrics Organization and Analysis Tools

This directory contains tools for consolidating and analyzing evaluation metrics from the LLaVA-NeXT project.

## Overview

The evaluation metrics are stored in individual JSON files across multiple subdirectories in `results_debug/eval/`. These tools help consolidate and analyze all metrics in one place.

## Files

- **`consolidate_metrics.py`**: Consolidates all metrics from individual files into single JSON and CSV files
- **`analyze_metrics.py`**: Analyzes the consolidated metrics and provides various comparison views

## Usage

### 1. Consolidate Metrics

Gather all metrics from `results_debug/eval/` subdirectories and create consolidated files:

```bash
python3 scripts/eval/consolidate_metrics.py
```

**Options:**
- `--input-dir`: Input directory containing evaluation results (default: `results_debug/eval`)
- `--output-dir`: Output directory for consolidated files (default: `results_debug`)
- `--output-prefix`: Prefix for output files (default: `consolidated_metrics`)

**Output files:**
- `results_debug/consolidated_metrics.json`: Complete data with summary and all metrics
- `results_debug/consolidated_metrics.csv`: Flat table format for easy viewing in spreadsheets

### 2. Analyze Metrics

Generate various analysis views from the consolidated metrics:

```bash
# Show all analyses
python3 scripts/eval/analyze_metrics.py

# Show only task-wise comparison
python3 scripts/eval/analyze_metrics.py --mode task

# Show only model-wise comparison
python3 scripts/eval/analyze_metrics.py --mode model

# Show only best performers per task/metric
python3 scripts/eval/analyze_metrics.py --mode best

# Show only average performance across tasks
python3 scripts/eval/analyze_metrics.py --mode avg
```

**Options:**
- `--input`: Path to consolidated metrics JSON file (default: `results_debug/consolidated_metrics.json`)
- `--mode`: Analysis mode - `all`, `task`, `model`, `best`, or `avg` (default: `all`)

## Metrics Structure

### Consolidated JSON Format

```json
{
  "summary": {
    "total_evaluations": 30,
    "unique_models": 6,
    "unique_tasks": 6,
    "models": ["model1", "model2", ...],
    "tasks": ["task1", "task2", ...]
  },
  "metrics": [
    {
      "model": "llava-next-apertus-8b",
      "task": "coco",
      "Bleu_1": 68.32,
      "Bleu_2": 50.60,
      ...
    },
    ...
  ]
}
```

### CSV Format

The CSV file contains one row per evaluation with columns:
- `model`: Model name
- `task`: Task/dataset name
- Various metric columns (Bleu_1, Bleu_2, accuracy, etc.)
- `runtime_seconds`: Evaluation runtime

## Evaluation Tasks

The tools process metrics from these tasks:
- **coco**: COCO Captioning (Bleu, METEOR, ROUGE_L, CIDEr, SPICE)
- **flickr**: Flickr30k Captioning (Bleu, METEOR, ROUGE_L, CIDEr, SPICE)
- **okvqa**: OK-VQA Question Answering (accuracy)
- **textvqa**: TextVQA (accuracy)
- **vizwiz**: VizWiz VQA (accuracy)
- **vqav2**: VQA v2 (accuracy)

## Analysis Views

### Task-wise Comparison
Shows all models compared side-by-side for each task, sorted by primary metric.

### Model-wise Comparison
Shows each model's performance across all tasks.

### Best Performers
Lists the best performing model for each task/metric combination.

### Average Performance
Shows average metric values for each model across all tasks where applicable.

## Example Workflow

```bash
# Step 1: Run evaluations (generates individual metric files)
# ... your evaluation scripts ...

# Step 2: Consolidate all metrics
python3 scripts/eval/consolidate_metrics.py

# Step 3: View best performers
python3 scripts/eval/analyze_metrics.py --mode best

# Step 4: Open CSV in spreadsheet for detailed analysis
# Open: results_debug/consolidated_metrics.csv
```

## Notes

- Metrics files are automatically discovered by searching for `*_metrics.json` files
- Model names and tasks are extracted from file paths and names
- Missing metrics are handled gracefully (shown as "N/A" or empty in outputs)
- All numeric values are formatted to 2 decimal places in analysis output
