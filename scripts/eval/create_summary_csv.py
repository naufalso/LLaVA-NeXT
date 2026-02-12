#!/usr/bin/env python3
"""
Generate a summary CSV with models as rows and tasks as columns.

For coco and flickr tasks, uses CIDEr metric.
For other tasks (okvqa, textvqa, vizwiz, vqav2), uses accuracy metric.

Output format:
model,coco,flickr,okvqa,textvqa,vizwiz,vqav2
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_consolidated_metrics(filepath: str) -> Dict[str, Any]:
    """Load the consolidated metrics JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_metric_for_task(entry: Dict[str, Any], task: str) -> float:
    """Get the appropriate metric value for a task.
    
    For coco and flickr, returns CIDEr.
    For other tasks, returns accuracy.
    """
    if task in ['coco', 'flickr']:
        return entry.get('CIDEr')
    else:
        return entry.get('accuracy')


def create_summary_table(metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Create a pivot table with models as rows and tasks as columns."""
    
    # Get unique tasks and models, sorted
    tasks = sorted(set(entry['task'] for entry in metrics))
    models = sorted(set(entry['model'] for entry in metrics))
    
    # Create a lookup dictionary for quick access
    metrics_lookup = {}
    for entry in metrics:
        key = (entry['model'], entry['task'])
        metrics_lookup[key] = entry
    
    # Build the summary table
    summary = {}
    for model in models:
        summary[model] = {}
        for task in tasks:
            entry = metrics_lookup.get((model, task))
            if entry:
                value = get_metric_for_task(entry, task)
                summary[model][task] = value
            else:
                summary[model][task] = None
    
    return summary, tasks


def save_summary_csv(summary: Dict[str, Dict[str, float]], tasks: List[str], output_path: str):
    """Save the summary table to CSV."""
    
    fieldnames = ['model'] + tasks
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for model in sorted(summary.keys()):
            row = {'model': model}
            for task in tasks:
                value = summary[model].get(task)
                if value is not None:
                    row[task] = f"{value:.2f}"
                else:
                    row[task] = ''
            writer.writerow(row)
    
    print(f"Saved summary CSV to: {output_path}")


def print_summary_table(summary: Dict[str, Dict[str, float]], tasks: List[str]):
    """Print the summary table in a formatted way."""
    
    print("\n" + "=" * 120)
    print("SUMMARY TABLE: Models vs Tasks")
    print("=" * 120)
    print("Note: CIDEr metric for COCO and Flickr, Accuracy for others\n")
    
    # Print header
    print(f"{'Model':<45} ", end="")
    for task in tasks:
        print(f"{task:>15} ", end="")
    print()
    print("-" * 120)
    
    # Print rows
    for model in sorted(summary.keys()):
        print(f"{model:<45} ", end="")
        for task in tasks:
            value = summary[model].get(task)
            if value is not None:
                print(f"{value:>15.2f} ", end="")
            else:
                print(f"{'N/A':>15} ", end="")
        print()
    
    print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a summary CSV with models vs tasks"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results_debug/consolidated_metrics.json",
        help="Path to consolidated metrics JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_debug/summary_metrics.csv",
        help="Output path for summary CSV file"
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the summary table to console"
    )
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent
    input_path = workspace_root / args.input
    output_path = workspace_root / args.output
    
    print(f"Loading metrics from: {input_path}")
    data = load_consolidated_metrics(str(input_path))
    
    metrics = data['metrics']
    summary, tasks = create_summary_table(metrics)
    
    if args.print:
        print_summary_table(summary, tasks)
    
    save_summary_csv(summary, tasks, str(output_path))
    print("✓ Summary CSV generation complete!")


if __name__ == "__main__":
    main()
