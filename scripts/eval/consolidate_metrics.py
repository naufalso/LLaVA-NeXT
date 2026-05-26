#!/usr/bin/env python3
"""
Consolidate all metrics from results_debug/eval subdirectories into a single JSON and CSV file.

This script:
1. Scans all subdirectories in results_debug/eval for *_metrics.json files
2. Extracts model name, dataset/task, and all metrics
3. Consolidates into a structured JSON file
4. Exports to CSV for easy viewing and analysis
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse


def extract_model_and_task(filepath: Path, base_dir: Path) -> tuple:
    """
    Extract model name and task from the filepath.
    
    Args:
        filepath: Path to the metrics file
        base_dir: Base directory (results_debug/eval)
    
    Returns:
        tuple: (model_name, task_name)
    """
    # Get relative path from base_dir
    rel_path = filepath.relative_to(base_dir)
    
    # Task is the first directory
    task = rel_path.parts[0]
    
    # Model name is extracted from filename (remove _metrics.json)
    filename = filepath.stem  # filename without extension
    model_name = filename.replace(f"-{task}_metrics", "").replace(f"_{task}_metrics", "")
    
    return model_name, task


def load_metrics_files(base_dir: str) -> List[Dict[str, Any]]:
    """
    Load all metrics files from the base directory.
    
    Args:
        base_dir: Base directory containing subdirectories with metrics files
    
    Returns:
        List of dictionaries with model, task, and metrics
    """
    base_path = Path(base_dir)
    metrics_files = list(base_path.rglob("*_metrics.json"))
    
    all_metrics = []
    
    for metrics_file in sorted(metrics_files):
        # Skip consolidated outputs in the base directory
        if metrics_file.parent == base_path:
            continue
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            model_name, task = extract_model_and_task(metrics_file, base_path)
            
            # Create consolidated entry
            entry = {
                "model": model_name,
                "task": task,
                **metrics_data  # Unpack all metrics
            }
            
            all_metrics.append(entry)
            print(f"Loaded: {model_name} - {task}")
            
        except Exception as e:
            print(f"Error loading {metrics_file}: {e}")
    
    return all_metrics


def save_to_json(data: List[Dict[str, Any]], output_path: str):
    """
    Save consolidated metrics to JSON file.
    
    Args:
        data: List of metrics dictionaries
        output_path: Path to output JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved JSON to: {output_path}")


def save_to_csv(data: List[Dict[str, Any]], output_path: str):
    """
    Save consolidated metrics to CSV file.
    
    Args:
        data: List of metrics dictionaries
        output_path: Path to output CSV file
    """
    if not data:
        print("No data to save to CSV")
        return
    
    # Get all unique field names
    fieldnames = set()
    for entry in data:
        fieldnames.update(entry.keys())
    
    # Sort fieldnames: model, task, then alphabetically for metrics
    priority_fields = ["model", "task"]
    metric_fields = sorted([f for f in fieldnames if f not in priority_fields])
    fieldnames = priority_fields + metric_fields
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved CSV to: {output_path}")


def get_metric_for_task(entry: Dict[str, Any], task: str) -> float:
    """Get the appropriate metric value for a task.

    For coco and flickr, returns CIDEr.
    For other tasks, returns accuracy.
    """
    if task in ["coco", "flickr"]:
        return entry.get("CIDEr")
    return entry.get("accuracy")


def create_summary_table(metrics: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Create a pivot table with models as rows and tasks as columns."""

    tasks = sorted(set(entry["task"] for entry in metrics))
    models = sorted(set(entry["model"] for entry in metrics))

    metrics_lookup = {}
    for entry in metrics:
        key = (entry["model"], entry["task"])
        metrics_lookup[key] = entry

    summary: Dict[str, Dict[str, float]] = {}
    for model in models:
        summary[model] = {}
        for task in tasks:
            entry = metrics_lookup.get((model, task))
            if entry:
                summary[model][task] = get_metric_for_task(entry, task)
            else:
                summary[model][task] = None

    return summary, tasks


def save_summary_csv(summary: Dict[str, Dict[str, float]], tasks: List[str], output_path: str):
    """Save the summary table to CSV."""

    fieldnames = ["model"] + tasks

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model in sorted(summary.keys()):
            row = {"model": model}
            for task in tasks:
                value = summary[model].get(task)
                if value is not None:
                    row[task] = f"{value:.2f}"
                else:
                    row[task] = ""
            writer.writerow(row)

    print(f"Saved summary CSV to: {output_path}")


def create_summary_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create summary statistics from the consolidated data.
    
    Args:
        data: List of metrics dictionaries
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_evaluations": len(data),
        "unique_models": len(set(entry["model"] for entry in data)),
        "unique_tasks": len(set(entry["task"] for entry in data)),
        "models": sorted(set(entry["model"] for entry in data)),
        "tasks": sorted(set(entry["task"] for entry in data))
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate evaluation metrics into JSON and CSV files"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results_debug/eval",
        help="Input directory containing evaluation results (default: results_debug/eval)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_debug",
        help="Output directory for consolidated files (default: results_debug)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="consolidated_metrics",
        help="Prefix for output files (default: consolidated_metrics)"
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="summary_metrics.csv",
        help="Filename for summary CSV (default: summary_metrics.csv)"
    )
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent
    input_dir = workspace_root / args.input_dir
    output_dir = workspace_root / args.output_dir
    
    print(f"Loading metrics from: {input_dir}")
    print("=" * 80)
    
    # Load all metrics
    all_metrics = load_metrics_files(str(input_dir))
    
    if not all_metrics:
        print("\nNo metrics files found!")
        return
    
    # Create summary
    summary = create_summary_stats(all_metrics)
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total evaluations: {summary['total_evaluations']}")
    print(f"  Unique models: {summary['unique_models']}")
    print(f"  Unique tasks: {summary['unique_tasks']}")
    print(f"  Models: {', '.join(summary['models'])}")
    print(f"  Tasks: {', '.join(summary['tasks'])}")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save consolidated data
    json_output = output_dir / f"{args.output_prefix}.json"
    csv_output = output_dir / f"{args.output_prefix}.csv"
    summary_csv_output = output_dir / args.summary_csv
    
    # Create output with metadata
    output_data = {
        "summary": summary,
        "metrics": all_metrics
    }
    
    save_to_json(output_data, str(json_output))
    save_to_csv(all_metrics, str(csv_output))

    # Save summary CSV
    summary_table, tasks = create_summary_table(all_metrics)
    save_summary_csv(summary_table, tasks, str(summary_csv_output))
    
    print("\n✓ Consolidation complete!")


if __name__ == "__main__":
    main()
