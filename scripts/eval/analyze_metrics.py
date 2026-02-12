#!/usr/bin/env python3
"""
Analyze and visualize consolidated metrics from evaluation results.

This script provides:
1. Model comparison across different tasks
2. Best performing models per task
3. Statistical summaries
4. Comparison tables
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_consolidated_metrics(filepath: str) -> Dict[str, Any]:
    """Load the consolidated metrics JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_metrics_by_task(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group metrics by task."""
    by_task = defaultdict(list)
    for entry in data:
        by_task[entry['task']].append(entry)
    return dict(by_task)


def get_metrics_by_model(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group metrics by model."""
    by_model = defaultdict(list)
    for entry in data:
        by_model[entry['model']].append(entry)
    return dict(by_model)


def print_task_comparison(by_task: Dict[str, List[Dict[str, Any]]]):
    """Print comparison table for each task."""
    print("\n" + "=" * 100)
    print("TASK-WISE COMPARISON")
    print("=" * 100)
    
    for task, entries in sorted(by_task.items()):
        print(f"\n{task.upper()}")
        print("-" * 100)
        
        # Get all metric names (excluding model, task, runtime)
        metric_names = set()
        for entry in entries:
            metric_names.update(k for k in entry.keys() 
                              if k not in ['model', 'task', 'runtime_seconds'])
        metric_names = sorted(metric_names)
        
        # Print header
        print(f"{'Model':<40} ", end="")
        for metric in metric_names:
            print(f"{metric:>12} ", end="")
        print()
        print("-" * 100)
        
        # Sort by first metric if available
        if metric_names:
            entries_sorted = sorted(entries, 
                                  key=lambda x: x.get(metric_names[0], 0), 
                                  reverse=True)
        else:
            entries_sorted = entries
        
        # Print each model's metrics
        for entry in entries_sorted:
            print(f"{entry['model']:<40} ", end="")
            for metric in metric_names:
                value = entry.get(metric, '-')
                if isinstance(value, (int, float)):
                    print(f"{value:>12.2f} ", end="")
                else:
                    print(f"{str(value):>12} ", end="")
            print()


def print_model_comparison(by_model: Dict[str, List[Dict[str, Any]]]):
    """Print comparison table for each model across tasks."""
    print("\n" + "=" * 100)
    print("MODEL-WISE COMPARISON (Across Tasks)")
    print("=" * 100)
    
    for model, entries in sorted(by_model.items()):
        print(f"\n{model}")
        print("-" * 100)
        
        # Get all metric names
        metric_names = set()
        for entry in entries:
            metric_names.update(k for k in entry.keys() 
                              if k not in ['model', 'task', 'runtime_seconds'])
        metric_names = sorted(metric_names)
        
        # Print header
        print(f"{'Task':<15} ", end="")
        for metric in metric_names:
            print(f"{metric:>12} ", end="")
        print()
        print("-" * 100)
        
        # Print each task's metrics
        for entry in sorted(entries, key=lambda x: x['task']):
            print(f"{entry['task']:<15} ", end="")
            for metric in metric_names:
                value = entry.get(metric, '-')
                if isinstance(value, (int, float)):
                    print(f"{value:>12.2f} ", end="")
                else:
                    print(f"{str(value):>12} ", end="")
            print()


def find_best_performers(by_task: Dict[str, List[Dict[str, Any]]]):
    """Find best performing model for each task/metric combination."""
    print("\n" + "=" * 100)
    print("BEST PERFORMERS PER TASK AND METRIC")
    print("=" * 100)
    
    for task, entries in sorted(by_task.items()):
        print(f"\n{task.upper()}")
        print("-" * 50)
        
        # Get all numeric metrics
        metric_names = set()
        for entry in entries:
            for k, v in entry.items():
                if k not in ['model', 'task', 'runtime_seconds'] and isinstance(v, (int, float)):
                    metric_names.add(k)
        
        # Find best for each metric
        for metric in sorted(metric_names):
            best_entry = max(entries, key=lambda x: x.get(metric, -float('inf')))
            best_value = best_entry.get(metric)
            if best_value is not None:
                print(f"  {metric:<20}: {best_entry['model']:<40} ({best_value:.2f})")


def calculate_average_performance(by_model: Dict[str, List[Dict[str, Any]]]):
    """Calculate average performance across all tasks for each model."""
    print("\n" + "=" * 100)
    print("AVERAGE PERFORMANCE ACROSS ALL TASKS")
    print("=" * 100)
    
    # Collect all metrics
    all_metrics = set()
    for entries in by_model.values():
        for entry in entries:
            for k, v in entry.items():
                if k not in ['model', 'task', 'runtime_seconds'] and isinstance(v, (int, float)):
                    all_metrics.add(k)
    
    # Calculate averages
    model_averages = {}
    for model, entries in by_model.items():
        averages = {}
        for metric in all_metrics:
            values = [e.get(metric) for e in entries if metric in e and e.get(metric) is not None]
            if values:
                averages[metric] = sum(values) / len(values)
        model_averages[model] = averages
    
    # Print table
    print(f"\n{'Model':<40} ", end="")
    for metric in sorted(all_metrics):
        print(f"{metric:>12} ", end="")
    print()
    print("-" * 100)
    
    for model in sorted(model_averages.keys()):
        print(f"{model:<40} ", end="")
        for metric in sorted(all_metrics):
            value = model_averages[model].get(metric)
            if value is not None:
                print(f"{value:>12.2f} ", end="")
            else:
                print(f"{'N/A':>12} ", end="")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze consolidated evaluation metrics"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results_debug/consolidated_metrics.json",
        help="Path to consolidated metrics JSON file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "task", "model", "best", "avg"],
        default="all",
        help="Analysis mode: all, task (by task), model (by model), best (best performers), avg (averages)"
    )
    
    args = parser.parse_args()
    
    # Get absolute path
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent
    input_path = workspace_root / args.input
    
    print(f"Loading metrics from: {input_path}")
    data = load_consolidated_metrics(str(input_path))
    
    metrics = data['metrics']
    summary = data['summary']
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Models: {summary['unique_models']}")
    print(f"Tasks: {summary['unique_tasks']}")
    print(f"Model List: {', '.join(summary['models'])}")
    print(f"Task List: {', '.join(summary['tasks'])}")
    
    by_task = get_metrics_by_task(metrics)
    by_model = get_metrics_by_model(metrics)
    
    # Run requested analyses
    if args.mode in ["all", "task"]:
        print_task_comparison(by_task)
    
    if args.mode in ["all", "model"]:
        print_model_comparison(by_model)
    
    if args.mode in ["all", "best"]:
        find_best_performers(by_task)
    
    if args.mode in ["all", "avg"]:
        calculate_average_performance(by_model)
    
    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
