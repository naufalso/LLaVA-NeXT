import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import pandas as pd


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_model_name(filename: str, language: str) -> str:
    """Extract model name from filename by removing language suffix and extension."""
    # Remove .gpt_eval.jsonl extension
    name = filename.replace(".gpt_eval.jsonl", "")
    # Remove language suffix
    if name.endswith(f"-{language}"):
        name = name[: -len(f"-{language}")]
    return name


def compute_scores(records: List[Dict]) -> Dict[str, float]:
    """Compute normalized mean and mean of assistant_2 from evaluation records."""
    if not records:
        return {"normalized_mean": 0.0, "assistant_2_mean": 0.0}
    
    assistant_1_scores = []
    assistant_2_scores = []
    
    for record in records:
        a1 = record.get("assistant_1", 0)
        a2 = record.get("assistant_2", 0)
        
        # Only include valid scores
        if a1 > 0 and a2 >= 0:
            assistant_1_scores.append(a1)
            assistant_2_scores.append(a2)
    
    if not assistant_1_scores:
        return {"normalized_mean": 0.0, "assistant_2_mean": 0.0}
    
    # Compute normalized mean (assistant_2 / assistant_1)
    normalized_scores = [a2 / a1 for a1, a2 in zip(assistant_1_scores, assistant_2_scores)]
    normalized_mean = sum(normalized_scores) / len(normalized_scores)
    
    # Compute mean of assistant_2
    assistant_2_mean = sum(assistant_2_scores) / len(assistant_2_scores)
    
    return {
        "normalized_mean": normalized_mean,
        "assistant_2_mean": assistant_2_mean,
    }


def aggregate_scores(scores_dir: str) -> tuple:
    """
    Aggregate scores from all models and languages.
    
    Returns:
        (normalized_df, assistant2_df): Two dataframes with model names as rows
                                        and languages as columns
    """
    # Structure: {model_name: {language: {normalized_mean: X, assistant_2_mean: Y}}}
    aggregated_data = defaultdict(lambda: defaultdict(dict))
    
    # Find all language directories
    language_dirs = sorted([
        d for d in os.listdir(scores_dir)
        if os.path.isdir(os.path.join(scores_dir, d))
    ])
    
    print(f"Found {len(language_dirs)} languages: {', '.join(language_dirs)}")
    
    # Process each language directory
    for language in language_dirs:
        lang_dir = os.path.join(scores_dir, language)
        jsonl_files = glob.glob(os.path.join(lang_dir, "*.gpt_eval.jsonl"))
        
        # Filter out .done files
        jsonl_files = [f for f in jsonl_files if not f.endswith(".done")]
        
        print(f"  {language}: {len(jsonl_files)} models")
        
        for jsonl_file in jsonl_files:
            filename = os.path.basename(jsonl_file)
            model_name = extract_model_name(filename, language)
            
            # Load evaluation records
            records = load_jsonl(jsonl_file)
            
            # Compute scores
            scores = compute_scores(records)
            
            aggregated_data[model_name][language] = scores
    
    # Convert to DataFrames
    # First, collect all unique models and languages
    all_models = sorted(aggregated_data.keys())
    all_languages = sorted(language_dirs)
    
    # Create normalized mean dataframe
    normalized_data = []
    assistant2_data = []
    
    for model in all_models:
        norm_row = {"model": model}
        ass2_row = {"model": model}
        
        for language in all_languages:
            if language in aggregated_data[model]:
                norm_row[language] = aggregated_data[model][language]["normalized_mean"]
                ass2_row[language] = aggregated_data[model][language]["assistant_2_mean"]
            else:
                norm_row[language] = None
                ass2_row[language] = None
        
        normalized_data.append(norm_row)
        assistant2_data.append(ass2_row)
    
    normalized_df = pd.DataFrame(normalized_data)
    assistant2_df = pd.DataFrame(assistant2_data)

    high_resource_langs = {"english", "chinese", "french", "spanish", "russian", "japanese"}
    low_resource_langs = {"arabic", "hindi", "bengali", "urdu"}

    def add_avg_columns(df: pd.DataFrame) -> pd.DataFrame:
        lang_cols = [c for c in df.columns if c != "model"]
        numeric_df = df[lang_cols].apply(pd.to_numeric, errors="coerce")

        high_cols = [c for c in lang_cols if c in high_resource_langs]
        low_cols = [c for c in lang_cols if c in low_resource_langs]

        df["Avg.H"] = numeric_df[high_cols].mean(axis=1, skipna=True) if high_cols else None
        df["Avg.L"] = numeric_df[low_cols].mean(axis=1, skipna=True) if low_cols else None
        df["Avg."] = numeric_df.mean(axis=1, skipna=True) if lang_cols else None
        return df

    normalized_df = add_avg_columns(normalized_df)
    assistant2_df = add_avg_columns(assistant2_df)
    
    return normalized_df, assistant2_df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LLaVA-Bench GPT evaluation scores across models and languages."
    )
    parser.add_argument(
        "--scores-dir",
        default="/leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/results_debug/eval_multilingual_llavabench_scores",
        help="Directory containing language subdirectories with GPT eval scores.",
    )
    parser.add_argument(
        "--output-dir",
        default="/leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/results_debug",
        help="Directory to save output CSV files.",
    )
    parser.add_argument(
        "--normalized-output",
        default="llavabench_normalized_scores.csv",
        help="Filename for normalized mean scores CSV.",
    )
    parser.add_argument(
        "--assistant2-output",
        default="llavabench_assistant2_scores.csv",
        help="Filename for assistant 2 mean scores CSV.",
    )
    
    args = parser.parse_args()
    
    scores_dir = os.path.expanduser(args.scores_dir)
    output_dir = os.path.expanduser(args.output_dir)
    
    if not os.path.isdir(scores_dir):
        raise FileNotFoundError(f"Scores directory not found: {scores_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Aggregating scores...")
    normalized_df, assistant2_df = aggregate_scores(scores_dir)
    
    # Save to CSV
    normalized_path = os.path.join(output_dir, args.normalized_output)
    assistant2_path = os.path.join(output_dir, args.assistant2_output)
    
    normalized_df.to_csv(normalized_path, index=False, float_format="%.4f")
    assistant2_df.to_csv(assistant2_path, index=False, float_format="%.4f")
    
    print(f"\n✓ Saved normalized mean scores to: {normalized_path}")
    print(f"✓ Saved assistant 2 mean scores to: {assistant2_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("NORMALIZED MEAN SCORES (Assistant 2 / Assistant 1)")
    print("="*80)
    print(normalized_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("ASSISTANT 2 MEAN SCORES")
    print("="*80)
    print(assistant2_df.to_string(index=False))


if __name__ == "__main__":
    main()
