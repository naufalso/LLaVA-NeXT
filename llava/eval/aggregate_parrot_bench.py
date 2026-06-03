import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict


LANG_MAP = {
    "en": "en",
    "cn": "zh",
    "zh": "zh",
    "pt": "pt",
    "ar": "ar",
    "tr": "tr",
    "ru": "ru",
}

LANG_ORDER = ["en", "zh", "pt", "ar", "tr", "ru"]

BENCHMARKS = {
    "mmbench": "mmbench",
    "mmmb": "mmmb",
}

COLUMN_ORDER = [
    "model_name",

    "mmbench_en",
    "mmbench_zh",
    "mmbench_pt",
    "mmbench_ar",
    "mmbench_tr",
    "mmbench_ru",
    "mmbench_all",

    "mmmb_en",
    "mmmb_zh",
    "mmmb_pt",
    "mmmb_ar",
    "mmmb_tr",
    "mmmb_ru",
    "mmmb_all",
]


def load_metrics_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_benchmark_and_language(path: Path, root: Path):
    """
    Expected structure:
        root / mmbench / en / model_mmbench_en_metrics.json
        root / mmbench / cn / model_mmbench_cn_metrics.json
        root / mmmb / tr / model_mmmb_tr_metrics.json
    """
    parts = path.relative_to(root).parts

    benchmark = None
    language = None

    for part in parts:
        if part in BENCHMARKS:
            benchmark = BENCHMARKS[part]
        if part in LANG_MAP:
            language = LANG_MAP[part]

    return benchmark, language


def infer_model_name(path: Path, benchmark: str) -> str:
    """
    Removes suffixes like:
        _mmbench_en_metrics.json
        _mmbench_cn_metrics.json
        _mmmb_tr_metrics.json
    """
    name = path.name

    if name.endswith("_metrics.json"):
        name = name[: -len("_metrics.json")]

    possible_langs = set(LANG_MAP.keys()) | set(LANG_MAP.values())

    for lang in possible_langs:
        suffix = f"_{benchmark}_{lang}"
        if name.endswith(suffix):
            return name[: -len(suffix)]

    # Fallback for unexpected filename patterns
    name = re.sub(r"_(mmbench|mmmb)_[a-z]{2}$", "", name)
    return name


def add_average_columns(rows: dict):
    """
    Adds:
        mmbench_all = average of mmbench_en, zh, pt, ar, tr, ru
        mmmb_all = average of mmmb_en, zh, pt, ar, tr, ru

    Missing language scores are ignored.
    If no language score exists for a benchmark, the average is left blank.
    """
    for model_name, row in rows.items():
        for benchmark in ["mmbench", "mmmb"]:
            values = []

            for lang in LANG_ORDER:
                column = f"{benchmark}_{lang}"
                value = row.get(column)

                if value == "" or value is None:
                    continue

                try:
                    values.append(float(value))
                except ValueError:
                    continue

            avg_column = f"{benchmark}_all"

            if values:
                row[avg_column] = sum(values) / len(values)
            else:
                row[avg_column] = ""


def aggregate_metrics(input_dir: Path, metric_key: str = "accuracy"):
    rows = defaultdict(dict)

    metric_files = sorted(input_dir.rglob("*_metrics.json"))

    if not metric_files:
        raise FileNotFoundError(f"No *_metrics.json files found under: {input_dir}")

    for path in metric_files:
        benchmark, language = infer_benchmark_and_language(path, input_dir)

        if benchmark is None or language is None:
            print(f"[WARN] Skipping file because benchmark/language could not be inferred: {path}")
            continue

        metrics = load_metrics_json(path)

        if metric_key not in metrics:
            print(f"[WARN] Skipping file because '{metric_key}' is missing: {path}")
            continue

        model_name = infer_model_name(path, benchmark)
        column_name = f"{benchmark}_{language}"

        rows[model_name]["model_name"] = model_name
        rows[model_name][column_name] = metrics[metric_key]

    add_average_columns(rows)

    return rows


def write_csv(rows: dict, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMN_ORDER)
        writer.writeheader()

        for model_name in sorted(rows):
            row = {col: "" for col in COLUMN_ORDER}
            row.update(rows[model_name])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multilingual MMBench/MMMB metrics into a single CSV."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Root directory containing mmbench and mmmb result folders.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="aggregated_eval_metrics.csv",
        help="Path to save the aggregated CSV.",
    )
    parser.add_argument(
        "--metric_key",
        type=str,
        default="accuracy",
        help="Metric key to aggregate from each *_metrics.json file.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    rows = aggregate_metrics(input_dir=input_dir, metric_key=args.metric_key)
    write_csv(rows, output_csv)

    print(f"Saved aggregated CSV to: {output_csv}")
    print(f"Aggregated {len(rows)} models.")


if __name__ == "__main__":
    main()