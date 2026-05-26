"""Aggregate evaluation metrics into a single JSON mapping.

Given an evaluation results directory (e.g.
`Evaluations/.../checkpoints/Temp_0.0/llava-next-apertus-8b-finetune-full/Encoder_none`),
this script scans JSON result files, extracts the mean metric per dataset, and
writes a consolidated JSON such as `{"COCO": 32.36, "FLICKR30": 35.49, ...}`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


IGNORED_KEYS = {"model", "attack"}


def _coerce_number(value: Any) -> Optional[float]:
	"""Return a float if the value looks numeric, otherwise None."""

	if value is None:
		return None
	if isinstance(value, (int, float)):
		# Filter out NaN values that show up in some eval dumps.
		return None if (isinstance(value, float) and math.isnan(value)) else float(value)
	try:
		parsed = float(value)
	except (TypeError, ValueError):
		return None
	return parsed


def _mean_of_list(values: Iterable[Any]) -> Optional[float]:
	numeric = [_coerce_number(v) for v in values]
	numeric = [v for v in numeric if v is not None]
	if not numeric:
		return None
	return sum(numeric) / len(numeric)


def _pick_from_dict(dct: Dict[str, Any]) -> Optional[float]:
	"""Pick a representative metric from a dict of metrics.

	Preference order matches common eval outputs: cider > accuracy/acc > success_rate,
	then the first numeric value encountered.
	"""

	for key in ("cider", "accuracy", "acc", "success_rate"):
		if key in dct:
			val = _coerce_number(dct[key])
			if val is not None:
				return val
	for val in dct.values():
		coerced = _coerce_number(val)
		if coerced is not None:
			return coerced
	return None


def extract_mean(entry: Any) -> Optional[float]:
	"""Extract a mean value from a dataset entry.

	Handles structures like:
	- entry["mean"] is a number
	- entry["mean"] is a list of numbers
	- entry["mean"] is a dict of named metrics
	Falls back to averaging the "trials" field when mean is absent.
	"""

	if not isinstance(entry, dict):
		return None

	if "mean" in entry:
		mean_val = entry["mean"]
		if isinstance(mean_val, dict):
			return _pick_from_dict(mean_val)
		if isinstance(mean_val, list):
			return _mean_of_list(mean_val)
		return _coerce_number(mean_val)

	if "trials" in entry:
		trials = entry["trials"]
		if isinstance(trials, dict):
			return _pick_from_dict(trials)
		if isinstance(trials, list):
			return _mean_of_list(trials)
	return None


def extract_dataset_means(data: Any) -> Dict[str, float]:
	"""Extract means for all dataset keys in a single result JSON payload.

	Accepts dict payloads (preferred) or lists of dict payloads produced by some evals.
	"""

	# Some eval dumps store a list of payloads; merge them.
	if isinstance(data, list):
		merged: Dict[str, float] = {}
		for item in data:
			if isinstance(item, dict):
				merged.update(extract_dataset_means(item))
		return merged

	if not isinstance(data, dict):
		return {}

	results: Dict[str, float] = {}
	for key, value in data.items():
		if key in IGNORED_KEYS:
			continue
		# Expect a list of per-shot entries; take the first by convention.
		entries: List[Any]
		if isinstance(value, list):
			entries = value
		else:
			entries = [value]
		mean_val = None
		for entry in entries:
			mean_val = extract_mean(entry)
			if mean_val is not None:
				break
		if mean_val is not None:
			results[key.upper()] = mean_val
	return results


def gather_metrics(results_dir: Path, recursive: bool = True) -> Dict[str, float]:
	"""Walk the results directory and consolidate dataset means."""

	metrics: Dict[str, float] = {}
	pattern = "**/*.json" if recursive else "*.json"
	for json_path in results_dir.glob(pattern):
		if json_path.is_dir():
			continue
		try:
			with json_path.open("r", encoding="utf-8") as f:
				payload = json.load(f)
		except Exception:
			# Skip files that are not valid JSON payloads for eval outputs.
			continue
		dataset_means = extract_dataset_means(payload)
		# Later files overwrite earlier ones for the same dataset key.
		metrics.update(dataset_means)
	return metrics


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Aggregate evaluation metrics into a single JSON mapping dataset name to mean value.",
	)
	parser.add_argument(
		"results_dir",
		type=Path,
		help="Path to the evaluation results directory containing dataset JSON files.",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help="Optional path to write aggregated metrics JSON. Defaults to <results_dir>/aggregated_metrics.json.",
	)
	parser.add_argument(
		"--no-recursive",
		action="store_true",
		help="Disable recursive search; only read JSON files directly under results_dir.",
	)
	args = parser.parse_args()

	results_dir: Path = args.results_dir
	if not results_dir.exists():
		raise FileNotFoundError(f"Results directory not found: {results_dir}")

	metrics = gather_metrics(results_dir, recursive=not args.no_recursive)
	output_path = args.output or results_dir / "aggregated_metrics.json"

	with output_path.open("w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	# Also print to stdout for quick inspection.
	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()
