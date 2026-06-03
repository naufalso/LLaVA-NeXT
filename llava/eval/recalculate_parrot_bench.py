#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_model_output(
    output_text: str,
    valid_letters: Sequence[str],
) -> Optional[str]:
    """
    Robust multilingual parser for multiple-choice model outputs.

    Supports:
      - ANSWER: A
      - ANSWER: [A]
      - ANSWER: A.
      - A
      - A.
      - A) text
      - الإجابة: A
      - الجواب الصحيح هو B
      - 正确答案是：D
      - 答案：C
      - resposta correta é C
      - doğru cevap: B
      - правильный ответ: D
    """

    if not output_text:
        return None

    valid_letters = [
        unicodedata.normalize("NFKC", str(letter)).strip().upper()
        for letter in valid_letters
        if str(letter).strip()
    ]
    valid_letters = sorted(set(letter for letter in valid_letters if len(letter) == 1))

    if not valid_letters:
        return None

    letter_class = "".join(re.escape(letter) for letter in valid_letters)

    text = unicodedata.normalize("NFKC", output_text).strip()

    text = (
        text.replace("\u200e", "")
        .replace("\u200f", "")
        .replace("\ufeff", "")
    )

    if re.search(r"</think\s*>", text, flags=re.IGNORECASE):
        text = re.split(r"</think\s*>", text, flags=re.IGNORECASE)[-1].strip()

    special_token_re = re.compile(
        r"<\|[^|<>]+\|>|<\/?(?:s|think)>|<\/?(?:assistant|user|system)[^>]*>",
        flags=re.IGNORECASE,
    )
    text = special_token_re.sub("", text).strip()

    # 1. Requested benchmark format.
    answer_format_pattern = (
        rf"\bANSWER\s*[:：]\s*[\[\(\{{<\"'`«“‘]?\s*"
        rf"([{letter_class}])"
        rf"\s*[\]\)\}}>\"'`»”’]?\s*[\.\。]?"
    )

    matches = re.findall(answer_format_pattern, text, flags=re.IGNORECASE | re.UNICODE)
    if matches:
        return matches[-1].upper()

    # 2. Multilingual answer cues.
    cue_patterns = [
        # English
        r"final\s+answer",
        r"correct\s+answer",
        r"right\s+answer",
        r"selected\s+answer",
        r"answer",
        r"option",
        r"choice",
        r"selection",

        # Chinese
        r"正确答案",
        r"正確答案",
        r"最终答案",
        r"最終答案",
        r"最后答案",
        r"最後答案",
        r"正确选项",
        r"正確選項",
        r"答案",
        r"答",
        r"选项",
        r"選項",

        # Arabic
        r"الإجابة\s+الصحيحة",
        r"الاجابة\s+الصحيحة",
        r"الجواب\s+الصحيح",
        r"الإجابة\s+النهائية",
        r"الاجابة\s+النهائية",
        r"الاختيار\s+الصحيح",
        r"الخيار\s+الصحيح",
        r"الإجابة",
        r"الاجابة",
        r"إجابة",
        r"اجابة",
        r"الجواب",
        r"جواب",
        r"الخيار",
        r"اختيار",
        r"الاختيار",
        r"الحل",

        # Portuguese
        r"resposta\s+correta",
        r"resposta\s+final",
        r"op[cç][aã]o\s+correta",
        r"alternativa\s+correta",
        r"resposta",
        r"op[cç][aã]o",
        r"alternativa",
        r"letra",

        # Turkish
        r"do[gğ]ru\s+cevap",
        r"do[gğ]ru\s+yan[ıi]t",
        r"do[gğ]ru\s+se[cç]enek",
        r"do[gğ]ru\s+[sş][ıi]k",
        r"nihai\s+cevap",
        r"son\s+cevap",
        r"cevap",
        r"yan[ıi]t",
        r"se[cç]enek",
        r"[sş][ıi]k",

        # Russian
        r"правильн(?:ый|ого)?\s+ответ",
        r"верн(?:ый|ого)?\s+ответ",
        r"окончательн(?:ый|ого)?\s+ответ",
        r"правильн(?:ый|ого)?\s+вариант",
        r"ответ",
        r"вариант",
        r"выбор",
    ]

    cue_regex = "|".join(f"(?:{cue})" for cue in cue_patterns)

    cue_answer_pattern = (
        rf"(?:{cue_regex})"
        rf"[\s\S]{{0,100}}?"
        rf"(?:[:：=\-–—]|is|are|é|e|é a|é o|是|为|為|هو|هي|это|будет)?"
        rf"\s*"
        rf"[\[\(\{{<\"'`«“‘]?"
        rf"\s*([{letter_class}])"
        rf"\s*[\]\)\}}>\"'`»”’]?"
        rf"(?![A-Za-z0-9])"
    )

    matches = re.findall(cue_answer_pattern, text, flags=re.IGNORECASE | re.UNICODE)
    if matches:
        return matches[-1].upper()

    # 3. Exact single-letter output.
    exact_single_letter = re.match(
        rf"^\s*[\[\(\{{<\"'`«“‘]?\s*"
        rf"([{letter_class}])"
        rf"\s*[\]\)\}}>\"'`»”’]?"
        rf"\s*[\.\。]?\s*$",
        text,
        flags=re.IGNORECASE | re.UNICODE,
    )
    if exact_single_letter:
        return exact_single_letter.group(1).upper()

    # 4. Single option-like line.
    option_line_re = re.compile(
        rf"^\s*(?:[-*•]\s*)?"
        rf"(?:"
        rf"[\[\(\{{（]\s*([{letter_class}])\s*[\]\)\}}）]"
        rf"|"
        rf"([{letter_class}])\s*(?:[\.\)、）。:：\-]|\s+|$)"
        rf")",
        flags=re.IGNORECASE | re.UNICODE,
    )

    option_line_candidates = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = option_line_re.match(line)
        if match:
            letter = match.group(1) or match.group(2)
            option_line_candidates.append(letter.upper())

    if len(option_line_candidates) == 1:
        return option_line_candidates[0]

    # 5. Conservative tail fallback.
    tail_lines = [line.strip() for line in text.splitlines() if line.strip()][-3:]

    tail_option_lines = []
    for line in tail_lines:
        match = option_line_re.match(line)
        if match:
            letter = match.group(1) or match.group(2)
            tail_option_lines.append(letter.upper())

    if len(tail_option_lines) <= 1:
        tail = "\n".join(tail_lines)[-500:]

        marked_patterns = [
            rf"[\(（\[\{{]\s*([{letter_class}])\s*[\)）\]\}}]",
            rf"(?<![A-Za-z0-9])([{letter_class}])\s*(?=[\.\)、）。:：\-])",
            rf"(?<![A-Za-z0-9])[\[\(\{{<\"'`«“‘]?\s*"
            rf"([{letter_class}])"
            rf"\s*[\]\)\}}>\"'`»”’]?\s*[\.\。!！?؟？,，;；:：]*\s*$",
        ]

        marked = []
        for pattern in marked_patterns:
            marked.extend(
                match.group(1).upper()
                for match in re.finditer(pattern, tail, flags=re.IGNORECASE | re.UNICODE)
            )

        if marked:
            return marked[-1]

    return None


def is_metrics_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith("_metrics.json")
        or name.endswith("_metrics.jsonl")
        or name == "summary_metrics.json"
        or name == "summary_metrics.jsonl"
    )


def is_prediction_file(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False

    if is_metrics_file(path):
        return False

    return True


def load_prediction_file(path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[SKIP] Could not read {path}: {exc}")
        return None

    if not isinstance(data, list):
        print(f"[SKIP] Not a list JSON prediction file: {path}")
        return None

    if not data:
        return []

    if not all(isinstance(row, dict) for row in data):
        print(f"[SKIP] JSON list does not contain dict rows: {path}")
        return None

    return data


def get_output_text(row: Dict[str, Any]) -> str:
    clean_output = row.get("clean_output")
    if isinstance(clean_output, str) and clean_output.strip():
        return clean_output

    raw_output = row.get("raw_output")
    if isinstance(raw_output, str) and raw_output.strip():
        return raw_output

    return ""


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def recalc_file(
    path: Path,
    valid_letters: Sequence[str],
    update_predictions: bool,
    metrics_ext: str,
) -> Dict[str, Any]:
    rows = load_prediction_file(path)
    if rows is None:
        return {
            "path": str(path),
            "status": "skipped",
        }

    start_time = time.time()

    total = len(rows)
    parsed = 0
    correct = 0
    changed_answers = 0
    missing_ground_truth = 0
    format_errors = 0

    examples_unparsed = []
    examples_changed = []

    for row in rows:
        old_answer = normalize_answer(row.get("answer"))
        ground_truth = normalize_answer(row.get("ground_truth"))

        parsed_answer = parse_model_output(
            get_output_text(row),
            valid_letters=valid_letters,
        )

        new_answer = parsed_answer or ""

        if not ground_truth:
            missing_ground_truth += 1

        if new_answer:
            parsed += 1
        else:
            format_errors += 1
            if len(examples_unparsed) < 5:
                examples_unparsed.append(
                    {
                        "question_id": row.get("question_id"),
                        "clean_output": row.get("clean_output", ""),
                        "raw_output": row.get("raw_output", ""),
                    }
                )

        if new_answer and ground_truth and new_answer == ground_truth:
            correct += 1

        if old_answer != new_answer:
            changed_answers += 1
            if len(examples_changed) < 5:
                examples_changed.append(
                    {
                        "question_id": row.get("question_id"),
                        "old_answer": old_answer,
                        "new_answer": new_answer,
                    }
                )

        if update_predictions:
            row["answer"] = new_answer

    accuracy = (correct / total * 100.0) if total else 0.0
    answered_rate = (parsed / total * 100.0) if total else 0.0
    format_error_rate = (format_errors / total * 100.0) if total else 0.0

    elapsed = time.time() - start_time

    metrics = {
        "file": str(path),
        "accuracy": accuracy,
        "correct": correct,
        "answered": parsed,
        "answered_rate": answered_rate,
        "format_errors": format_errors,
        "format_error_rate": format_error_rate,
        "missing_ground_truth": missing_ground_truth,
        "total": total,
        "changed_answers": changed_answers,
        "runtime_seconds": elapsed,
    }

    if update_predictions:
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    if metrics_ext == "jsonl":
        metrics_path = path.with_name(path.stem + "_metrics.jsonl")
        with metrics_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    else:
        metrics_path = path.with_name(path.stem + "_metrics.json")
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    metrics["metrics_path"] = str(metrics_path)
    metrics["status"] = "ok"
    metrics["examples_unparsed"] = examples_unparsed
    metrics["examples_changed"] = examples_changed

    return metrics


def find_prediction_files(output_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in output_dir.rglob("*.json")
        if is_prediction_file(path)
    )


def write_summary(
    summary_path: Path,
    results: List[Dict[str, Any]],
) -> None:
    with summary_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalculate answers and metrics for multilingual MCQ prediction JSON files."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory to scan recursively.",
    )
    parser.add_argument(
        "--valid-letters",
        type=str,
        default="ABCDE",
        help="Valid answer letters. Default: ABCDE",
    )
    parser.add_argument(
        "--update-predictions",
        action="store_true",
        help="Overwrite prediction JSON files with newly parsed answer fields.",
    )
    parser.add_argument(
        "--metrics-ext",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help="Write metrics as _metrics.json or _metrics.jsonl. Default: json",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="summary_metrics.jsonl",
        help="Name of the summary JSONL file written under output-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run parsing and print summary without writing prediction or metrics files.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    valid_letters = list(args.valid_letters.strip().upper())
    prediction_files = find_prediction_files(output_dir)

    print(f"Found {len(prediction_files)} prediction JSON files under {output_dir}")

    results = []

    for path in prediction_files:
        if args.dry_run:
            result = recalc_file(
                path=path,
                valid_letters=valid_letters,
                update_predictions=False,
                metrics_ext=args.metrics_ext,
            )

            metrics_path = Path(result.get("metrics_path", ""))
            if metrics_path.exists():
                metrics_path.unlink()
        else:
            result = recalc_file(
                path=path,
                valid_letters=valid_letters,
                update_predictions=args.update_predictions,
                metrics_ext=args.metrics_ext,
            )

        results.append(result)

        if result.get("status") == "ok":
            print(
                f"[OK] {path} | "
                f"acc={result['accuracy']:.2f} | "
                f"answered={result['answered']}/{result['total']} | "
                f"format_error={result['format_error_rate']:.2f}% | "
                f"changed={result['changed_answers']}"
            )
        else:
            print(f"[SKIP] {path}")

    ok_results = [r for r in results if r.get("status") == "ok"]

    total = sum(r["total"] for r in ok_results)
    correct = sum(r["correct"] for r in ok_results)
    answered = sum(r["answered"] for r in ok_results)
    format_errors = sum(r["format_errors"] for r in ok_results)

    overall = {
        "status": "overall",
        "files": len(ok_results),
        "total": total,
        "correct": correct,
        "answered": answered,
        "format_errors": format_errors,
        "accuracy": correct / total * 100.0 if total else 0.0,
        "answered_rate": answered / total * 100.0 if total else 0.0,
        "format_error_rate": format_errors / total * 100.0 if total else 0.0,
    }

    results.append(overall)

    if not args.dry_run:
        summary_path = output_dir / args.summary_name
        write_summary(summary_path, results)
        print(f"Saved summary to {summary_path}")

    print(
        "\nOverall | "
        f"files={overall['files']} | "
        f"acc={overall['accuracy']:.2f} | "
        f"answered={overall['answered']}/{overall['total']} | "
        f"format_error={overall['format_error_rate']:.2f}%"
    )


if __name__ == "__main__":
    main()