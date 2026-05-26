import argparse
import glob
import json
import os
import time
from typing import Dict, Iterable, List, Optional

from openai import OpenAI
from tqdm import tqdm


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


NUM_SECONDS_TO_SLEEP = 0.5
MAX_RETRIES = 12


def get_eval_score(
    client: OpenAI,
    content: str,
    max_tokens: int,
    model: str,
) -> Dict:
    schema = {
        "name": "score_pair",
        "schema": {
            "type": "object",
            "properties": {
                "assistant_1": {"type": "number", "description": "Score 0–10"},
                "assistant_2": {"type": "number", "description": "Score 0–10"},
                "reasoning": {"type": "string", "description": "Brief rationale"},
            },
            "required": ["assistant_1", "assistant_2", "reasoning"],
            "additionalProperties": False
        },
        "strict": True
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": 'You are a helpful and precise assistant for checking the quality of the answer.'},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,
                seed=42,
                max_tokens=max_tokens,
                response_format={"type": "json_schema", "json_schema": schema},
            )
            msg = resp.choices[0].message
            try:
                return msg.parsed
            except AttributeError:
                return json.loads(msg.content)
        except Exception as e:
            print(f"[Attempt {attempt}/{MAX_RETRIES}] Error: {e}")
            if attempt == MAX_RETRIES:
                print("❌ Reached max retries — returning empty result.")
                return {}
            time.sleep(NUM_SECONDS_TO_SLEEP * attempt)


def build_caption_eval_prompt(
    context_text: str,
    question: str,
    reference_caption: str,
    caption: str,
    role: str,
    prompt_text: str,
) -> str:
    return (
        f"[Context]\n{context_text}\n\n"
        f"[Question]\n{question}\n\n"
        f"[{role} 1]\n{reference_caption}\n\n[End of {role} 1]\n\n"
        f"[{role} 2]\n{caption}\n\n[End of {role} 2]\n\n"
        f"[System]\n{prompt_text}\n\n"
    )


def iter_result_files(results_dir: str, language: Optional[str], file_pattern: str = "*.json") -> Iterable[str]:
    if language:
        lang_dir = os.path.join(results_dir, language)
        if not os.path.isdir(lang_dir):
            raise FileNotFoundError(f"Language directory not found: {lang_dir}")
        yield from sorted(glob.glob(os.path.join(lang_dir, file_pattern)))
        return

    for lang_dir in sorted(glob.glob(os.path.join(results_dir, "*"))):
        if os.path.isdir(lang_dir):
            yield from sorted(glob.glob(os.path.join(lang_dir, file_pattern)))


def load_existing_reviews(output_path: str) -> Dict[int, Dict]:
    if not os.path.isfile(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {r.get("question_id"): r for r in records if "question_id" in r}


def load_context_map(context_jsonl_path: str) -> Dict[str, Dict]:
    if not os.path.isfile(context_jsonl_path):
        raise FileNotFoundError(f"Context file not found: {context_jsonl_path}")
    context_list = load_jsonl(context_jsonl_path)
    return {context["image"]: context for context in context_list if "image" in context}


def load_question_map(question_jsonl_path: str) -> Dict[int, Dict]:
    if not os.path.isfile(question_jsonl_path):
        raise FileNotFoundError(f"Question file not found: {question_jsonl_path}")
    question_list = load_jsonl(question_jsonl_path)
    return {q["question_id"]: q for q in question_list if "question_id" in q}


def load_rule_map(rule_path: str) -> Dict[str, Dict]:
    if not os.path.isfile(rule_path):
        raise FileNotFoundError(f"Rule file not found: {rule_path}")
    return load_json(rule_path)


def evaluate_results_file(
    client: OpenAI,
    results_path: str,
    output_path: str,
    judge_model: str,
    max_tokens: int,
    question_map: Dict[int, Dict],
    rule_map: Dict[str, Dict],
    context_map: Dict[str, Dict],
    max_samples: Optional[int] = None,
) -> None:
    done_marker = f"{output_path}.done"
    if os.path.isfile(done_marker):
        print(f"Skipping {results_path} because {done_marker} exists.")
        return

    data = load_json(results_path)
    existing = load_existing_reviews(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    review_file = open(output_path, "a", encoding="utf-8")

    processed = 0
    for item in tqdm(data, desc=f"Scoring {os.path.basename(results_path)}"):
        question_id = item.get("question_id")
        if question_id in existing:
            continue

        question = question_map.get(question_id, {})
        image_name = question.get("image", "")
        image_id = item.get("image_id", "") or image_name.replace(".jpg", "")
        base_image = f"{image_name.split('_')[0]}.jpg" if image_name else ""
        context_item = context_map.get(base_image, {})
        context_caption = context_item.get("caption", "")
        if isinstance(context_caption, list):
            context_text = "\n".join(context_caption)
        else:
            context_text = context_caption

        category = question.get("category")
        if category is None:
            raise ValueError(f"Missing category for question_id={question_id}")
        category_key = f"llava_bench_{category}"
        if category_key not in rule_map:
            raise ValueError(f"Visual QA category not found in rule file: {category_key}.")
        rule = rule_map[category_key]
        role = rule["role"]
        prompt_text = rule["prompt"]

        prompt = build_caption_eval_prompt(
            context_text=context_text,
            question=question.get("text", item.get("question", "")),
            reference_caption=item.get("reference_caption", ""),
            caption=item.get("caption", ""),
            role=role,
            prompt_text=prompt_text,
        )
        review = get_eval_score(
            client,
            prompt,
            max_tokens=max_tokens,
            model=judge_model,
        )
        score_1 = review.get("assistant_1", -1)
        score_2 = review.get("assistant_2", -1)
        reasoning = review.get("reasoning", "")

        record = {
            "question_id": question_id,
            "image_id": image_id,
            "category": category_key,
            "assistant_1": score_1,
            "assistant_2": score_2,
            "reasoning": reasoning,
            "tuple": [score_1, score_2],
            "content": review,
        }
        review_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        review_file.flush()

        processed += 1
        if max_samples is not None and processed >= max_samples:
            break

    review_file.close()

    if max_samples is None or processed >= len(data):
        with open(done_marker, "w", encoding="utf-8") as f:
            f.write("done\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-based multilingual LLaVA-Bench evaluation.")
    parser.add_argument(
        "--results-dir",
        default="/leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/results_debug/eval_multilingual_llavabench",
        help="Directory with per-language JSON outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="/leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/results_debug/eval_multilingual_llavabench_scores",
        help="Directory to write GPT scores (jsonl).",
    )
    parser.add_argument(
        "--context-dir",
        default="/leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/playground/eval_data/multilingual-llava-bench-in-the-wild_reformatted_correct",
        help="Directory containing per-language context.jsonl files.",
    )
    parser.add_argument(
        "--question-dir",
        default="/leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/playground/eval_data/multilingual-llava-bench-in-the-wild_reformatted_correct",
        help="Directory containing per-language question.jsonl files.",
    )
    parser.add_argument(
        "--rule-path",
        default="/leonardo_work/EUHPC_R04_192/fmohamma/LLaVA-NeXT/playground/eval_data/rule.json",
        help="Rule JSON path for prompt templates.",
    )
    parser.add_argument(
        "--file-pattern",
        default="*.json",
        help="Glob pattern to match result files within each language directory.",
    )
    parser.add_argument("--language", default=None, help="Evaluate only a single language folder.")
    parser.add_argument("--judge-model", default="openai/gpt-4.1-nano", help="OpenAI model for judging.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum output tokens for the judge.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per file.")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY).")

    args = parser.parse_args()

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided. Use --openai-api-key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",)

    results_dir = os.path.expanduser(args.results_dir)
    output_dir = os.path.expanduser(args.output_dir)
    context_dir = os.path.expanduser(args.context_dir)
    question_dir = os.path.expanduser(args.question_dir)
    rule_map = load_rule_map(os.path.expanduser(args.rule_path))
    os.makedirs(output_dir, exist_ok=True)

    for results_path in iter_result_files(results_dir, args.language, args.file_pattern):
        lang_name = os.path.basename(os.path.dirname(results_path))
        context_path = os.path.join(context_dir, lang_name, "context.jsonl")
        context_map = load_context_map(context_path)
        question_path = os.path.join(question_dir, lang_name, "question.jsonl")
        question_map = load_question_map(question_path)
        base_name = os.path.splitext(os.path.basename(results_path))[0]
        lang_output_dir = os.path.join(output_dir, lang_name)
        os.makedirs(lang_output_dir, exist_ok=True)
        output_path = os.path.join(lang_output_dir, f"{base_name}.gpt_eval.jsonl")
        print(f"Evaluating {results_path} -> {output_path}")
        evaluate_results_file(
            client=client,
            results_path=results_path,
            output_path=output_path,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            question_map=question_map,
            rule_map=rule_map,
            context_map=context_map,
            max_samples=args.max_samples,
        )
