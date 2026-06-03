import argparse
import json
import os
import re
import time
from typing import Dict, Optional, Sequence
from io import BytesIO
import base64

from tqdm import tqdm

import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.utils import disable_torch_init
from datasets import load_dataset


def remove_eot_token(text: str) -> str:
    """Remove common EOT or special end-of-turn tokens."""
    if not text:
        return ""

    # Handle tokens with spaces: < |assistant_end| >, < |eot_id| >, < |im_end| >
    text = re.sub(r"<\s*\|[a-z_]*end[a-z_]*\|\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*\|[a-z_]*id\|\s*>", "", text, flags=re.IGNORECASE)

    # Handle tokens without spaces: <|assistant_end|>, <|eot_id|>, <|im_end|>
    text = re.sub(r"<\|[a-z_]*end[a-z_]*\|>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[a-z_]*id\|>", "", text, flags=re.IGNORECASE)

    # Handle </s> and < /s > tokens
    text = re.sub(r"<\s*/\s*s\s*>", "", text, flags=re.IGNORECASE)

    return text.strip()


def build_prompt(
    question: str,
    choices: Dict[str, str],
    mm_use_im_start_end: bool,
    prompt_override: Optional[str] = None,
    direct_answer: bool = False,
) -> str:
    """Build the multimodal MCQ prompt."""

    choices_str = "\n".join([f"{letter}. {text}" for letter, text in choices.items()])
    letters = ", ".join(choices.keys())

    if prompt_override:
        instruction = prompt_override
        prompt = instruction.format(
            question=question,
            choices=choices_str,
            letters=letters,
        )
    elif direct_answer:
        instruction = """Answer the following multiple choice question.
Directly output the answer letter where [LETTER] is one of {letters}.
{question}
{choices}

Answer:"""
        prompt = instruction.format(
            question=question,
            choices=choices_str,
            letters=letters,
        )
    else:
        instruction = """Answer the following multiple choice question.
The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.
Think step by step before answering.
{question}
{choices}"""
        prompt = instruction.format(
            question=question,
            choices=choices_str,
            letters=letters,
        )

    if mm_use_im_start_end:
        return f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{prompt}"

    return f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"


def resolve_dataset(args):
    supported_languages = ["ar", "cn", "en", "pt", "ru", "tr"]

    if args.language not in supported_languages:
        raise ValueError(
            f"Unsupported language '{args.language}'. "
            f"Supported languages: {supported_languages}"
        )

    if args.dataset_type == "mmbench":
        data_file = f"{args.dataset_path}/mmbench/mmbench_dev_{args.language}.tsv"
    elif args.dataset_type == "mmmb":
        data_file = f"{args.dataset_path}/mmmb/mmmb_{args.language}.tsv"
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    return load_dataset("csv", data_files=data_file, delimiter="\t")["train"]


def parse_model_output(
    output_text: str,
    valid_letters: Sequence[str],
) -> Optional[str]:
    """
    Robustly parse the model output to extract an answer letter.

    Supported examples:
    - ANSWER: B
    - Answer: [B]
    - The answer is B
    - Option B
    - B.
    - B)
    - B
    """

    if not output_text:
        return None

    valid_letters = [letter.upper() for letter in valid_letters]
    if not valid_letters:
        return None

    letter_class = "".join(re.escape(letter) for letter in valid_letters)
    text = output_text.strip()

    # Strip hidden reasoning if the model emits it.
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Prefer matches near the end, since reasoning may mention wrong options earlier.
    patterns = [
        rf"\b(?:final\s+)?answer\s*[:\-]?\s*\[?\s*([{letter_class}])\s*\]?",
        rf"\b(?:the\s+)?(?:answer|option)\s+is\s*\[?\s*([{letter_class}])\s*\]?",
        rf"\boption\s*\[?\s*([{letter_class}])\s*\]?",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Starts with "A." or "A)"
    match = re.match(rf"^\s*([{letter_class}])[\.\)]", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Exact single-letter output
    match = re.match(rf"^\s*([{letter_class}])\s*$", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Last-resort fallback: a valid letter at the start.
    match = re.match(rf"^\s*([{letter_class}])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. "
            f"Use one of: {sorted(dtype_map.keys())}"
        )

    return dtype_map[dtype_name]


def get_model_device(model) -> torch.device:
    if hasattr(model, "device") and model.device is not None:
        return model.device

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_dtype(model, fallback_dtype: torch.dtype) -> torch.dtype:
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is not None:
        return model_dtype

    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return fallback_dtype


def collect_choices(sample) -> Dict[str, str]:
    choices = {}

    for letter in ["A", "B", "C", "D", "E"]:
        if letter in sample and sample[letter] is not None and str(sample[letter]).strip():
            choices[letter] = str(sample[letter]).strip()

    return choices


def decode_image_from_sample(image_base64_str: str) -> Image.Image:
    if "," in image_base64_str and image_base64_str.strip().startswith("data:"):
        image_base64_str = image_base64_str.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64_str)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def main(args):
    disable_torch_init()
    start_time = time.time()

    torch_dtype = get_torch_dtype(args.dtype)

    model_name = get_model_name_from_path(args.model_path)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        torch_dtype=torch_dtype,
    )

    model.eval()

    device = get_model_device(model)
    model_dtype = get_model_dtype(model, torch_dtype)

    if "llama3" in model_name.lower():
        conv_mode = "llava_llama_3"
    elif "qwen25" in model_name.lower():
        conv_mode = "qwen_2_5"
    elif "qwen" in model_name.lower():
        conv_mode = "qwen_2"
    elif "apertus" in model_name.lower():
        conv_mode = "apertus_ori"
    elif "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            f"[WARNING] Auto-inferred conversation mode is {conv_mode}, "
            f"but --conv-mode is {args.conv_mode}. Using {args.conv_mode}."
        )
        conv_mode = args.conv_mode
    elif args.conv_mode is not None:
        conv_mode = args.conv_mode

    if conv_mode not in conv_templates:
        raise ValueError(
            f"Conversation mode '{conv_mode}' not found in conv_templates. "
            f"Available modes: {list(conv_templates.keys())}"
        )

    conv = conv_templates[conv_mode].copy()
    roles = ("user", "assistant") if "mpt" in model_name.lower() else conv.roles

    dataset = resolve_dataset(args)

    if args.sample_eval:
        sample_size = min(500, len(dataset))
        dataset = dataset.select(range(sample_size))
    elif args.sample_size is not None:
        sample_size = min(args.sample_size, len(dataset))
        dataset = dataset.select(range(sample_size))

    predictions = []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    for sample in tqdm(dataset, desc="Evaluating"):
        image_base64_str = sample["image"]
        image = decode_image_from_sample(image_base64_str)

        question = sample["question"]
        question_id = sample["index"]

        choices = collect_choices(sample)
        if not choices:
            raise ValueError(f"No choices found for question_id {question_id}")

        ground_truth = sample.get("answer")
        if ground_truth is not None:
            ground_truth = str(ground_truth).strip().upper()

        if ground_truth not in choices:
            print("Warning: Ground truth answer '{}' not found in choices for question_id {}. Available choices: {}".format(
                ground_truth, question_id, list(choices.keys())
            ))
            print("Skipping this sample.")
            continue
            # raise ValueError(
            #     f"Unexpected ground truth answer '{ground_truth}' "
            #     f"for question_id {question_id}. Available choices: {list(choices.keys())}"
            # )

        image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ]
        image_tensor = image_tensor.to(device=device, dtype=model_dtype)

        prompt = build_prompt(
            question=question,
            choices=choices,
            mm_use_im_start_end=getattr(model.config, "mm_use_im_start_end", False),
            prompt_override=args.prompt,
            direct_answer=args.direct_answer,
        )

        new_conv = conv.copy()
        new_conv.append_message(roles[0], prompt)
        new_conv.append_message(roles[1], None)

        full_prompt = new_conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(device)

        attention_mask = torch.ones_like(input_ids)

        stop_str = new_conv.sep if new_conv.sep_style != SeparatorStyle.TWO else new_conv.sep2
        if hasattr(new_conv, "stop_str") and new_conv.stop_str:
            stop_str = new_conv.stop_str

        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                images=image_tensor,
                do_sample=args.temperature > 0.0,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        if torch.equal(output_ids[0, : input_ids.shape[1]], input_ids[0]):
            raw_output = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:],
                skip_special_tokens=False,
            ).strip()
        else:
            raw_output = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=False,
            ).strip()

        clean_output = remove_eot_token(raw_output).replace(stop_str, "").strip()
        answer = parse_model_output(clean_output, valid_letters=list(choices.keys()))

        if answer is None:
            answer = ""

        pred_entry = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "raw_output": raw_output,
            "clean_output": clean_output,
        }
        predictions.append(pred_entry)

        if args.debug:
            print(f"Processed question_id {question_id}: {pred_entry}")

    output_path = args.output or f"{args.dataset_type}_{args.language}.json"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    valid_predictions = [
        pred for pred in predictions
        if pred["answer"] and pred["ground_truth"]
    ]

    if valid_predictions:
        accuracy = (
            sum(
                1
                for pred in valid_predictions
                if pred["answer"].upper() == pred["ground_truth"].upper()
            )
            / len(predictions)
            * 100
        )
    else:
        accuracy = 0.0

    elapsed_time = time.time() - start_time

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Answered: {len(valid_predictions)} / {len(predictions)}")
    print(f"Runtime: {elapsed_time:.2f} seconds")
    print(f"Saved predictions to {output_path}")

    if output_path.endswith(".json"):
        metrics_output_path = output_path[:-5] + "_metrics.json"
    else:
        metrics_output_path = output_path + "_metrics.json"

    metrics = {
        "accuracy": accuracy,
        "answered": len(valid_predictions),
        "format_error_rate": len(valid_predictions) / len(predictions) * 100 if predictions else 0.0,
        "total": len(predictions),
        "runtime_seconds": elapsed_time,
    }

    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics to {metrics_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["mmbench", "mmmb"],
        required=True,
        help="Which dataset to evaluate on: 'mmbench' or 'mmmb'.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset directory containing mmbench/ and mmmb/ subdirectories.",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["ar", "cn", "en", "pt", "ru", "tr"],
        required=True,
        help="Language code for evaluation, for example: en, cn, ar.",
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--sample-eval",
        action="store_true",
        help="Use the first 500 samples, capped by dataset length.",
    )
    parser.add_argument(
        "--direct-answer",
        action="store_true",
        help="Directly output the answer letter without thinking before answering.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Optional prompt template override. "
            "Can use {question}, {choices}, and {letters}."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save prediction JSON.",
    )

    args = parser.parse_args()
    main(args)