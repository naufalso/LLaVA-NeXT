import argparse
import json
import os
import re
import time
from typing import Tuple

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
from open_flamingo.eval.eval_datasets import VQADataset
from open_flamingo.eval.ok_vqa_utils import postprocess_ok_vqa_generation
from open_flamingo.eval.vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation


def remove_eot_token(text: str) -> str:
    """Remove any EOT (End of Turn) style tokens dynamically."""
    # Handle tokens with spaces: < |assistant_end| >, < |eot_id| >, < |im_end| >
    text = re.sub(r"<\s*\|[a-z_]*end[a-z_]*\|\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*\|[a-z_]*id\|\s*>", "", text, flags=re.IGNORECASE)
    # Handle tokens without spaces: <|assistant_end|>, <|eot_id|>, <|im_end|>
    text = re.sub(r"<\|[a-z_]*end[a-z_]*\|>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[a-z_]*id\|>", "", text, flags=re.IGNORECASE)
    # Handle </s> and < /s > tokens
    text = re.sub(r"<\s*/\s*s\s*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def build_prompt(instruction: str, question: str, model) -> str:
    base_prompt = f"{instruction}\nQuestion: {question}\nAnswer:"
    if getattr(model.config, "mm_use_im_start_end", False):
        return f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{base_prompt}"
    return f"{DEFAULT_IMAGE_TOKEN}\n{base_prompt}"


def resolve_paths(args) -> Tuple[str, str, str]:
    if args.dataset == "vqav2":
        return args.vqav2_image_dir, args.vqav2_questions_json, args.vqav2_annotations_json
    if args.dataset == "ok_vqa":
        return args.okvqa_image_dir, args.okvqa_questions_json, args.okvqa_annotations_json
    if args.dataset == "vizwiz":
        return args.vizwiz_image_dir, args.vizwiz_questions_json, args.vizwiz_annotations_json
    if args.dataset == "textvqa":
        return args.textvqa_image_dir, args.textvqa_questions_json, args.textvqa_annotations_json
    raise ValueError(f"Unsupported dataset {args.dataset}")


def select_postprocess_fn(dataset: str):
    if dataset == "ok_vqa":
        return postprocess_ok_vqa_generation
    return postprocess_vqa_generation


def main(args):
    disable_torch_init()
    start_time = time.time()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        torch_dtype=args.dtype,
    )

    if "llama3" in model_name.lower():
        conv_mode = "llava_llama_3"
    elif "qwen25" in model_name.lower():
        conv_mode = "qwen_2_5"
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
            f"[WARNING] the auto inferred conversation mode is {conv_mode}, while `--conv-mode` is {args.conv_mode}, using {args.conv_mode}"
        )
        conv_mode = args.conv_mode
    elif args.conv_mode is not None:
        conv_mode = args.conv_mode

    conv = conv_templates[conv_mode].copy()
    roles = ("user", "assistant") if "mpt" in model_name.lower() else conv.roles

    image_dir, questions_json, annotations_json = resolve_paths(args)
    postprocess_fn = select_postprocess_fn(args.dataset)

    dataset = VQADataset(
        image_dir_path=image_dir,
        question_path=questions_json,
        annotations_path=annotations_json,
        is_train=False,
        dataset_name=args.dataset,
    )

    predictions = []

    for idx in range(len(dataset)):
        if args.sample_size is not None and idx >= args.sample_size:
            break

        sample = dataset[idx]
        image: Image.Image = sample["image"].convert("RGB")
        question = sample["question"]
        question_id = sample["question_id"]
        # Get ground truth answers if available
        ground_truth = sample.get("answers", [])

        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].cuda()
        image_tensor = image_tensor.to(dtype=model.dtype)

        prompt = build_prompt(args.prompt, question, model)

        new_conv = conv.copy()
        new_conv.append_message(roles[0], prompt)
        new_conv.append_message(roles[1], None)

        full_prompt = new_conv.get_prompt()
        input_ids = tokenizer_image_token(
            full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        # Create attention mask
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
                pad_token_id=tokenizer.pad_token_id,
                images=image_tensor,
                do_sample=args.temperature > 0.0,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        if torch.equal(output_ids[0, : input_ids.shape[1]], input_ids[0]):
            raw_output = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        else:
            raw_output = tokenizer.decode(output_ids[0]).strip()

        answer = postprocess_fn(raw_output)
        answer = remove_eot_token(answer).replace(stop_str, "").strip()
        
        pred_entry = {"question_id": question_id, "question": question, "answer": answer}
        if ground_truth:
            pred_entry["ground_truth"] = list(set(ground_truth)) # remove duplicates
        predictions.append(pred_entry)

        if args.debug:
            gt_str = f" | GT: {ground_truth}" if ground_truth else ""
            print(f"Processed question_id {question_id}: {answer}{gt_str}")

    output_path = args.output or f"{args.dataset}_vqa_predictions.json"
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    if annotations_json is None:
        print(f"Saved predictions to {output_path} (accuracy not computed; missing annotations json)")
        return

    accuracy = compute_vqa_accuracy(
        result_json_path=output_path,
        question_json_path=questions_json,
        annotation_json_path=annotations_json,
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Runtime: {elapsed_time:.2f} seconds")
    print(f"Saved predictions to {output_path}")

    metrics_output_path = output_path.replace(".json", "_metrics.json")
    with open(metrics_output_path, "w") as f:
        json.dump({"accuracy": accuracy, "runtime_seconds": elapsed_time}, f, indent=2)
    print(f"Saved metrics to {metrics_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["vqav2", "ok_vqa", "vizwiz", "textvqa"], default="vqav2")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--prompt", type=str, default="Answer the question using a single word or phrase.")

    # VQAv2 paths
    parser.add_argument("--vqav2-image-dir", type=str, help="Path to VQAv2 image directory (val2014/test2015).")
    parser.add_argument("--vqav2-questions-json", type=str, help="Path to VQAv2 questions JSON.")
    parser.add_argument("--vqav2-annotations-json", type=str, help="Path to VQAv2 annotations JSON.")

    # OK-VQA paths
    parser.add_argument("--okvqa-image-dir", type=str, help="Path to OK-VQA image directory (train2014/val2014).")
    parser.add_argument("--okvqa-questions-json", type=str, help="Path to OK-VQA questions JSON.")
    parser.add_argument("--okvqa-annotations-json", type=str, help="Path to OK-VQA annotations JSON.")

    # VizWiz paths
    parser.add_argument("--vizwiz-image-dir", type=str, help="Path to VizWiz image directory.")
    parser.add_argument("--vizwiz-questions-json", type=str, help="Path to VizWiz questions JSON.")
    parser.add_argument("--vizwiz-annotations-json", type=str, help="Path to VizWiz annotations JSON.")

    # TextVQA paths
    parser.add_argument("--textvqa-image-dir", type=str, help="Path to TextVQA image directory.")
    parser.add_argument("--textvqa-questions-json", type=str, help="Path to TextVQA questions JSON.")
    parser.add_argument("--textvqa-annotations-json", type=str, help="Path to TextVQA annotations JSON.")

    parser.add_argument("--output", type=str, default=None, help="Where to save prediction JSON.")
    args = parser.parse_args()

    if args.dataset == "vqav2":
        required = [args.vqav2_image_dir, args.vqav2_questions_json]
        names = ["--vqav2-image-dir", "--vqav2-questions-json"]
    elif args.dataset == "ok_vqa":
        required = [args.okvqa_image_dir, args.okvqa_questions_json]
        names = ["--okvqa-image-dir", "--okvqa-questions-json"]
    elif args.dataset == "vizwiz":
        required = [args.vizwiz_image_dir, args.vizwiz_questions_json]
        names = ["--vizwiz-image-dir", "--vizwiz-questions-json"]
    else:
        required = [args.textvqa_image_dir, args.textvqa_questions_json]
        names = ["--textvqa-image-dir", "--textvqa-questions-json"]

    for val, name in zip(required, names):
        if val is None:
            raise ValueError(f"Argument {name} is required for dataset {args.dataset}")

    main(args)
