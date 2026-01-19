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
from open_flamingo.eval.coco_metric import compute_cider, postprocess_captioning_generation
from open_flamingo.eval.eval_datasets import CaptionDataset


def remove_eot_token(text: str) -> str:
    """Remove any EOT (End of Turn) style tokens dynamically."""
    text = re.sub(r"<\|[a-z_]*end[a-z_]*\|>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[a-z_]*id\|>", "", text, flags=re.IGNORECASE)
    return text.strip()


def build_prompt(base_prompt: str, model) -> str:
    if getattr(model.config, "mm_use_im_start_end", False):
        return f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{base_prompt}"
    return f"{DEFAULT_IMAGE_TOKEN}\n{base_prompt}"


def resolve_paths(args) -> Tuple[str, str, str]:
    """Return (image_train_dir, image_val_dir, annotations_json) for the chosen dataset."""
    if args.dataset == "coco":
        return args.coco_train_image_dir, args.coco_val_image_dir, args.coco_annotations_json
    if args.dataset == "flickr":
        # Flickr uses a single image root; pass as train, keep val None for CaptionDataset
        return args.flickr_image_dir, None, args.flickr_annotations_json
    raise ValueError(f"Unsupported dataset {args.dataset}")


def resolve_split_json(args) -> str:
    if args.dataset == "coco":
        return args.coco_karpathy_json
    if args.dataset == "flickr":
        return args.flickr_karpathy_json
    raise ValueError(f"Unsupported dataset {args.dataset}")


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

    print(f"Using conversation mode: {conv_mode}")

    conv = conv_templates[conv_mode].copy()
    roles = ("user", "assistant") if "mpt" in model_name.lower() else conv.roles

    image_train_dir, image_val_dir, annotations_json = resolve_paths(args)
    karpathy_json = resolve_split_json(args)

    dataset = CaptionDataset(
        image_train_dir_path=image_train_dir,
        image_val_dir_path=image_val_dir,
        annotations_path=karpathy_json,
        is_train=False,
        dataset_name=args.dataset,
    )

    predictions = []

    for idx in range(len(dataset)):
        if args.sample_size is not None and idx >= args.sample_size:
            break

        sample = dataset[idx]
        image: Image.Image = sample["image"].convert("RGB")
        image_id = sample["image_id"] if args.dataset == "flickr" else int(sample["image_id"])

        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].cuda()
        image_tensor = image_tensor.to(dtype=model.dtype)

        prompt = build_prompt(args.prompt, model)

        new_conv = conv.copy()
        new_conv.append_message(roles[0], prompt)
        new_conv.append_message(roles[1], None)

        full_prompt = new_conv.get_prompt()
        input_ids = tokenizer_image_token(
            full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        stop_str = new_conv.sep if new_conv.sep_style != SeparatorStyle.TWO else new_conv.sep2
        if hasattr(new_conv, "stop_str") and new_conv.stop_str:
            stop_str = new_conv.stop_str
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
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

        caption = postprocess_captioning_generation(raw_output)
        caption = remove_eot_token(caption).replace(stop_str, "").strip()
        predictions.append({"image_id": image_id, "caption": caption})

        if args.debug:
            print(f"Processed image_id {image_id}: {caption}")

    output_path = args.output or f"{args.dataset}_predictions.json"
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    if annotations_json is None:
        print(f"Saved predictions to {output_path} (CIDEr not computed; missing annotations json)")
        return

    metrics = compute_cider(result_path=output_path, annotations_path=annotations_json)
    metrics = {key: value * 100.0 for key, value in metrics.items()}
    cider = metrics.get("CIDEr", 0.0)
    metrics["CIDEr"] = cider
    
    elapsed_time = time.time() - start_time
    metrics["runtime_seconds"] = elapsed_time
    
    print(f"CIDEr: {cider:.2f}")
    print(f"Runtime: {elapsed_time:.2f} seconds")
    print(f"Saved predictions to {output_path}")

    metrics_output_path = output_path.replace(".json", "_metrics.json")
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["coco", "flickr"], default="coco")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--prompt", type=str, default="Provide a short caption for this image.")
    # COCO paths
    parser.add_argument("--coco-train-image-dir", type=str, help="Path to COCO train2014 images.")
    parser.add_argument("--coco-val-image-dir", type=str, help="Path to COCO val2014 images.")
    parser.add_argument("--coco-karpathy-json", type=str, help="Path to Karpathy split JSON (dataset_coco.json).")
    parser.add_argument(
        "--coco-annotations-json", type=str, help="Path to COCO captions annotations JSON (captions_val2014.json)."
    )
    # Flickr paths
    parser.add_argument("--flickr-image-dir", type=str, help="Path to flickr30k_images directory.")
    parser.add_argument("--flickr-karpathy-json", type=str, help="Path to dataset_flickr30k.json.")
    parser.add_argument(
        "--flickr-annotations-json",
        type=str,
        help="Path to dataset_flickr30k_coco_style.json (COCO-style annotations).",
    )
    parser.add_argument("--output", type=str, default=None, help="Where to save prediction JSON.")
    args = parser.parse_args()

    if args.dataset == "coco":
        required = [args.coco_train_image_dir, args.coco_val_image_dir, args.coco_karpathy_json, args.coco_annotations_json]
        names = ["--coco-train-image-dir", "--coco-val-image-dir", "--coco-karpathy-json", "--coco-annotations-json"]
    else:
        required = [args.flickr_image_dir, args.flickr_karpathy_json, args.flickr_annotations_json]
        names = ["--flickr-image-dir", "--flickr-karpathy-json", "--flickr-annotations-json"]

    for val, name in zip(required, names):
        if val is None:
            raise ValueError(f"Argument {name} is required for dataset {args.dataset}")

    main(args)
