import argparse
import json
import os
import re
import time

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
from open_flamingo.eval.coco_metric import postprocess_captioning_generation
from open_flamingo.eval.eval_datasets import CaptionDataset_MultiLingual


def remove_eot_token(text: str) -> str:
	"""Remove any EOT (End of Turn) style tokens dynamically."""
	text = re.sub(r"<\|[a-z_]*end[a-z_]*\|>", "", text, flags=re.IGNORECASE)
	text = re.sub(r"<\|[a-z_]*id\|>", "", text, flags=re.IGNORECASE)
	return text.strip()


def build_question_prompt(question: str, model) -> str:
	if getattr(model.config, "mm_use_im_start_end", False):
		return f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{question}"
	return f"{DEFAULT_IMAGE_TOKEN}\n{question}"


def get_caption_prompt(model, conv_mode, caption=None, language=None):
	qs = "Provide a short caption for this image."

	translations = {
		"arabic": "قدّم تعليقًا قصيرًا لهذه الصورة.",
		"bengali": "এই ছবির জন্য ছোট একটি ক্যাপশন লিখুন।",
		"chinese": "为这张图片写一段简短的说明。",
		"english": "Provide a short caption for this image.",
		"french": "Fournissez une légende courte pour cette image.",
		"hindi": "इस तस्वीर के लिए एक छोटा कैप्शन लिखें।",
		"japanese": "この画像の短いキャプションを付けてください。",
		"russian": "Напишите короткую подпись к этому изображению.",
		"spanish": "Proporcione un título breve para esta imagen.",
		"urdu": "اس تصویر کے لیے ایک مختصر تبصرہ دیں۔"
	}
	if language:
		if language in translations:
			qs = translations[language]
			print(f"Using language-specific prompt: {qs}")

	if model.config.mm_use_im_start_end:
		qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
	else:
		qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

	conv = conv_templates[conv_mode].copy()
	conv.append_message(conv.roles[0], qs)
	conv.append_message(conv.roles[1], caption)

	return conv


def load_multilingual_dataset(args) -> CaptionDataset_MultiLingual:
	return CaptionDataset_MultiLingual(
		image_val_dir_path=args.multilingual_llavabench_image_path,
		questions_path=args.multilingual_llavabench_questions_path,
		answers_path=args.multilingual_llavabench_answers_path,
		dataset_name="multilingual_llavabench",
	)


def main(args: argparse.Namespace) -> None:
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

	dataset = load_multilingual_dataset(args)

	# Use language-specific prompt if language is specified
	if args.language:
		print(f"Using language: {args.language}")

	predictions = []

	for idx in range(len(dataset)):
		if args.sample_size is not None and idx >= args.sample_size:
			break

		sample = dataset[idx]
		image: Image.Image = sample["image"].convert("RGB")
		image_id = sample["image_id"]
		question_id = sample["question_id"]
		question = sample["question_caption"]
		reference_caption = sample["caption"]

		image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].cuda()
		image_tensor = image_tensor.to(dtype=model.dtype)

		# Always use the question from the dataset (it's already in the correct language)
		prompt = build_question_prompt(question, model)
		new_conv = conv.copy()
		if conv_mode == "plain":
			new_conv.append_message("", prompt)
		else:
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

		predictions.append(
			{
				"image_id": image_id,
				"question_id": question_id,
				"question": question,
				"caption": caption,
				"reference_caption": reference_caption,
			}
		)

		if args.debug:
			print(f"Processed question_id {question_id}: {caption}")

	output_path = args.output or "multilingual_llavabench_predictions.json"
	if os.path.dirname(output_path):
		os.makedirs(os.path.dirname(output_path), exist_ok=True)

	with open(output_path, "w") as f:
		json.dump(predictions, f, indent=2, ensure_ascii=False)

	elapsed_time = time.time() - start_time
	print(f"Saved predictions to {output_path}")
	print(f"Runtime: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
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
	parser.add_argument(
		"--multilingual_llavabench_image_path",
		type=str,
		help="Path to the multilingual_llavabench images directory.",
		required=True,
	)
	parser.add_argument(
		"--multilingual_llavabench_questions_path",
		type=str,
		help="Path to the multilingual_llavabench questions (jsonl).",
		required=True,
	)
	parser.add_argument(
		"--multilingual_llavabench_answers_path",
		type=str,
		help="Path to the multilingual_llavabench answers (jsonl).",
		required=True,
	)
	parser.add_argument("--output", type=str, default=None, help="Where to save prediction JSON.")
	parser.add_argument(
		"--language",
		type=str,
		default=None,
		choices=["arabic", "bengali", "chinese", "english", "french", "hindi", "japanese", "russian", "spanish", "urdu"],
		help="Language for caption prompt. Supported: arabic, bengali, chinese, english, french, hindi, japanese, russian, spanish, urdu"
	)
	args = parser.parse_args()

	main(args)
