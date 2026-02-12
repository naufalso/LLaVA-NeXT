"""
Focused training entrypoint for Apertus models.

This module trims down the generic `train.py` flow to the essentials needed
for Apertus-style chat formatting while retaining the same data loading,
quantization, and trainer integration points. The goal is clearer structure
and safer defaults for production use.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import pathlib
import random
import re
import time
import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from transformers.trainer_utils import IntervalStrategy
from PIL import Image, ImageFile
from packaging import version
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.mm_utils import process_anyres_image, process_highres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.train.llava_trainer import LLaVATrainer
from llava.utils import process_video_with_decord, process_video_with_pyav, rank0_print

# Reuse the model loader from the main trainer to avoid code drift.
from llava.train.train import get_model


torch.multiprocessing.set_sharing_strategy("file_system")
ImageFile.LOAD_TRUNCATED_IMAGES = True

local_rank: Optional[int] = None
IS_TOKENIZER_GE_014 = version.parse(transformers.__version__) >= version.parse("4.36.0")


# ---------------------------------------------------------------------------
# Argument dataclasses (kept largely in sync with train.py for compatibility)
# ---------------------------------------------------------------------------


@dataclass
class ModelArguments:
	model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
	model_class_name: Optional[str] = field(default=None)
	mm_tunable_parts: Optional[str] = field(default=None)
	version: Optional[str] = field(default="apertus")
	freeze_backbone: bool = field(default=False)
	tune_mm_mlp_adapter: bool = field(default=False)
	tune_mm_vision_resampler: bool = field(default=False)
	vision_tower: Optional[str] = field(default=None)
	vision_tower_pretrained: Optional[str] = field(default=None)
	unfreeze_mm_vision_tower: bool = field(default=False)
	unfreeze_language_model: bool = field(default=False)
	mm_vision_select_layer: Optional[int] = field(default=-1)
	pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
	mm_projector_type: Optional[str] = field(default="linear")
	mm_use_im_start_end: bool = field(default=False)
	mm_use_im_patch_token: bool = field(default=True)
	mm_patch_merge_type: Optional[str] = field(default="flat")
	mm_vision_select_feature: Optional[str] = field(default="patch")
	mm_resampler_type: Optional[str] = field(default=None)
	mm_mask_drop_mode: str = field(default="fixed")
	mm_mask_drop_skip_percentage: float = field(default=0.0)
	mm_mask_drop_ratio: float = field(default=0.25)
	mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
	mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
	mm_spatial_pool_stride: Optional[int] = field(default=None)
	mm_spatial_pool_mode: str = field(default="bilinear")
	mm_spatial_pool_out_channels: Optional[int] = field(default=None)
	mm_perceiver_depth: Optional[int] = field(default=3)
	mm_perceiver_latents: Optional[int] = field(default=32)
	mm_perceiver_ff_mult: Optional[float] = field(default=4)
	mm_perceiver_pretrained: Optional[str] = field(default=None)
	mm_qformer_depth: Optional[int] = field(default=3)
	mm_qformer_latents: Optional[int] = field(default=32)
	mm_qformer_pretrained: Optional[str] = field(default=None)
	rope_scaling_factor: Optional[float] = field(default=None)
	rope_scaling_type: Optional[str] = field(default=None)
	s2: Optional[bool] = field(default=False)
	s2_scales: Optional[str] = field(default="336,672,1008")
	use_pos_skipping: Optional[bool] = field(default=False)
	pos_skipping_range: Optional[int] = field(default=4096)
	mm_newline_position: Optional[str] = field(default="grid")
	delay_load: Optional[bool] = field(default=True)
	add_faster_video: Optional[bool] = field(default=False)
	faster_token_stride: Optional[int] = field(default=10)


@dataclass
class DataArguments:
	data_path: str = field(default=None, metadata={"help": "Path to the training data (instruction JSON/JSONL or YAML collection)."})
	lazy_preprocess: bool = False
	is_multimodal: bool = False
	early_mix_text: bool = False
	image_folder: Optional[str] = field(default=None)
	image_aspect_ratio: str = "square"
	image_grid_pinpoints: Optional[str] = field(default=None)
	image_crop_resolution: Optional[int] = field(default=None)
	image_split_resolution: Optional[int] = field(default=None)
	video_folder: Optional[str] = field(default=None)
	video_fps: Optional[int] = field(default=1)
	frames_upbound: Optional[int] = field(default=0)
	add_time_instruction: Optional[bool] = field(default=False)
	force_sample: Optional[bool] = field(default=False)
	eval_split_ratio: float = field(default=0.0, metadata={"help": "Fraction of data to reserve for eval (0 disables split)."})
	eval_split_seed: int = field(default=42, metadata={"help": "Seed for deterministic train/eval split."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
	cache_dir: Optional[str] = field(default=None)
	optim: str = field(default="adamw_torch")
	remove_unused_columns: bool = field(default=False)
	freeze_mm_mlp_adapter: bool = field(default=False)
	freeze_mm_vision_resampler: bool = field(default=False)
	mpt_attn_impl: Optional[str] = field(default="triton")
	model_max_length: int = field(default=4096, metadata={"help": "Maximum sequence length (tokens)."})
	evaluation_strategy: IntervalStrategy = field(default=IntervalStrategy.NO, metadata={"help": "Evaluation strategy: no | steps | epoch."})
	eval_steps: Optional[int] = field(default=None, metadata={"help": "Run evaluation every N steps when using steps strategy."})
	eval_delay: Optional[float] = field(default=0, metadata={"help": "Wait N steps or epochs before the first eval."})
	double_quant: bool = field(default=True)
	quant_type: str = field(default="nf4")
	bits: int = field(default=16)
	lora_enable: bool = False
	lora_r: int = 64
	lora_alpha: int = 16
	lora_dropout: float = 0.05
	lora_weight_path: str = ""
	lora_bias: str = "none"
	mm_projector_lr: Optional[float] = None
	mm_vision_tower_lr: Optional[float] = None
	group_by_varlen: bool = field(default=False)
	group_by_modality_length: bool = field(default=False)
	group_by_modality_length_auto: bool = field(default=False)
	auto_find_batch_size: bool = field(default=False)
	gradient_checkpointing: bool = field(default=True)
	verbose_logging: bool = field(default=False)
	attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Attention backend."})
	trainer_mode: str = field(default="regular")
	zo_eps: float = field(default=1e-3)
	zo_num_directions: int = field(default=1)


# ---------------------------------------------------------------------------
# Utility helpers (quantized state save / tokenizer resize)
# ---------------------------------------------------------------------------


def maybe_zero_3(param, ignore_status: bool = False, name: Optional[str] = None):
	from deepspeed import zero
	from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

	if hasattr(param, "ds_id"):
		if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
			raise RuntimeError(f"Tried to gather param {name} with status {param.ds_status}")
		with zero.GatheredParameters([param]):
			param = param.detach().cpu().clone()
	else:
		param = param.detach().cpu().clone()
	return param


def get_mm_adapter_state(named_params, keys_to_match):
	to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
	return {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
	"""Save either the full model or only tunable multimodal adapters when requested."""
	only_mm = False
	if getattr(trainer.args, "tune_mm_mlp_adapter", False):
		only_mm = True
	elif getattr(trainer.args, "mm_tunable_parts", None):
		mm_parts = trainer.args.mm_tunable_parts.split(",")
		only_mm = len(mm_parts) == 1 and any(part in ["mm_mlp_adapter", "mm_vision_resampler"] for part in mm_parts)

	trainer.accelerator.wait_for_everyone()
	torch.cuda.synchronize()
	rank0_print(f"Only save projectors: {only_mm}")

	if only_mm:
		keys_to_match = ["mm_projector", "vision_resampler"]
		weight_to_save = get_mm_adapter_state(trainer.model.named_parameters(), keys_to_match)
		trainer.model.config.save_pretrained(output_dir)

		current_folder = output_dir.split("/")[-1]
		parent_folder = os.path.dirname(output_dir)
		if trainer.args.local_rank in (0, -1):
			for filename in ["adapter_model.bin", "mm_projector.bin"]:
				torch.save(weight_to_save, os.path.join(parent_folder, filename))
		return

	if trainer.deepspeed:
		trainer.save_model(output_dir)
		return

	state_dict = trainer.model.state_dict()
	if trainer.args.should_save:
		cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
		del state_dict
		trainer._save(output_dir, state_dict=cpu_state_dict)


def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel) -> None:
	num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
	model.resize_token_embeddings(len(tokenizer))

	if num_new_tokens > 0:
		input_embeddings = model.get_input_embeddings().weight.data
		output_embeddings = model.get_output_embeddings().weight.data
		input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
		output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
		input_embeddings[-num_new_tokens:] = input_embeddings_avg
		output_embeddings[-num_new_tokens:] = output_embeddings_avg


# ---------------------------------------------------------------------------
# Apertus-specific preprocessing
# ---------------------------------------------------------------------------


def _tokenize_no_specials(text: str, tokenizer: transformers.PreTrainedTokenizer) -> List[int]:
	return tokenizer(text, add_special_tokens=False).input_ids


def preprocess_apertus(
	sources: Sequence[List[Dict[str, str]]],
	tokenizer: transformers.PreTrainedTokenizer,
	has_image: bool = False,
	max_len: int = 2048,
	system_message: str = "You are Apertus, a helpful assistant created by the SwissAI initiative.",
) -> Dict[str, torch.Tensor]:
	"""
	Build token/label tensors for Apertus chat format, masking non-assistant tokens.
	Enforces max_len with right-side truncation to keep the most recent content.
	"""

	# Work on a copy so we do not mutate the shared tokenizer (special tokens added below).
	tokenizer = copy.deepcopy(tokenizer)
	if has_image:
		tokenizer.add_tokens(["<image>"], special_tokens=True)

	image_token_id = tokenizer.convert_tokens_to_ids("<image>")
	system_start = tokenizer.convert_tokens_to_ids("<|system_start|>")
	system_end = tokenizer.convert_tokens_to_ids("<|system_end|>")
	user_start = tokenizer.convert_tokens_to_ids("<|user_start|>")
	user_end = tokenizer.convert_tokens_to_ids("<|user_end|>")
	assistant_start = tokenizer.convert_tokens_to_ids("<|assistant_start|>")
	assistant_end = tokenizer.convert_tokens_to_ids("<|assistant_end|>")

	# Build samples
	all_input_ids: List[List[int]] = []
	all_labels: List[List[int]] = []

	for sample in sources:
		# Strip possible leading assistant/system noise so we always start with user.
		if sample and sample[0].get("from", sample[0].get("role", "human")).lower() != "human":
			sample = sample[1:]

		input_ids: List[int] = []
		labels: List[int] = []

		# System segment (always masked)
		sys_tokens = _tokenize_no_specials(f"<|system_start|>{system_message}<|system_end|>", tokenizer)
		input_ids.extend(sys_tokens)
		labels.extend([IGNORE_INDEX] * len(sys_tokens))

		for turn in sample:
			role = turn.get("role") or turn.get("from")
			content = turn.get("content") or turn.get("value")
			role = role.lower()

			if role == "human":
				user_tokens = _tokenize_no_specials(f"<|user_start|>{content}<|user_end|>", tokenizer)
				input_ids.extend(user_tokens)
				labels.extend([IGNORE_INDEX] * len(user_tokens))
			elif role == "gpt" or role == "assistant":
				assistant_tokens = _tokenize_no_specials(f"<|assistant_start|>{content}<|assistant_end|>", tokenizer)
				input_ids.extend(assistant_tokens)
				labels.extend(assistant_tokens)
			else:
				raise ValueError(f"Unexpected role: {role}")

		# Replace image token id with sentinel used by the model
		for idx, tok in enumerate(input_ids):
			if tok == image_token_id:
				input_ids[idx] = IMAGE_TOKEN_INDEX

		# Truncate to max_len (keep the tail where latest assistant response lives)
		if len(input_ids) > max_len:
			input_ids = input_ids[-max_len:]
			labels = labels[-max_len:]

		all_input_ids.append(input_ids)
		all_labels.append(labels)

	return dict(
		input_ids=torch.tensor(all_input_ids, dtype=torch.long),
		labels=torch.tensor(all_labels, dtype=torch.long),
	)


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Sequence[str]:
	"""Inject image placeholders into text if multimodal training is enabled."""
	if not data_args.is_multimodal:
		return sources

	for source in sources:
		for sentence in source:
			role = sentence.get("from", sentence.get("role", "")).lower()
			value = sentence.get("value") or sentence.get("content", "")
			if role == "human" and DEFAULT_IMAGE_TOKEN in value:
				sentence["value"] = value.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
	return sources


# ---------------------------------------------------------------------------
# Dataset & collator
# ---------------------------------------------------------------------------


def _load_datasets_from_path(data_path: str, data_args: DataArguments) -> List[Dict]:
	"""Load instruction data from json/jsonl/yaml or brace-expanded paths."""
	records: List[Dict] = []

	if "{" in data_path and "}" in data_path:
		base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
		file_names = file_pattern.split(",")
		data_args.dataset_paths = []
		for file_name in file_names:
			full_path = f"{base_path}{file_name}.json"
			data_args.dataset_paths.append(full_path)
			rank0_print(f"Loading {full_path}")
			with open(full_path, "r") as fh:
				cur_data = json.load(fh)
				rank0_print(f"Loaded {len(cur_data)} samples from {full_path}")
				records.extend(cur_data)
	elif data_path.endswith(".yaml"):
		with open(data_path, "r") as fh:
			yaml_data = yaml.safe_load(fh)
			datasets = yaml_data.get("datasets")
		data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
		for dataset in datasets:
			json_path = dataset.get("json_path")
			sampling_strategy = dataset.get("sampling_strategy", "all")
			sampling_number = None

			rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

			if json_path.endswith(".jsonl"):
				cur_data = [json.loads(line.strip()) for line in open(json_path, "r")]
			elif json_path.endswith(".json"):
				cur_data = json.load(open(json_path, "r"))
			else:
				raise ValueError(f"Unsupported file type: {json_path}")

			if ":" in sampling_strategy:
				sampling_strategy, sampling_number = sampling_strategy.split(":")
				if "%" in sampling_number:
					sampling_number = math.ceil(int(sampling_number.split("%" )[0]) * len(cur_data) / 100)
				else:
					sampling_number = int(sampling_number)

			if sampling_strategy == "first" and sampling_number is not None:
				cur_data = cur_data[:sampling_number]
			elif sampling_strategy == "end" and sampling_number is not None:
				cur_data = cur_data[-sampling_number:]
			elif sampling_strategy == "random" and sampling_number is not None:
				random.shuffle(cur_data)
				cur_data = cur_data[:sampling_number]

			rank0_print(f"Loaded {len(cur_data)} samples from {json_path}")
			records.extend(cur_data)
	else:
		data_args.dataset_paths = [data_path]
		rank0_print(f"Loading {data_path}")
		with open(data_path, "r") as fh:
			cur_data = json.load(fh)
			rank0_print(f"Loaded {len(cur_data)} samples from {data_path}")
			records.extend(cur_data)

	rank0_print(f"Loaded {len(records)} total samples")
	return records


class ApertusSupervisedDataset(Dataset):
	def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, prefetched_data: Optional[List[Dict]] = None):
		super().__init__()
		self.tokenizer = tokenizer
		self.data_args = data_args
		self.list_data_dict = prefetched_data if prefetched_data is not None else _load_datasets_from_path(data_path, data_args)
		rank0_print("Formatting inputs...Skip in lazy mode")

	def __len__(self) -> int:
		return len(self.list_data_dict)

	@property
	def lengths(self) -> List[int]:
		length_list = []
		for sample in self.list_data_dict:
			img_tokens = 128 if "image" in sample else 0
			length_list.append(
				sum(len((conv.get("value") or conv.get("content", "")).split()) for conv in sample["conversations"]) + img_tokens
			)
		return length_list

	@property
	def modality_lengths(self) -> List[int]:
		length_list = []
		for sample in self.list_data_dict:
			cur_len = sum(len((conv.get("value") or conv.get("content", "")).split()) for conv in sample["conversations"])
			if "image" in sample or "video" in sample or self.data_args.early_mix_text:
				length_list.append(cur_len)
			else:
				length_list.append(-cur_len)
		return length_list

	# Image/video helpers mirror train.py for consistency
	def process_image(self, image_file, overwrite_image_aspect_ratio=None):
		image_folder = self.data_args.image_folder
		processor = self.data_args.image_processor
		try:
			image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
		except Exception as exn:
			print(f"Failed to open image {image_file}. Exception: {exn}")
			raise exn

		original_size = image.size
		image_aspect_ratio = overwrite_image_aspect_ratio or self.data_args.image_aspect_ratio
		if image_aspect_ratio == "highres":
			image = process_highres_image(image, processor, self.data_args.image_grid_pinpoints)
		elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
			image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints)
		elif image_aspect_ratio == "crop_split":
			image = process_highres_image_crop_split(image, self.data_args)
		elif image_aspect_ratio == "pad":
			def expand2square(pil_img, background_color):
				width, height = pil_img.size
				if width == height:
					return pil_img
				if width > height:
					result = Image.new(pil_img.mode, (width, width), background_color)
					result.paste(pil_img, (0, (width - height) // 2))
					return result
				result = Image.new(pil_img.mode, (height, height), background_color)
				result.paste(pil_img, ((height - width) // 2, 0))
				return result

			image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
			image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
		else:
			image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
		return image, original_size, "image"

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		# Basic retry strategy for flaky storage
		for attempt in range(3):
			try:
				return self._get_item(idx)
			except Exception as exc:
				print(f"[Try #{attempt}] Failed to fetch sample {idx}. Exception: {exc}")
				time.sleep(1)
		return self._get_item(idx)

	def _get_item(self, idx: int) -> Dict[str, torch.Tensor]:
		sources = self.list_data_dict[idx]
		sources = [sources] if isinstance(idx, int) else sources
		assert len(sources) == 1, "Dataset wrapper expected a single sample."

		if "image" in sources[0]:
			image_file = self.list_data_dict[idx]["image"]
			if isinstance(image_file, list):
				image = [self.process_image(f, "pad" if len(image_file) > 1 else None) for f in image_file]
				image = [[im[0], im[1], "image"] for im in image]
			else:
				image = [self.process_image(image_file)]
			sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
		elif "video" in sources[0]:
			video_file = os.path.join(self.data_args.video_folder, self.list_data_dict[idx]["video"])
			if not os.path.exists(video_file):
				raise FileNotFoundError(f"Video file {video_file} not found")

			try:
				if "shareVideoGPTV" in video_file:
					frame_files = sorted([os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))])
					num_frames_to_sample = self.data_args.frames_upbound if self.data_args.force_sample else 10
					total_frames = len(frame_files)
					sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
					video = []
					for frame_idx in sampled_indices:
						with Image.open(frame_files[frame_idx]) as img:
							video.append(img.convert("RGB"))
					avg_fps = 2
					frame_time = ",".join([f"{i/avg_fps:.2f}s" for i in sampled_indices])
					video_time = total_frames / avg_fps
				else:
					video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)

				processor = self.data_args.image_processor
				image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
				# Capture original spatial size for unpadding (width, height)
				if isinstance(video, list):
					orig_size = video[0].size  # PIL Image size -> (W, H)
				else:
					frame_shape = video[0].shape  # numpy array (H, W, C)
					orig_size = (frame_shape[1], frame_shape[0])
				if self.data_args.add_time_instruction:
					time_instr = (
						f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. "
						f"These frames are located at {frame_time}. Please answer the following questions related to this video."
					)
					sources[0]["conversations"][0]["value"] = f"{DEFAULT_IMAGE_TOKEN}\n{time_instr}\n{sources[0]['conversations'][0]['value'].replace(DEFAULT_IMAGE_TOKEN, '')}"
				image = [(image, orig_size, "video")]
				sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
			except Exception as exc:
				print(f"Failed to process video {video_file}: {exc}")
				return self._get_item(idx + 1)
		else:
			sources = copy.deepcopy([e["conversations"] for e in sources])

		has_image = ("image" in self.list_data_dict[idx]) or ("video" in self.list_data_dict[idx])
		data_dict = preprocess_apertus(sources, self.tokenizer, has_image=has_image, max_len=self.tokenizer.model_max_length)

		if isinstance(idx, int):
			data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

		if "image" in self.list_data_dict[idx] or "video" in self.list_data_dict[idx]:
			data_dict["image"] = image
		elif self.data_args.is_multimodal:
			crop_size = self.data_args.image_processor.crop_size
			data_dict["image"] = [
				(torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text")
			]

		data_dict["id"] = self.list_data_dict[idx].get("id", idx)
		return data_dict


class DataCollatorForSupervisedDataset(object):
	tokenizer: transformers.PreTrainedTokenizer
	printed_samples: int = 0
	max_print_samples: int = 2

	def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
		self.tokenizer = tokenizer

	def _pad(self, tensors, padding_value):
		if self.tokenizer.padding_side == "left":
			tensors = [torch.flip(t, [0]) for t in tensors]
		tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)
		if self.tokenizer.padding_side == "left":
			tensors = torch.flip(tensors, [1])
		return tensors

	def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
		input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
		input_ids = [_ids[: self.tokenizer.model_max_length] for _ids in input_ids]
		labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]

		if self.tokenizer.pad_token_id is None:
			self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

		input_ids = self._pad(input_ids, self.tokenizer.pad_token_id)
		labels = self._pad(labels, IGNORE_INDEX)

		batch = dict(
			input_ids=input_ids,
			labels=labels.long() if labels.dtype == torch.int32 else labels,
			attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
		)

		if "image" in instances[0]:
			images = [instance["image"] for instance in instances]
			batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
			batch["modalities"] = [im[2] for im_list in images for im in im_list]
			batch["images"] = [im[0] for im_list in images for im in im_list]

		if "prompt" in instances[0]:
			batch["prompts"] = [instance["prompt"] for instance in instances]

		# Lightweight visualization of the first batch for sanity checking.
		if self.printed_samples < self.max_print_samples:
			self._log_first_batch(batch)
			self.printed_samples += 1

		return batch

	def _log_first_batch(self, batch: Dict[str, torch.Tensor]) -> None:
		try:
			# Only log on rank 0 when distributed is initialized
			import torch.distributed as dist

			if dist.is_initialized() and dist.get_rank() != 0:
				return

			pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

			input_ids = batch["input_ids"][0]
			labels = batch["labels"][0]

			# Render tokens while showing the image sentinel explicitly instead of mapping it to pad.
			def render_tokens(tokens: List[int]) -> str:
				parts: List[str] = []
				for tok in tokens:
					if tok == IMAGE_TOKEN_INDEX:
						parts.append("<image_sentinel>")
					else:
						parts.append(self.tokenizer.decode([tok], skip_special_tokens=False))
				return "".join(parts)

			safe_input = [tok.item() if tok.item() >= 0 else pad_id for tok in input_ids]
			sanitized_labels: List[int] = []
			for tok in labels:
				tok_val = tok.item()
				if tok_val == IGNORE_INDEX:
					sanitized_labels.append(pad_id)
				else:
					sanitized_labels.append(tok_val if tok_val >= 0 else pad_id)

			decoded_input = render_tokens(safe_input)
			decoded_labels = render_tokens(sanitized_labels)

			rank0_print("")  # blank line
			rank0_print("")  # blank line
			rank0_print("==== Apertus Batch Preview ====")
			rank0_print(f"Input shape: {tuple(input_ids.shape)} | Labels shape: {tuple(labels.shape)}")
			rank0_print("-- Decoded Input --")
			rank0_print(decoded_input)
			rank0_print("-- Decoded Labels (IGNORE masked) --")
			rank0_print(decoded_labels)

			# Token-level view for first 100 tokens
			limit = min(100, len(input_ids))
			rank0_print("")  # blank line
			rank0_print("-- Token/Label Pairs (first 100) --")
			for idx in range(limit):
				tok_id = input_ids[idx].item()
				lab_id = labels[idx].item()
				lab_raw = labels[idx].item()
				if tok_id == IMAGE_TOKEN_INDEX:
					tok_str = "<image_sentinel>"
				else:
					tok_str = self.tokenizer.decode([tok_id]) if tok_id >= 0 else f"<special:{tok_id}>"
				if lab_raw == IGNORE_INDEX:
					lab_str = "[IGN]"
				elif lab_raw == IMAGE_TOKEN_INDEX:
					lab_str = "<image_sentinel>"
				else:
					lab_str = self.tokenizer.decode([lab_id]) if lab_id >= 0 else f"<special:{lab_id}>"
				rank0_print(f"{idx:02d}: input={tok_id}:{repr(tok_str)} | label={lab_raw}:{lab_str}")
		except Exception as exc:
			rank0_print(f"[batch preview] logging failed: {exc}")


class ForceEvalCallback(transformers.TrainerCallback):
	"""Ensure evaluations fire on the configured step cadence."""

	def __init__(self, eval_steps: Optional[int]):
		super().__init__()
		self.eval_steps = eval_steps

	def on_step_end(self, args, state, control, **kwargs):
		if self.eval_steps and state.global_step > 0 and state.global_step % self.eval_steps == 0:
			control.should_evaluate = True
			rank0_print(f"ForceEvalCallback: triggering evaluation at global_step={state.global_step}")
		return control


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments) -> Dict:
	all_records = _load_datasets_from_path(data_args.data_path, data_args)

	eval_dataset = None
	train_records = all_records

	if data_args.eval_split_ratio and data_args.eval_split_ratio > 0.0:
		random.seed(data_args.eval_split_seed)
		random.shuffle(train_records)
		num_eval = max(1, int(len(train_records) * data_args.eval_split_ratio))
		num_eval = min(num_eval, len(train_records) - 1) if len(train_records) > 1 else 0
		if num_eval > 0:
			eval_records = train_records[:num_eval]
			train_records = train_records[num_eval:]
			rank0_print(f"Dataset split: {len(train_records)} train / {len(eval_records)} eval (ratio={data_args.eval_split_ratio})")
			eval_dataset = ApertusSupervisedDataset(
				tokenizer=tokenizer,
				data_path=data_args.data_path,
				data_args=data_args,
				prefetched_data=eval_records,
			)

	train_dataset = ApertusSupervisedDataset(
		tokenizer=tokenizer,
		data_path=data_args.data_path,
		data_args=data_args,
		prefetched_data=train_records,
	)

	# Always log dataset cardinalities for visibility.
	if eval_dataset is not None:
		rank0_print(f"train_apertus.py: using train/eval datasets with sizes {len(train_dataset)} / {len(eval_dataset)}")
	else:
		rank0_print(f"train_apertus.py: using train dataset only with size {len(train_dataset)} (no eval split)")

	data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
	return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


# ---------------------------------------------------------------------------
# Tokenizer / template utilities
# ---------------------------------------------------------------------------


def build_apertus_tokenizer(model_args: ModelArguments, training_args: TrainingArguments) -> transformers.PreTrainedTokenizer:
	tokenizer = transformers.AutoTokenizer.from_pretrained(
		model_args.model_name_or_path,
		cache_dir=training_args.cache_dir,
		use_fast=IS_TOKENIZER_GE_014,
		padding_side="right",
		trust_remote_code=True,
	)

	# Ensure essential tokens exist. Prefer a dedicated pad token to avoid overlapping with <|assistant_end|>.
	pad_token = tokenizer.pad_token
	if pad_token is None or pad_token == tokenizer.eos_token:
		pad_token = "<pad>"
	special_tokens = {"pad_token": pad_token}
	if tokenizer.eos_token is None:
		special_tokens["eos_token"] = "</s>"
	tokenizer.add_special_tokens(special_tokens)

	# Align conversation template to Apertus
	conversation_lib.default_conversation = conversation_lib.conv_templates["apertus"].copy()
	if hasattr(tokenizer, "chat_template"):
		tokenizer.chat_template = None  # rely on conv template for formatting

	return tokenizer


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------


def train(attn_implementation: Optional[str] = None):
	global local_rank

	parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()

	if attn_implementation:
		training_args.attn_implementation = attn_implementation

	if training_args.verbose_logging:
		transformers.logging.set_verbosity_info()
		transformers.logging.enable_default_handler()
		transformers.logging.enable_explicit_format()

	local_rank = training_args.local_rank
	compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

	bnb_model_from_pretrained_args = {}
	if training_args.bits in [4, 8]:
		from transformers import BitsAndBytesConfig

		compute_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
		bnb_model_from_pretrained_args.update(
			dict(
				load_in_4bit=training_args.bits == 4,
				load_in_8bit=training_args.bits == 8,
				quantization_config=BitsAndBytesConfig(
					load_in_4bit=training_args.bits == 4,
					load_in_8bit=training_args.bits == 8,
					llm_int8_threshold=6.0,
					llm_int8_has_fp16_weight=False,
					bnb_4bit_compute_dtype=compute_dtype,
					bnb_4bit_use_double_quant=training_args.double_quant,
					bnb_4bit_quant_type=training_args.quant_type,
				),
			)
		)

	tokenizer = build_apertus_tokenizer(model_args, training_args)
	tokenizer.model_max_length = training_args.model_max_length

	model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
	model.config.use_cache = False

	# Normalize evaluation strategy in case it was provided as a raw string.
	if isinstance(training_args.evaluation_strategy, str):
		training_args.evaluation_strategy = IntervalStrategy(training_args.evaluation_strategy)

	# Ensure pad token is present and embeddings are aligned
	if tokenizer.pad_token is None:
		smart_tokenizer_and_embedding_resize(dict(pad_token="[PAD]"), tokenizer=tokenizer, model=model)
	elif tokenizer.pad_token_id not in tokenizer.all_special_ids:
		smart_tokenizer_and_embedding_resize(dict(pad_token=tokenizer.pad_token), tokenizer=tokenizer, model=model)

	conversation_lib.default_conversation = conversation_lib.conv_templates["apertus"].copy()

	if model_args.freeze_backbone:
		model.get_model().requires_grad_(False)
		if model.get_vision_tower() is not None:
			model.get_vision_tower().requires_grad_(False)
		if model.get_mm_projector() is not None:
			model.get_mm_projector().requires_grad_(False)

	if training_args.bits in [4, 8]:
		for name, module in model.named_modules():
			if "norm" in name:
				module = module.to(torch.float32)

	if training_args.gradient_checkpointing:
		model.enable_input_require_grads()
		model.gradient_checkpointing_enable()

	if model_args.vision_tower is not None:
		model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

		vision_tower = model.get_vision_tower()
		vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

		data_args.image_processor = vision_tower.image_processor
		data_args.is_multimodal = True

		model.config.image_aspect_ratio = data_args.image_aspect_ratio
		if data_args.image_grid_pinpoints is not None:
			if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
				try:
					patch_size = data_args.image_processor.size[0]
				except Exception:
					patch_size = data_args.image_processor.size["shortest_edge"]

				assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
				matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
				range_start = tuple(map(int, matches[0]))
				range_end = tuple(map(int, matches[-1]))
				grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
				data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
			elif isinstance(data_args.image_grid_pinpoints, str):
				data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

		model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
		model.config.image_crop_resolution = data_args.image_crop_resolution
		model.config.image_split_resolution = data_args.image_split_resolution
		model.config.tokenizer_padding_side = tokenizer.padding_side
		model.config.tokenizer_model_max_length = tokenizer.model_max_length
		model.config.mm_newline_position = model_args.mm_newline_position
		model.config.add_faster_video = model_args.add_faster_video
		model.config.faster_token_stride = model_args.faster_token_stride
		model.config.add_time_instruction = data_args.add_time_instruction
		model.config.force_sample = data_args.force_sample
		model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride

		model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

		if training_args.freeze_mm_vision_resampler and model.get_vision_resampler() is not None:
			model.get_vision_resampler().requires_grad_(False)
		if training_args.freeze_mm_mlp_adapter and model.get_mm_projector() is not None:
			model.get_mm_projector().requires_grad_(False)

	data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

	# Ensure evaluation is enabled when an eval split is present.
	if data_module.get("eval_dataset") is not None:
		training_args.do_eval = True
		rank0_print(f"Eval dataset detected ({len(data_module['eval_dataset'])} samples); enabling evaluation with strategy={training_args.evaluation_strategy} eval_steps={training_args.eval_steps}")
	else:
		rank0_print("No eval dataset provided; evaluation will be skipped.")
	trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

	if training_args.evaluation_strategy == IntervalStrategy.STEPS:
		effective_eval_steps = training_args.eval_steps or training_args.logging_steps
		if effective_eval_steps:
			trainer.add_callback(ForceEvalCallback(effective_eval_steps))

	if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
		trainer.train(resume_from_checkpoint=True)
	else:
		trainer.train()
	trainer.save_state()

	# Run a final evaluation pass explicitly when an eval set exists to guarantee metrics are produced.
	if training_args.do_eval and data_module.get("eval_dataset") is not None:
		final_metrics = trainer.evaluate(eval_dataset=data_module["eval_dataset"], metric_key_prefix="eval_final")
		rank0_print(f"Final evaluation metrics: {final_metrics}")

	model.config.use_cache = True
	safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
	rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
	train()
