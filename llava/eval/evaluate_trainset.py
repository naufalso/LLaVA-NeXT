import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import json

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_dataset(datset_path, sample_size=None):
    with open(datset_path, 'r') as f:
        data = json.load(f)
    
    if sample_size is not None:
        data = data[:sample_size]

    return data

def iterate_dataset(dataset, dataset_path):
    for item in dataset:
        data_id = item.get("id", None)
        image_path = item.get("image", None)

        if image_path is not None:
            image = load_image(os.path.join(dataset_path, image_path))
        
        conversations = item.get("conversations", [])

        if len(conversations) < 1:
            continue

        human = []
        gpt = []
        for conv in conversations:
            if conv['from'] == 'human':
                human.append(conv['value'])
            elif conv['from'] == 'gpt':
                gpt.append(conv['value'])

        assert len(human) == len(gpt), "Mismatched human and gpt pairs"

        yield data_id, image, human, gpt


        

    

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, torch_dtype=args.dtype)

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
        print("[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ("user", "assistant")
    else:
        roles = conv.roles

    dataset = load_dataset(args.dataset_path, sample_size=args.sample_size)
    
    outputs_list = []
    for data_id, image, human, gpt in iterate_dataset(dataset, os.path.dirname(args.dataset_path)):
        print(f"Processing data ID: {data_id}")
        first_human = human[0]
        first_gpt = gpt[0]

        prompt = first_human
        if image is not None:
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].cuda()
            # Ensure image_tensor matches model dtype
            image_tensor = image_tensor.to(dtype=model.dtype)
            if model.config.mm_use_im_start_end:
                prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
            else:
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        
        new_conv = conv.copy()
        new_conv.append_message(roles[0], prompt)
        new_conv.append_message(roles[1], None)

        full_prompt = new_conv.get_prompt()
        input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        stop_str = new_conv.sep if new_conv.sep_style != SeparatorStyle.TWO else new_conv.sep2
        # Use stop_str from conversation config if available (e.g., for Apertus)
        if hasattr(new_conv, 'stop_str') and new_conv.stop_str:
            stop_str = new_conv.stop_str
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(input_ids, images=image_tensor, do_sample=True if args.temperature > 0.0 else False, temperature=args.temperature, max_new_tokens=args.max_new_tokens, use_cache=True, stopping_criteria=[stopping_criteria])

        if torch.equal(output_ids[0, :input_ids.shape[1]], input_ids[0]):
            # print("The generated IDs contain the prompt IDs as prefix.")
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        else:
            # print("The generated IDs do NOT contain the prompt IDs as prefix.")
            outputs = tokenizer.decode(output_ids[0]).strip()
        new_conv.messages[-1][-1] = outputs
        if args.debug:
            print(f"Query: {first_human}")
            print()
            print(f"Full Prompt: {full_prompt}")
            print()
            print(f"Prediction: {outputs}")
            print()
            print(f"Ground Truth: {first_gpt}")
            print()
        result = {
            "id": data_id,
            "query": first_human,
            "prediction": outputs,
            "ground_truth": first_gpt
        }
        outputs_list.append(result)

    # Save outputs_list to a JSON file
    output_path = args.output if args.output is not None else "eval_outputs.json"
    # Ensure the output directory exists
    if os.path.dirname(output_path) != "":
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(outputs_list, f, indent=4)
    print(f"Saved evaluation outputs to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sample-size", type=int, default=10) 
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the evaluation dataset JSON file.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the evaluation outputs JSON file.")
    args = parser.parse_args()
    main(args)
