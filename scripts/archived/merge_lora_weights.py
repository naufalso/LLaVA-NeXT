import argparse
import torch
from PIL import Image
import requests
from io import BytesIO

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import disable_torch_init


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def merge_lora(args):
    disable_torch_init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = get_model_name_from_path(args.model_path)

    # Load weights on CPU first, then move to GPU for inference and warmup
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map="cpu", torch_dtype=args.dtype
    )

    model.to(device)
    model.eval()

    # Build a simple prompt + image inference to warm up weights
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "apertus" in model_name.lower():
        conv_mode = "apertus_ori"
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

    # Load image and prepare tensor
    image = load_image(args.image_file)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
    image_tensor = image_tensor.to(device)

    # Ensure image tensor dtype matches model dtype
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    image_tensor = image_tensor.to(dtype=model_dtype)

    # Non-interactive single query to describe image
    inp = "Describe the image in detail"
    print(f"{roles[0]}: {inp}")

    if image is not None:
        if getattr(model.config, "mm_use_im_start_end", False):
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if hasattr(conv, 'stop_str') and conv.stop_str:
        stop_str = conv.stop_str
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Image Tensor shape: {image_tensor.shape}")

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    print(f"Output IDs shape: {output_ids.shape}")
    print(f"Full decoded output: {tokenizer.decode(output_ids[0])}")

    if torch.equal(output_ids[0, : input_ids.shape[1]], input_ids[0]):
        print("The generated IDs contain the prompt IDs as prefix.")
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
    else:
        print("The generated IDs do NOT contain the prompt IDs as prefix.")
        outputs = tokenizer.decode(output_ids[0]).strip()
    conv.messages[-1][-1] = outputs

    print(f"{roles[1]}: {outputs}")
    if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

    # Save merged model + tokenizer
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--image-file", type=str, default="docs/ov_chat_images/example2_dog.jpg")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    merge_lora(args)
