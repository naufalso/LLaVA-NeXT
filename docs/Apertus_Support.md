# LLaVA-Apertus Support

This document describes the integration of the Apertus language model into LLaVA-NeXT.

## Overview

LLaVA-Apertus extends the LLaVA framework to support the [Apertus-8B-2509](https://huggingface.co/swiss-ai/Apertus-8B-2509) language model from Swiss AI. Apertus is a multilingual, compliant language model with 8 billion parameters.

## Model Architecture

Apertus is based on a transformer architecture similar to LLaMA with some key differences:

- **Architecture**: ApertusForCausalLM
- **Hidden Size**: 4096
- **Intermediate Size**: 21504
- **Layers**: 32
- **Attention Heads**: 32
- **Key-Value Heads**: 8 (Grouped Query Attention)
- **Vocabulary Size**: 131072
- **Max Position Embeddings**: 65536
- **Activation Function**: xielu (instead of silu)
- **QK Normalization**: Enabled
- **RoPE Scaling**: llama3 type with factor 8.0

## Requirements

- transformers >= 4.56.0 (for native Apertus support)
- torch >= 2.1.0
- All standard LLaVA-NeXT dependencies

## Usage

### Loading a Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.model.language_model.llava_apertus import LlavaApertusForCausalLM, LlavaApertusConfig

# Load config
config = LlavaApertusConfig.from_pretrained("your-llava-apertus-model")

# Load model
model = LlavaApertusForCausalLM.from_pretrained(
    "your-llava-apertus-model",
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### Training

LLaVA-Apertus can be trained using the same training scripts as other LLaVA models. Simply specify the appropriate base model:

```bash
bash scripts/train/pretrain.sh \
    --model_name_or_path swiss-ai/Apertus-8B-2509 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    ...
```

### Inference

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "path/to/llava-apertus"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
```

## Implementation Details

The LLaVA-Apertus implementation follows the same architecture pattern as other LLaVA models:

1. **LlavaApertusConfig**: Configuration class extending `ApertusConfig`
2. **LlavaApertusModel**: Model class inheriting from both `LlavaMetaModel` and `ApertusModel`
3. **LlavaApertusForCausalLM**: CausalLM class combining `ApertusForCausalLM` and `LlavaMetaForCausalLM`

The model supports all standard LLaVA features:
- Multi-modal input processing (images + text)
- Vision tower integration
- Efficient generation with KV-caching
- DPO training compatibility

## Notes

- The Apertus model uses a custom `xielu` activation function. If CUDA-fused xIELU is not available, it falls back to a Python implementation.
- For optimal performance with xIELU, you may optionally install the CUDA-fused version (experimental). Check the [XIELU repository](https://github.com/nickjbrowning/XIELU) for installation instructions and version information.
- The model requires transformers 4.56+ for native support. A compatibility layer is included for older versions.

## References

- [Apertus-8B-2509 on HuggingFace](https://huggingface.co/swiss-ai/Apertus-8B-2509)
- [Swiss AI](https://www.swiss.ai/)
- [LLaVA-NeXT Documentation](./README.md)
