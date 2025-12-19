#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


"""
Unified LLaVA Language Model Architecture Factory

This module provides a factory-based approach to generate LLaVA model variants for
different base language models (LLaMA, Mistral, Mixtral, Gemma, Qwen, Qwen-MoE, MPT).

The factory pattern eliminates code duplication across 7 separate model files by:
1. Centralizing common model logic in factory functions
2. Defining model-specific characteristics in a registry
3. Dynamically generating model classes at module load time

Key Components:
- BASE_MODEL_REGISTRY: Defines characteristics for each base model
- create_llava_config_class(): Factory for config classes
- create_llava_model_class(): Factory for model classes
- create_llava_causal_lm_class(): Factory for CausalLM classes

All generated classes maintain backward compatibility with existing code.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


# Registry defining characteristics of each base model
BASE_MODEL_REGISTRY = {
    "llava_llama": {
        "base_config": "LlamaConfig",
        "base_model": "LlamaModel", 
        "base_causal_lm": "LlamaForCausalLM",
        "model_attribute": "model",  # attribute name for the model in CausalLM class
        "supports_dpo": True,
        "supports_modalities": True,
        "use_direct_init": True,  # use BaseForCausalLM.__init__ vs super(Base, self).__init__
        "config_extras": {
            "temperature": 0.0,
            "max_new_tokens": 1024,
            "do_sample": False,
            "top_p": None,
        },
        "import_module": "transformers",
    },
    "llava_mistral": {
        "base_config": "MistralConfig",
        "base_model": "MistralModel",
        "base_causal_lm": "MistralForCausalLM",
        "model_attribute": "model",
        "supports_dpo": False,
        "supports_modalities": False,
        "use_direct_init": False,
        "config_extras": {
            "temperature": 0.0,
            "max_new_tokens": 1024,
            "do_sample": False,
            "top_p": None,
        },
        "import_module": "transformers",
    },
    "llava_mixtral": {
        "base_config": "MixtralConfig",
        "base_model": "MixtralModel",
        "base_causal_lm": "MixtralForCausalLM",
        "model_attribute": "model",
        "supports_dpo": True,
        "supports_modalities": True,
        "use_direct_init": False,
        "config_extras": {},
        "import_module": "transformers",
    },
    "llava_gemma": {
        "base_config": "GemmaConfig",
        "base_model": "GemmaModel",
        "base_causal_lm": "GemmaForCausalLM",
        "model_attribute": "model",
        "supports_dpo": False,
        "supports_modalities": False,
        "use_direct_init": False,
        "config_extras": {},
        "import_module": "transformers",
    },
    "llava_qwen": {
        "base_config": "Qwen2Config",
        "base_model": "Qwen2Model",
        "base_causal_lm": "Qwen2ForCausalLM",
        "model_attribute": "model",
        "supports_dpo": True,
        "supports_modalities": True,
        "use_direct_init": True,
        "config_extras": {},
        "import_module": "transformers",
    },
    "llava_qwen_moe": {
        "base_config": "Qwen2MoeConfig",
        "base_model": "Qwen2MoeModel",
        "base_causal_lm": "Qwen2MoeForCausalLM",
        "model_attribute": "model",
        "supports_dpo": True,
        "supports_modalities": True,
        "use_direct_init": True,
        "config_extras": {},
        "import_module": "transformers",
    },
    "llava_mpt": {
        "base_config": "MptConfig",
        "base_model": "MptModel",
        "base_causal_lm": "MptForCausalLM",
        "model_attribute": "transformer",  # MPT uses 'transformer' instead of 'model'
        "supports_dpo": False,
        "supports_modalities": False,
        "use_direct_init": False,
        "needs_hidden_size_init": True,  # MPT needs hidden_size = d_model
        "needs_embed_tokens": True,  # MPT model needs embed_tokens method
        "use_generation_config": True,  # MPT uses GenerationConfig
        "mpt_prepare_signature": True,  # MPT has different prepare_inputs signature
        "config_extras": {},
        "import_module": "transformers",
    },
}


def _import_base_classes(model_type):
    """Import base classes for a given model type from transformers."""
    registry_entry = BASE_MODEL_REGISTRY[model_type]
    import_module = registry_entry["import_module"]
    
    # Import from transformers
    import transformers
    
    config_class = getattr(transformers, registry_entry["base_config"])
    model_class = getattr(transformers, registry_entry["base_model"])
    causal_lm_class = getattr(transformers, registry_entry["base_causal_lm"])
    
    return config_class, model_class, causal_lm_class


def create_llava_config_class(model_type):
    """
    Factory function to create a LLaVA config class for a specific base model.
    
    Args:
        model_type: String identifier for the model (e.g., "llava_llama")
        
    Returns:
        A config class inheriting from the appropriate base config
    """
    registry_entry = BASE_MODEL_REGISTRY[model_type]
    base_config_class, _, _ = _import_base_classes(model_type)
    
    # Create config class name (e.g., LlavaConfig, LlavaMistralConfig)
    if model_type == "llava_llama":
        class_name = "LlavaConfig"
    else:
        # Extract the model name part (e.g., "mistral" from "llava_mistral")
        base_name = model_type.replace("llava_", "")
        class_name = f"Llava{base_name.capitalize()}Config"
        # Handle special cases
        if base_name == "qwen_moe":
            class_name = "LlavaQwenMoeConfig"
    
    # Create the config class dynamically
    config_attrs = {
        "model_type": model_type,
        **registry_entry["config_extras"]
    }
    
    config_class = type(class_name, (base_config_class,), config_attrs)
    return config_class


def create_llava_model_class(model_type, config_class):
    """
    Factory function to create a LLaVA model class for a specific base model.
    
    Args:
        model_type: String identifier for the model
        config_class: The config class for this model type
        
    Returns:
        A model class inheriting from LlavaMetaModel and the base model
    """
    registry_entry = BASE_MODEL_REGISTRY[model_type]
    _, base_model_class, _ = _import_base_classes(model_type)
    
    # Create model class name
    if model_type == "llava_llama":
        class_name = "LlavaLlamaModel"
    else:
        base_name = model_type.replace("llava_", "")
        class_name = f"Llava{base_name.capitalize()}Model"
        if base_name == "qwen_moe":
            class_name = "LlavaQwenMoeModel"
    
    def __init__(self, config):
        # MPT needs special handling for hidden_size
        if registry_entry.get("needs_hidden_size_init"):
            config.hidden_size = config.d_model
        # Call the base model's __init__ via the mixin chain
        # We can't use model_class here as it's not defined yet, so we rely on MRO
        base_model_class.__init__(self, config)
    
    # Create attributes dict
    attrs = {
        "config_class": config_class,
        "__init__": __init__,
    }
    
    # MPT needs embed_tokens method
    if registry_entry.get("needs_embed_tokens"):
        def embed_tokens(self, x):
            return self.wte(x)
        attrs["embed_tokens"] = embed_tokens
    
    # Create the model class with MRO: LlavaMetaModel, BaseModel
    model_class = type(class_name, (LlavaMetaModel, base_model_class), attrs)
    return model_class


def create_llava_causal_lm_class(model_type, config_class, model_class):
    """
    Factory function to create a LLaVA CausalLM class for a specific base model.
    
    Args:
        model_type: String identifier for the model
        config_class: The config class for this model type
        model_class: The model class for this model type
        
    Returns:
        A CausalLM class inheriting from the base CausalLM and LlavaMetaForCausalLM
    """
    registry_entry = BASE_MODEL_REGISTRY[model_type]
    _, _, base_causal_lm_class = _import_base_classes(model_type)
    
    # Create CausalLM class name
    if model_type == "llava_llama":
        class_name = "LlavaLlamaForCausalLM"
    else:
        base_name = model_type.replace("llava_", "")
        class_name = f"Llava{base_name.capitalize()}ForCausalLM"
        if base_name == "qwen_moe":
            class_name = "LlavaQwenMoeForCausalLM"
    
    model_attribute = registry_entry["model_attribute"]
    supports_dpo = registry_entry["supports_dpo"]
    supports_modalities = registry_entry["supports_modalities"]
    use_direct_init = registry_entry["use_direct_init"]
    
    def __init__(self, config):
        # Initialize base class
        if use_direct_init:
            base_causal_lm_class.__init__(self, config)
        else:
            super(base_causal_lm_class, self).__init__(config)
        
        # Set model type
        config.model_type = model_type
        config.rope_scaling = None
        
        # MPT uses GenerationConfig
        if registry_entry.get("use_generation_config"):
            self.generation_config = GenerationConfig(
                temperature=0.0,
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
            )
        
        # Create the model instance
        setattr(self, model_attribute, model_class(config))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_model(self):
        return getattr(self, model_attribute)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        cache_position = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # Prepare inputs for multimodal
        if inputs_embeds is None:
            if registry_entry.get("mpt_prepare_signature"):
                # MPT has different signature
                (input_ids, attention_mask, past_key_values, inputs_embeds, labels) = \
                    self.prepare_inputs_labels_for_multimodal(
                        input_ids, attention_mask, past_key_values, labels, images
                    )
            elif supports_modalities:
                # Models with modalities support
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = \
                    self.prepare_inputs_labels_for_multimodal(
                        input_ids, position_ids, attention_mask, past_key_values, labels, 
                        images, modalities, image_sizes
                    )
            else:
                # Models without modalities support
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = \
                    self.prepare_inputs_labels_for_multimodal(
                        input_ids, position_ids, attention_mask, past_key_values, labels, 
                        images, image_sizes
                    )
        
        # Handle DPO forward pass
        if supports_dpo and dpo_forward:
            outputs = getattr(self, model_attribute)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels
        else:
            # Standard forward pass - MPT has different signature
            if registry_entry.get("mpt_prepare_signature"):
                return base_causal_lm_class.forward(
                    self,
                    input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                return base_causal_lm_class.forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # Handle modalities parameter
        if supports_modalities:
            modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None:
            if registry_entry.get("mpt_prepare_signature"):
                # MPT doesn't use position_ids, modalities, image_sizes in generate
                (inputs, attention_mask, _, inputs_embeds, _) = \
                    self.prepare_inputs_labels_for_multimodal(
                        inputs, attention_mask, None, None, images
                    )
            elif supports_modalities:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = \
                    self.prepare_inputs_labels_for_multimodal(
                        inputs, position_ids, attention_mask, None, None, 
                        images, modalities, image_sizes=image_sizes
                    )
            else:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = \
                    self.prepare_inputs_labels_for_multimodal(
                        inputs, position_ids, attention_mask, None, None, 
                        images, image_sizes=image_sizes
                    )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        return base_causal_lm_class.generate(
            self,
            position_ids=position_ids, 
            attention_mask=attention_mask, 
            inputs_embeds=inputs_embeds, 
            **kwargs
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = base_causal_lm_class.prepare_inputs_for_generation(
            self, input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
    
    # Build attributes dict
    attrs = {
        "config_class": config_class,
        "__init__": __init__,
        "get_model": get_model,
        "forward": forward,
        "generate": generate,
        "prepare_inputs_for_generation": prepare_inputs_for_generation,
    }
    
    # MPT needs gradient checkpointing support
    if model_type == "llava_mpt":
        attrs["supports_gradient_checkpointing"] = True
        
        # Capture model_class in closure
        def _make_set_gradient_checkpointing(captured_model_class):
            def _set_gradient_checkpointing(self, module, value=False):
                if isinstance(module, captured_model_class):
                    module.gradient_checkpointing = value
            return _set_gradient_checkpointing
        
        attrs["_set_gradient_checkpointing"] = _make_set_gradient_checkpointing(model_class)
    
    # Create the CausalLM class
    causal_lm_class = type(class_name, (base_causal_lm_class, LlavaMetaForCausalLM), attrs)
    return causal_lm_class


# Generate all model classes and register them
_generated_classes = {}

for model_type in BASE_MODEL_REGISTRY.keys():
    # Create the three classes for this model type
    config_class = create_llava_config_class(model_type)
    model_class = create_llava_model_class(model_type, config_class)
    causal_lm_class = create_llava_causal_lm_class(model_type, config_class, model_class)
    
    # Store in module globals for imports
    globals()[config_class.__name__] = config_class
    globals()[model_class.__name__] = model_class
    globals()[causal_lm_class.__name__] = causal_lm_class
    
    # Store for reference
    _generated_classes[model_type] = {
        "config": config_class,
        "model": model_class,
        "causal_lm": causal_lm_class,
    }
    
    # Register with transformers
    AutoConfig.register(model_type, config_class)
    AutoModelForCausalLM.register(config_class, causal_lm_class)


# Export all generated classes
__all__ = [cls.__name__ for classes in _generated_classes.values() for cls in classes.values()]
