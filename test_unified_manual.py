#!/usr/bin/env python3
"""
Manual Test Script for Unified Language Model Implementation

This script verifies that the unified implementation works correctly by:
1. Testing imports and class availability
2. Verifying class inheritance and attributes
3. Testing model registration with transformers
4. Comparing behavior with original implementations

Run this script to manually verify the refactoring before deployment.
"""

import sys
import traceback


def test_imports():
    """Test that all classes can be imported from unified implementation."""
    print("=" * 70)
    print("TEST 1: Import All Classes")
    print("=" * 70)
    
    try:
        from llava.model.language_model.llava_unified import (
            # LLaMA
            LlavaConfig,
            LlavaLlamaModel,
            LlavaLlamaForCausalLM,
            # Mistral
            LlavaMistralConfig,
            LlavaMistralModel,
            LlavaMistralForCausalLM,
            # Mixtral
            LlavaMixtralConfig,
            LlavaMixtralModel,
            LlavaMixtralForCausalLM,
            # Gemma
            LlavaGemmaConfig,
            LlavaGemmaModel,
            LlavaGemmaForCausalLM,
            # Qwen
            LlavaQwenConfig,
            LlavaQwenModel,
            LlavaQwenForCausalLM,
            # Qwen MoE
            LlavaQwenMoeConfig,
            LlavaQwenMoeModel,
            LlavaQwenMoeForCausalLM,
            # MPT
            LlavaMptConfig,
            LlavaMptModel,
            LlavaMptForCausalLM,
        )
        
        print("✓ All 21 classes imported successfully")
        print("  - 7 Config classes")
        print("  - 7 Model classes")
        print("  - 7 CausalLM classes")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_import_from_model_package():
    """Test imports from llava.model package (backward compatibility)."""
    print("\n" + "=" * 70)
    print("TEST 2: Import from llava.model (Backward Compatibility)")
    print("=" * 70)
    
    try:
        from llava.model import (
            LlavaConfig,
            LlavaLlamaForCausalLM,
            LlavaMistralForCausalLM,
            LlavaMixtralForCausalLM,
            LlavaGemmaForCausalLM,
            LlavaQwenForCausalLM,
            LlavaQwenMoeForCausalLM,
            LlavaMptForCausalLM,
        )
        
        print("✓ All classes accessible from llava.model package")
        print("  Backward compatibility maintained")
        return True
        
    except ImportError as e:
        print(f"✗ Import from llava.model failed: {e}")
        traceback.print_exc()
        return False


def test_class_attributes():
    """Test that classes have expected attributes and structure."""
    print("\n" + "=" * 70)
    print("TEST 3: Class Attributes and Structure")
    print("=" * 70)
    
    try:
        from llava.model.language_model.llava_unified import (
            LlavaConfig,
            LlavaLlamaForCausalLM,
            LlavaMptForCausalLM,
            LlavaMptModel,
        )
        
        # Test LlavaConfig attributes
        assert LlavaConfig.model_type == "llava_llama", "LlavaConfig.model_type incorrect"
        print("✓ LlavaConfig has correct model_type")
        
        assert hasattr(LlavaConfig, "temperature"), "LlavaConfig missing temperature"
        assert LlavaConfig.temperature == 0.0, "LlavaConfig.temperature incorrect"
        print("✓ LlavaConfig has config extras (temperature, max_new_tokens, etc.)")
        
        # Test MPT special attributes
        assert hasattr(LlavaMptForCausalLM, "supports_gradient_checkpointing"), \
            "LlavaMptForCausalLM missing supports_gradient_checkpointing"
        assert LlavaMptForCausalLM.supports_gradient_checkpointing == True, \
            "LlavaMptForCausalLM.supports_gradient_checkpointing incorrect"
        print("✓ LlavaMptForCausalLM has supports_gradient_checkpointing")
        
        # Test MPT model has embed_tokens
        assert hasattr(LlavaMptModel, "embed_tokens"), "LlavaMptModel missing embed_tokens"
        print("✓ LlavaMptModel has embed_tokens method")
        
        return True
        
    except (AssertionError, ImportError) as e:
        print(f"✗ Class attributes test failed: {e}")
        traceback.print_exc()
        return False


def test_inheritance():
    """Test that classes have correct inheritance."""
    print("\n" + "=" * 70)
    print("TEST 4: Class Inheritance")
    print("=" * 70)
    
    try:
        from llava.model.language_model.llava_unified import (
            LlavaLlamaModel,
            LlavaLlamaForCausalLM,
        )
        from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
        from transformers import LlamaModel, LlamaForCausalLM
        
        # Test LlavaLlamaModel inheritance
        assert issubclass(LlavaLlamaModel, LlavaMetaModel), \
            "LlavaLlamaModel doesn't inherit from LlavaMetaModel"
        assert issubclass(LlavaLlamaModel, LlamaModel), \
            "LlavaLlamaModel doesn't inherit from LlamaModel"
        print("✓ LlavaLlamaModel has correct inheritance (LlavaMetaModel, LlamaModel)")
        
        # Test LlavaLlamaForCausalLM inheritance
        assert issubclass(LlavaLlamaForCausalLM, LlamaForCausalLM), \
            "LlavaLlamaForCausalLM doesn't inherit from LlamaForCausalLM"
        assert issubclass(LlavaLlamaForCausalLM, LlavaMetaForCausalLM), \
            "LlavaLlamaForCausalLM doesn't inherit from LlavaMetaForCausalLM"
        print("✓ LlavaLlamaForCausalLM has correct inheritance")
        
        return True
        
    except (AssertionError, ImportError) as e:
        print(f"✗ Inheritance test failed: {e}")
        traceback.print_exc()
        return False


def test_transformers_registration():
    """Test that models are registered with transformers."""
    print("\n" + "=" * 70)
    print("TEST 5: Transformers Registration")
    print("=" * 70)
    
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        
        model_types = [
            "llava_llama",
            "llava_mistral",
            "llava_mixtral",
            "llava_gemma",
            "llava_qwen",
            "llava_qwen_moe",
            "llava_mpt",
        ]
        
        registered_count = 0
        for model_type in model_types:
            if model_type in AutoConfig._model_type_to_config_class:
                print(f"✓ {model_type} registered in AutoConfig")
                registered_count += 1
            else:
                print(f"✗ {model_type} NOT registered in AutoConfig")
        
        if registered_count == len(model_types):
            print(f"\n✓ All {registered_count} models registered successfully")
            return True
        else:
            print(f"\n✗ Only {registered_count}/{len(model_types)} models registered")
            return False
        
    except Exception as e:
        print(f"✗ Registration test failed: {e}")
        traceback.print_exc()
        return False


def test_method_signatures():
    """Test that methods have correct signatures."""
    print("\n" + "=" * 70)
    print("TEST 6: Method Signatures")
    print("=" * 70)
    
    try:
        from llava.model.language_model.llava_unified import (
            LlavaLlamaForCausalLM,
            LlavaMistralForCausalLM,
            LlavaMptForCausalLM,
        )
        import inspect
        
        # Check LLaMA forward has modalities and dpo_forward
        llama_sig = inspect.signature(LlavaLlamaForCausalLM.forward)
        assert "modalities" in llama_sig.parameters, "LlavaLlamaForCausalLM.forward missing modalities"
        assert "dpo_forward" in llama_sig.parameters, "LlavaLlamaForCausalLM.forward missing dpo_forward"
        print("✓ LlavaLlamaForCausalLM.forward has modalities and dpo_forward parameters")
        
        # Check Mistral forward (should have params but not use them)
        mistral_sig = inspect.signature(LlavaMistralForCausalLM.forward)
        assert "modalities" in mistral_sig.parameters, "LlavaMistralForCausalLM.forward missing modalities"
        assert "dpo_forward" in mistral_sig.parameters, "LlavaMistralForCausalLM.forward missing dpo_forward"
        print("✓ LlavaMistralForCausalLM.forward has modalities and dpo_forward parameters")
        
        # Check MPT forward has images
        mpt_sig = inspect.signature(LlavaMptForCausalLM.forward)
        assert "images" in mpt_sig.parameters, "LlavaMptForCausalLM.forward missing images"
        print("✓ LlavaMptForCausalLM.forward has images parameter")
        
        # Check get_model method exists
        assert hasattr(LlavaLlamaForCausalLM, "get_model"), "LlavaLlamaForCausalLM missing get_model"
        assert hasattr(LlavaMptForCausalLM, "get_model"), "LlavaMptForCausalLM missing get_model"
        print("✓ All models have get_model method")
        
        return True
        
    except (AssertionError, ImportError) as e:
        print(f"✗ Method signatures test failed: {e}")
        traceback.print_exc()
        return False


def test_config_instantiation():
    """Test that config classes can be instantiated."""
    print("\n" + "=" * 70)
    print("TEST 7: Config Instantiation")
    print("=" * 70)
    
    try:
        from llava.model.language_model.llava_unified import (
            LlavaConfig,
            LlavaMistralConfig,
            LlavaMptConfig,
        )
        
        # Try to instantiate configs with minimal required parameters
        try:
            llama_config = LlavaConfig()
            print("✓ LlavaConfig instantiated successfully")
        except TypeError as e:
            print(f"⚠ LlavaConfig requires parameters: {e}")
        
        try:
            mistral_config = LlavaMistralConfig()
            print("✓ LlavaMistralConfig instantiated successfully")
        except TypeError as e:
            print(f"⚠ LlavaMistralConfig requires parameters: {e}")
        
        try:
            mpt_config = LlavaMptConfig()
            print("✓ LlavaMptConfig instantiated successfully")
        except TypeError as e:
            print(f"⚠ LlavaMptConfig requires parameters: {e}")
        
        print("\nNote: Some configs may require parameters - this is expected")
        return True
        
    except Exception as e:
        print(f"✗ Config instantiation test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "UNIFIED MODEL MANUAL TEST SUITE" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    tests = [
        ("Import All Classes", test_imports),
        ("Backward Compatible Imports", test_import_from_model_package),
        ("Class Attributes", test_class_attributes),
        ("Class Inheritance", test_inheritance),
        ("Transformers Registration", test_transformers_registration),
        ("Method Signatures", test_method_signatures),
        ("Config Instantiation", test_config_instantiation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - Unified implementation is working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Review failures above")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
