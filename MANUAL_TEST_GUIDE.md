# Manual Testing Guide for Unified Language Model Implementation

This guide provides instructions for manually testing the unified language model implementation to verify it works correctly in your environment.

## Quick Start

Run the provided test script:

```bash
python test_unified_manual.py
```

This will run 7 comprehensive tests to verify:
1. ✅ All 21 classes can be imported
2. ✅ Backward compatible imports work
3. ✅ Class attributes are correct
4. ✅ Inheritance is proper
5. ✅ Models are registered with transformers
6. ✅ Method signatures are correct
7. ✅ Configs can be instantiated

## Expected Output

You should see output like:

```
╔══════════════════════════════════════════════════════════════════════╗
║               UNIFIED MODEL MANUAL TEST SUITE                        ║
╚══════════════════════════════════════════════════════════════════════╝

==================================================================
TEST 1: Import All Classes
==================================================================
✓ All 21 classes imported successfully
  - 7 Config classes
  - 7 Model classes
  - 7 CausalLM classes

... (more tests) ...

==================================================================
TEST SUMMARY
==================================================================
✓ PASS   | Import All Classes
✓ PASS   | Backward Compatible Imports
✓ PASS   | Class Attributes
✓ PASS   | Class Inheritance
✓ PASS   | Transformers Registration
✓ PASS   | Method Signatures
✓ PASS   | Config Instantiation
==================================================================
Results: 7/7 tests passed

✅ ALL TESTS PASSED - Unified implementation is working correctly!
```

## Manual Model Loading Test

To test actual model loading (requires model weights):

```python
from transformers import AutoConfig, AutoModelForCausalLM

# Test loading a LLaMA-based model
config = AutoConfig.from_pretrained("your-model-path")
model = AutoModelForCausalLM.from_pretrained("your-model-path")

print(f"Model type: {config.model_type}")
print(f"Model class: {model.__class__.__name__}")
```

## Testing Specific Models

### Test LLaMA
```python
from llava.model import LlavaLlamaForCausalLM, LlavaConfig
config = LlavaConfig()
# Verify config has expected attributes
assert hasattr(config, 'temperature')
assert hasattr(config, 'max_new_tokens')
```

### Test MPT Special Features
```python
from llava.model import LlavaMptForCausalLM, LlavaMptModel
# Verify MPT-specific features
assert hasattr(LlavaMptForCausalLM, 'supports_gradient_checkpointing')
assert hasattr(LlavaMptModel, 'embed_tokens')
```

### Test DPO Models
```python
from llava.model import LlavaLlamaForCausalLM, LlavaMixtralForCausalLM
import inspect

# Verify DPO support
llama_sig = inspect.signature(LlavaLlamaForCausalLM.forward)
assert 'dpo_forward' in llama_sig.parameters
assert 'modalities' in llama_sig.parameters
```

## Testing with Existing Code

The refactoring is fully backward compatible. Test by running your existing code:

```python
# Old import style - should still work
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

# New import style - also works
from llava.model.language_model.llava_unified import LlavaLlamaForCausalLM

# Recommended import style
from llava.model import LlavaLlamaForCausalLM
```

All three import styles access the same class from the unified implementation.

## Integration Testing

Run your existing training/inference scripts to verify:

1. **Model Loading**: Existing checkpoints load correctly
2. **Training**: Training scripts run without errors
3. **Inference**: Inference produces expected outputs
4. **DPO Training**: DPO training works for supported models

## Troubleshooting

### Import Errors
If you get import errors:
```python
ModuleNotFoundError: No module named 'torch'
```
Ensure you have installed dependencies:
```bash
pip install torch transformers
```

### Config Missing Attributes
Some configs may require parameters during instantiation. This is expected behavior inherited from the base transformers configs.

### Attribute Errors
If you see errors like:
```
AttributeError: 'LlavaMptModel' object has no attribute 'wte'
```
This likely means the model hasn't been properly initialized with its base class.

## What to Report

If tests fail, please report:
1. Which test failed
2. Full error traceback
3. Python version: `python --version`
4. Transformers version: `pip show transformers`
5. Environment (local, cloud, Docker, etc.)

## Success Criteria

For the manual test to be considered successful:
- ✅ All 7 tests in `test_unified_manual.py` pass
- ✅ Can import all 21 classes
- ✅ Backward compatible imports work
- ✅ Existing code runs without modifications
- ✅ Model loading/inference works (if you have model weights)

## Additional Verification

### Check File Sizes
```bash
wc -l llava/model/language_model/llava_*.py
```

You should see:
- `llava_unified.py`: ~500 lines
- Original files: ~150 lines each (kept for reference)

### Verify No Breaking Changes
Run your existing test suite if you have one. All tests should pass without modification.

## Need Help?

If you encounter issues:
1. Check the error message carefully
2. Verify dependencies are installed
3. Try the troubleshooting steps above
4. Report the issue with full details

## Performance Testing

While functionality is preserved, you may want to verify performance:
1. Model loading time (should be similar)
2. Inference speed (should be identical)
3. Memory usage (should be identical)

The factory pattern has negligible overhead as classes are generated once at module import time.
