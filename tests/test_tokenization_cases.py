import pytest
from src.tokenizer.manager import TokenizerManager

# Define test cases here
# Format: (model_name, source, text, expected_count, expected_tokens, expected_ids)
TEST_CASES = [
    (
        "gpt-4o", 
        "tiktoken", 
        "Hello world", 
        2, 
        ["Hello", " world"], 
        [13225, 2375]
    ),
    (
        "gpt-4o", 
        "tiktoken", 
        "こんにちは", 
        1, 
        ["こんにちは"], 
        [95839]
    ),
    (
        "gpt-4", 
        "tiktoken", 
        "Hello world", 
        2, 
        ["Hello", " world"], 
        [9906, 1917]
    ),
    (
        "bert-base-uncased",
        "huggingface",
        "Hello world",
        2,
        ["Hello", "world"], # Preserves original casing because we use offsets
        [7592, 2088]
    )
]

@pytest.mark.parametrize("model_name, source, text, expected_count, expected_tokens, expected_ids", TEST_CASES)
def test_specific_tokenization(model_name, source, text, expected_count, expected_tokens, expected_ids):
    manager = TokenizerManager.get_instance()
    try:
        tokenizer = manager.get_tokenizer(model_name, source)
    except Exception as e:
        pytest.skip(f"Could not load tokenizer {model_name}: {e}")

    result = tokenizer.encode(text)
    
    if expected_count is not None:
        assert result.count == expected_count
        
    if expected_tokens is not None:
        assert result.tokens == expected_tokens
        
    if expected_ids is not None:
        assert result.ids == expected_ids
