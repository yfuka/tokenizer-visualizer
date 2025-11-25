from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import tiktoken
from transformers import AutoTokenizer
import os
import json

class TokenGroup:
    def __init__(self, text: str, tokens: List[str], ids: List[int]):
        self.text = text
        self.tokens = tokens
        self.ids = ids
    
    @property
    def is_split(self) -> bool:
        return len(self.tokens) > 1

class TokenizationResult:
    def __init__(self, tokens: List[str], ids: List[int], offsets: List[tuple[int, int]] = None, grouped_tokens: List[TokenGroup] = None):
        self.tokens = tokens
        self.ids = ids
        self.offsets = offsets # (start, end) character indices
        self.grouped_tokens = grouped_tokens or []

    @property
    def count(self) -> int:
        return len(self.ids)

class TokenizerWrapper(ABC):
    @abstractmethod
    def encode(self, text: str) -> TokenizationResult:
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

def _build_byte_to_char_map(text: str) -> Dict[int, int]:
    """
    Builds a map where key is the byte index and value is the character index.
    This is needed for Tiktoken to map byte offsets back to character indices.
    """
    byte_map = {}
    byte_idx = 0
    for char_idx, char in enumerate(text):
        char_bytes = char.encode('utf-8')
        for _ in char_bytes:
            byte_map[byte_idx] = char_idx
            byte_idx += 1
    # Add a sentinel for the end
    byte_map[byte_idx] = len(text)
    return byte_map

def _group_tokens(tokens: List[str], ids: List[int], offsets: List[tuple[int, int]]) -> List[TokenGroup]:
    """
    Groups tokens that map to the same character range.
    """
    if not tokens:
        return []

    groups = []
    current_tokens = []
    current_ids = []
    current_range = None

    for token, token_id, offset in zip(tokens, ids, offsets):
        # If this is the first token or matches the current group's range
        if current_range is None:
            current_tokens.append(token)
            current_ids.append(token_id)
            current_range = offset
        elif offset == current_range:
            current_tokens.append(token)
            current_ids.append(token_id)
        else:
            # Group changed, process previous group
            # Use the text from the first token as the group text (they are all the same char)
            groups.append(TokenGroup(current_tokens[0], current_tokens, current_ids))
            
            # Start new group
            current_tokens = [token]
            current_ids = [token_id]
            current_range = offset

    # Process final group
    if current_tokens:
        groups.append(TokenGroup(current_tokens[0], current_tokens, current_ids))

    return groups

class TiktokenWrapper(TokenizerWrapper):
    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback for generic GPT-4/3.5 if exact model name fails
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> TokenizationResult:
        ids = self.encoder.encode(text)
        
        raw_tokens = []
        offsets = []
        
        # Build byte to char map for accurate offset tracking
        byte_to_char = _build_byte_to_char_map(text)
        current_byte_idx = 0
        
        # Use decode_single_token_bytes to get the raw bytes for each token
        byte_tokens = [self.encoder.decode_single_token_bytes(token) for token in ids]
        
        for b_token in byte_tokens:
            token_len_bytes = len(b_token)
            start_byte = current_byte_idx
            end_byte = current_byte_idx + token_len_bytes
            
            # Map byte range to character range
            start_char = byte_to_char.get(start_byte, len(text))
            end_char_inclusive = byte_to_char.get(end_byte - 1, len(text))
            end_char = end_char_inclusive + 1
            
            token_str = text[start_char:end_char]
            
            raw_tokens.append(token_str)
            offsets.append((start_char, end_char))
            
            current_byte_idx = end_byte
            
        # Group tokens
        grouped_tokens = _group_tokens(raw_tokens, ids, offsets)
        
        return TokenizationResult(raw_tokens, ids, offsets, grouped_tokens)

    def decode(self, ids: List[int]) -> str:
        return self.encoder.decode(ids)

    @property
    def name(self) -> str:
        return f"tiktoken ({self.model_name})"

def get_bytes_to_unicode_map():
    """
    Returns the bytes_to_unicode map used by GPT-2/RoBERTa etc.
    Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py
    """
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class HuggingFaceWrapper(TokenizerWrapper):
    def __init__(self, model_name_or_path: str):
        self.model_name = model_name_or_path
        
        # Check if it's a local file (likely tokenizer.json)
        if os.path.isfile(model_name_or_path) and model_name_or_path.endswith(".json"):
            from transformers import PreTrainedTokenizerFast
            # Load directly from the JSON file
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=model_name_or_path)
        elif os.path.isdir(model_name_or_path):
             # Load from directory (standard HF format)
             try:
                 self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
             except Exception:
                 # Fallback: Try loading directly from tokenizer.json if it exists
                 # This handles cases where config.json is missing or doesn't have model_type
                 tokenizer_json = os.path.join(model_name_or_path, "tokenizer.json")
                 if os.path.exists(tokenizer_json):
                     from transformers import PreTrainedTokenizerFast
                     self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
                 else:
                     raise
        else:
            # fast=True is required for offsets
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        
        # Detect if this tokenizer uses ByteLevelBPE
        self.is_byte_level = self._is_byte_level_bpe()
        
        # Initialize byte decoder map for GPT-2 style models (only if ByteLevelBPE)
        if self.is_byte_level:
            byte_encoder = get_bytes_to_unicode_map()
            self.byte_decoder = {v: k for k, v in byte_encoder.items()}
        else:
            self.byte_decoder = {}
    
    def _is_byte_level_bpe(self) -> bool:
        """Check if the tokenizer uses ByteLevelBPE (like GPT-2/RoBERTa)."""
        # Check backend tokenizer model type
        if hasattr(self.tokenizer, 'backend_tokenizer'):
            bt = self.tokenizer.backend_tokenizer
            if hasattr(bt, 'model'):
                model_type_str = str(type(bt.model))
                # ByteLevelBPE appears as 'tokenizers.models.BPE' with a byte_level processor
                # Standard BPE also appears as 'tokenizers.models.BPE'
                # We need to check for byte_level in the pre_tokenizer or decoder
                if hasattr(bt, 'pre_tokenizer'):
                    pre_tok_str = str(type(bt.pre_tokenizer))
                    if 'ByteLevel' in pre_tok_str:
                        return True
                if hasattr(bt, 'decoder'):
                    decoder_str = str(type(bt.decoder))
                    if 'ByteLevel' in decoder_str:
                        return True
        return False

    def encode(self, text: str) -> TokenizationResult:
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = encoding.input_ids
        offsets = encoding.offset_mapping
        
        raw_tokens = []
        for i, (start, end) in enumerate(offsets):
            # Use the offsets to get the source text
            # This works even for split tokens because HF returns the full character span
            # for each token that is part of that character.
            
            # Special handling for special tokens which might have (0,0) offsets
            if start == 0 and end == 0:
                # Check if it's actually the first char or a special token
                token_str = text[start:end]
                if not token_str:
                    # Try to decode to see what it is (e.g. <s>)
                    decoded = self.tokenizer.decode([ids[i]])
                    raw_tokens.append(decoded)
                else:
                    raw_tokens.append(token_str)
            else:
                token_str = text[start:end]
                raw_tokens.append(token_str)
        
        # Group tokens
        grouped_tokens = _group_tokens(raw_tokens, ids, offsets)
        
        return TokenizationResult(raw_tokens, ids, offsets, grouped_tokens)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def name(self) -> str:
        return f"HF ({os.path.basename(self.model_name)})"

class TokenizerManager:
    _instance = None
    
    def __init__(self):
        self._cache: Dict[str, TokenizerWrapper] = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_tokenizer(self, model_name: str, source: str = "tiktoken") -> TokenizerWrapper:
        key = f"{source}:{model_name}"
        if key in self._cache:
            return self._cache[key]

        if source == "tiktoken":
            wrapper = TiktokenWrapper(model_name)
        elif source == "huggingface":
            wrapper = HuggingFaceWrapper(model_name)
        elif source == "local":
             # For local, model_name is the path
             wrapper = HuggingFaceWrapper(model_name)
        else:
            raise ValueError(f"Unknown source: {source}")

        self._cache[key] = wrapper
        return wrapper

    def load_local_tokenizer(self, path: str) -> TokenizerWrapper:
        """Loads a tokenizer from a local directory."""
        return self.get_tokenizer(path, source="local")
