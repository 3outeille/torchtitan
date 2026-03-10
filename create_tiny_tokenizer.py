#!/usr/bin/env python3
"""
Script to create a tiny tokenizer with 512 vocab size for testing.

Usage:
    python create_tiny_tokenizer.py
"""

import json
import os
from pathlib import Path


def create_tiny_tokenizer(vocab_size: int = 512, output_dir: str = "./tests/assets/tokenizer_tiny"):
    """
    Create a tiny BPE tokenizer with the specified vocab size.
    
    Args:
        vocab_size: Number of tokens in the vocabulary
        output_dir: Directory to save the tokenizer files
    """
    print(f"Creating tiny tokenizer with vocab_size={vocab_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build vocabulary
    # Start with byte-level tokens (256 bytes) + special tokens + common subwords
    vocab = {}
    token_id = 0
    
    # Add all single byte tokens (0-255) - this covers basic ASCII and byte fallback
    for i in range(256):
        if i < 33 or i == 127:  # Control characters
            # Use unicode escape for control chars
            vocab[f"<0x{i:02X}>"] = token_id
        elif i < 127:  # Printable ASCII
            char = chr(i)
            if char == '"':
                vocab['\\"'] = token_id
            elif char == '\\':
                vocab['\\\\'] = token_id
            else:
                vocab[char] = token_id
        else:  # Extended ASCII (128-255)
            vocab[f"<0x{i:02X}>"] = token_id
        token_id += 1
    
    # Add common subword tokens to fill up to vocab_size
    common_tokens = [
        # Whitespace variants (Ġ represents space in GPT-style tokenizers)
        "Ġ", "Ċ", "ĠĠ", "ĠĠĠĠ",
        # Common English words with space prefix
        "Ġthe", "Ġa", "Ġto", "Ġof", "Ġand", "Ġis", "Ġin", "Ġfor", "Ġthat", "Ġit",
        "Ġwith", "Ġas", "Ġwas", "Ġon", "Ġbe", "Ġare", "Ġhave", "Ġfrom", "Ġby", "Ġan",
        "Ġnot", "Ġthis", "Ġat", "Ġor", "Ġyou", "Ġcan", "Ġwill", "Ġall", "Ġwe", "Ġhas",
        "Ġbut", "Ġwhich", "Ġtheir", "Ġwere", "Ġmore", "Ġone", "Ġwould", "Ġabout",
        "Ġthere", "Ġwhen", "Ġif", "Ġother", "Ġbeen", "Ġalso", "Ġsome", "Ġhad", "Ġnew",
        "Ġthey", "Ġwho", "Ġtime", "Ġout", "Ġup", "Ġno", "Ġinto", "Ġthem", "Ġyear",
        "Ġhis", "Ġso", "Ġtwo", "Ġfirst", "Ġsaid", "Ġjust", "Ġover", "Ġthen", "Ġshe",
        "Ġher", "Ġhe", "Ġafter", "Ġmay", "Ġbefore", "Ġmost", "Ġnow", "Ġhim", "Ġonly",
        "Ġwhat", "Ġpeople", "Ġits", "Ġlike", "Ġhow", "Ġdo", "Ġmake", "Ġget", "Ġback",
        "Ġour", "Ġway", "Ġwork", "Ġthan", "Ġsee", "Ġbeing", "Ġlong", "Ġown", "Ġuse",
        # Common suffixes
        "ing", "ed", "er", "es", "ly", "tion", "al", "ment", "ness", "ous",
        "able", "ible", "ive", "ful", "less", "ity", "ism", "ist", "ize", "ify",
        # Common prefixes
        "re", "un", "in", "dis", "en", "non", "pre", "mis", "over", "sub",
        # Numbers
        "Ġ0", "Ġ1", "Ġ2", "Ġ3", "Ġ4", "Ġ5", "Ġ6", "Ġ7", "Ġ8", "Ġ9",
        "10", "11", "12", "20", "100", "1000",
        # ML/coding related
        "Ġdata", "Ġmodel", "Ġtrain", "Ġtest", "Ġloss", "Ġlayer", "Ġbatch",
        "Ġinput", "Ġoutput", "Ġweight", "Ġvalue", "Ġfunction", "Ġclass",
        "Ġdef", "Ġreturn", "Ġimport", "Ġfrom", "Ġself", "Ġtrue", "Ġfalse",
        # Special tokens
        "<|endoftext|>", "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>",
        "<|image|>", "<|begin|>", "<|end|>",
    ]
    
    # Add common tokens until we reach vocab_size
    for token in common_tokens:
        if token_id >= vocab_size:
            break
        if token not in vocab:
            vocab[token] = token_id
            token_id += 1
    
    # Fill remaining slots with numbered tokens
    while token_id < vocab_size:
        vocab[f"<extra_{token_id}>"] = token_id
        token_id += 1
    
    print(f"Created vocabulary with {len(vocab)} tokens")
    
    # Create tokenizer.json
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": vocab.get("<|endoftext|>", vocab_size - 5),
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
            {
                "id": vocab.get("<|pad|>", vocab_size - 4),
                "content": "<|pad|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
            {
                "id": vocab.get("<|unk|>", vocab_size - 3),
                "content": "<|unk|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<|unk|>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": []
        }
    }
    
    # Create tokenizer_config.json
    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            str(vocab.get("<|endoftext|>", vocab_size - 5)): {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            str(vocab.get("<|pad|>", vocab_size - 4)): {
                "content": "<|pad|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            str(vocab.get("<|unk|>", vocab_size - 3)): {
                "content": "<|unk|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
        },
        "bos_token": "<|bos|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|endoftext|>",
        "model_max_length": 131072,
        "pad_token": "<|pad|>",
        "tokenizer_class": "GPT2Tokenizer",
        "unk_token": "<|unk|>",
    }
    
    # Create special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|bos|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
    }
    
    # Save files
    tokenizer_json_path = Path(output_dir) / "tokenizer.json"
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
    print(f"Saved: {tokenizer_json_path}")
    
    tokenizer_config_path = Path(output_dir) / "tokenizer_config.json"
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"Saved: {tokenizer_config_path}")
    
    special_tokens_path = Path(output_dir) / "special_tokens_map.json"
    with open(special_tokens_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2)
    print(f"Saved: {special_tokens_path}")
    
    print(f"\nTiny tokenizer created successfully in: {output_dir}")
    print(f"Vocab size: {vocab_size}")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a tiny tokenizer for testing")
    parser.add_argument("--vocab-size", type=int, default=512, help="Vocabulary size")
    parser.add_argument("--output-dir", type=str, default="./tests/assets/tokenizer_tiny",
                       help="Output directory for tokenizer files")
    
    args = parser.parse_args()
    
    create_tiny_tokenizer(vocab_size=args.vocab_size, output_dir=args.output_dir)

