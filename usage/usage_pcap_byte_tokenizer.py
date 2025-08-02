#!/usr/bin/env python3
"""
Usage example for PCAPByteTokenizer with a real PCAP file.

This script demonstrates how to use the PCAPByteTokenizer to tokenize
network packet data from a PCAP file.
"""

import os
import tempfile
from collections import Counter

from src.byte.raw.pcap_byte_tokenizer import PCAPByteTokenizer


def analyze_pcap_tokenization(pcap_path: str):
    """
    Analyze a PCAP file using the PCAPByteTokenizer.

    Args:
        pcap_path: Path to the PCAP file to analyze
    """
    print(f"🔍 Analyzing PCAP file: {pcap_path}")
    print("=" * 60)

    # Check if file exists
    if not os.path.exists(pcap_path):
        print(f"❌ Error: PCAP file not found at {pcap_path}")
        return

    # Initialize the tokenizer
    print("📚 Initializing PCAPByteTokenizer...")
    tokenizer = PCAPByteTokenizer()

    # Get file size for context
    file_size = os.path.getsize(pcap_path)
    print(f"📁 File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")

    # Read raw PCAP bytes
    print("\n🔧 Reading raw packet bytes...")
    try:
        raw_bytes = tokenizer.read_pcap_bytes(pcap_path)
        print(f"📦 Total packet bytes extracted: {len(raw_bytes):,}")

        if len(raw_bytes) == 0:
            print("⚠️  Warning: No packet data found in PCAP file")
            return

    except Exception as e:
        print(f"❌ Error reading PCAP: {e}")
        return

    # Tokenize the PCAP
    print("\n🎯 Tokenizing PCAP data...")
    try:
        tokens = tokenizer.tokenize(pcap_path)
        print(f"🏷️  Generated {len(tokens):,} tokens")

        # Convert to token IDs

        token_ids = tokenizer.tokenize_pcap_to_ids(pcap_path)
        print(f"🔢 Token IDs range: {min(token_ids)} to {max(token_ids)}")

    except Exception as e:
        print(f"❌ Error tokenizing PCAP: {e}")
        return

    # Analyze token distribution
    print("\n📊 Token Analysis:")
    print("-" * 30)

    # Count unique byte values
    unique_bytes = set(token_ids)
    print(f"🎨 Unique byte values: {len(unique_bytes)}/256 ({len(unique_bytes) / 256 * 100:.1f}%)")

    # Find most and least common bytes
    byte_counts = Counter(token_ids)
    most_common = byte_counts.most_common(5)
    least_common = byte_counts.most_common()[-5:]

    print(f"\n🔥 Most common bytes:")
    for byte_val, count in most_common:
        char_repr = repr(chr(byte_val)) if 32 <= byte_val <= 126 else f"\\x{byte_val:02x}"
        percentage = count / len(token_ids) * 100
        print(f"   Byte {byte_val:3d} ({char_repr:>4}): {count:6,} times ({percentage:5.1f}%)")

    print(f"\n🥶 Least common bytes:")
    for byte_val, count in least_common:
        char_repr = repr(chr(byte_val)) if 32 <= byte_val <= 126 else f"\\x{byte_val:02x}"
        percentage = count / len(token_ids) * 100
        print(f"   Byte {byte_val:3d} ({char_repr:>4}): {count:6,} times ({percentage:5.1f}%)")

    # Show missing byte values
    missing_bytes = set(range(256)) - unique_bytes
    if missing_bytes:
        print(f"\n🕳️  Missing byte values ({len(missing_bytes)}): {sorted(list(missing_bytes))}")
    else:
        print(f"\n✅ All 256 possible byte values are present!")

    # Show first few tokens as example
    print(f"\n🔤 First 20 tokens (as characters):")
    first_20_chars = []
    for i, token_id in enumerate(token_ids[:20]):
        if 32 <= token_id <= 126:  # Printable ASCII
            first_20_chars.append(chr(token_id))
        else:
            first_20_chars.append(f"\\x{token_id:02x}")
    print(f"   {''.join(first_20_chars)}")

    print(f"\n🔢 First 20 token IDs:")
    print(f"   {token_ids[:20]}")

    # Test round-trip encoding/decoding
    print(f"\n🔄 Testing round-trip encoding/decoding...")
    try:
        decoded_bytes = tokenizer.decode(token_ids)
        if decoded_bytes == raw_bytes:
            print("✅ Round-trip successful: Original bytes == Decoded bytes")
        else:
            print("❌ Round-trip failed: Original bytes != Decoded bytes")
            print(f"   Original length: {len(raw_bytes)}")
            print(f"   Decoded length:  {len(decoded_bytes)}")
    except Exception as e:
        print(f"❌ Round-trip test failed: {e}")

    # Test vocabulary
    print(f"\n📖 Vocabulary test:")
    vocab = tokenizer.get_vocab()
    print(f"   Vocabulary size: {len(vocab)}")
    print(f"   Expected size: {tokenizer.vocab_size}")
    print(f"   ✅ Vocabulary complete: {len(vocab) == tokenizer.vocab_size}")

    # Save and load test
    print(f"\n💾 Testing save/load functionality...")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save tokenizer
            tokenizer.save_pretrained(tmp_dir)
            print(f"   ✅ Tokenizer saved to temporary directory")

            # Load tokenizer
            loaded_tokenizer = PCAPByteTokenizer.from_pretrained(tmp_dir)
            print(f"   ✅ Tokenizer loaded successfully")

            # Test that loaded tokenizer works the same
            loaded_tokens = loaded_tokenizer.tokenize(pcap_path)
            if loaded_tokens == tokens:
                print(f"   ✅ Loaded tokenizer produces identical results")
            else:
                print(f"   ❌ Loaded tokenizer produces different results")

    except Exception as e:
        print(f"   ❌ Save/load test failed: {e}")

    print(f"\n🎉 Analysis complete!")
    print("=" * 60)


def main():
    """Main function to run the PCAP tokenization example."""

    # Path to the test PCAP file
    pcap_path = "../data/test.pcap"

    print("🚀 PCAP Byte Tokenizer Usage Example")
    print("=" * 60)

    # Run the analysis
    analyze_pcap_tokenization(pcap_path)


if __name__ == "__main__":
    main()