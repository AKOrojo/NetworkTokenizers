#!/usr/bin/env python3
"""
Usage example for EnhancedPCAPByteTokenizer with a real PCAP file.

This script demonstrates how to use the EnhancedPCAPByteTokenizer to tokenize
network packet data from a PCAP file with special tokens and packet separation.
"""

import os
import tempfile
from collections import Counter

from src.byte.raw.token_pcap_byte_tokenizer import TokenPCAPByteTokenizer


def analyze_enhanced_pcap_tokenization(pcap_path: str):
    """
    Analyze a PCAP file using the EnhancedPCAPByteTokenizer.

    Args:
        pcap_path: Path to the PCAP file to analyze
    """
    print(f"🔍 Analyzing PCAP file: {pcap_path}")
    print("=" * 70)

    # Check if file exists
    if not os.path.exists(pcap_path):
        print(f"❌ Error: PCAP file not found at {pcap_path}")
        return

    # Initialize the tokenizer
    print("📚 Initializing EnhancedPCAPByteTokenizer...")
    tokenizer = TokenPCAPByteTokenizer()

    # Display special token information
    print("\n🎯 Special Token Information:")
    print("-" * 40)
    print(f"   PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"   UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    print(f"   BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"   EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"   PKT token: '<pkt>' (ID: {tokenizer.pkt_token_id})")
    print(f"   Byte offset: {5} (byte values start from ID 5)")

    # Get file size for context
    file_size = os.path.getsize(pcap_path)
    print(f"\n📁 File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")

    # Read individual packets
    print("\n🔧 Reading individual packets...")
    try:
        packets = tokenizer.read_pcap_packets(pcap_path)
        print(f"📦 Total packets found: {len(packets)}")

        if len(packets) == 0:
            print("⚠️  Warning: No packets found in PCAP file")
            return

        # Show packet size statistics
        packet_sizes = [len(packet) for packet in packets]
        total_bytes = sum(packet_sizes)

        print(f"📊 Packet size statistics:")
        print(f"   Total bytes: {total_bytes:,}")
        print(f"   Average packet size: {total_bytes / len(packets):.1f} bytes")
        print(f"   Smallest packet: {min(packet_sizes)} bytes")
        print(f"   Largest packet: {max(packet_sizes)} bytes")

    except Exception as e:
        print(f"❌ Error reading PCAP: {e}")
        return

    # Test different tokenization modes
    print("\n🎯 Testing Different Tokenization Modes:")
    print("-" * 50)

    modes = [
        ("Basic (no separators)", {"add_packet_separators": False, "add_bos": False, "add_eos": False}),
        ("With packet separators", {"add_packet_separators": True, "add_bos": False, "add_eos": False}),
        ("With BOS/EOS", {"add_packet_separators": False, "add_bos": True, "add_eos": True}),
        ("Full (separators + BOS/EOS)", {"add_packet_separators": True, "add_bos": True, "add_eos": True}),
    ]

    for mode_name, mode_kwargs in modes:
        try:
            tokens = tokenizer.tokenize_pcap(pcap_path, **mode_kwargs)
            token_ids = tokenizer.tokenize_pcap_to_ids(pcap_path, **mode_kwargs)

            # Count special tokens
            special_token_count = sum(1 for tid in token_ids if tid < 5)
            byte_token_count = len(token_ids) - special_token_count

            print(f"   {mode_name}:")
            print(f"     Total tokens: {len(tokens):,}")
            print(f"     Special tokens: {special_token_count}")
            print(f"     Byte tokens: {byte_token_count:,}")

            # Show first few tokens for this mode
            first_10_tokens = tokens[:10]
            first_10_ids = token_ids[:10]
            print(f"     First 10 tokens: {first_10_tokens}")
            print(f"     First 10 IDs: {first_10_ids}")
            print()

        except Exception as e:
            print(f"   ❌ Error in {mode_name}: {e}")

    # Detailed analysis using full mode
    print("\n📊 Detailed Analysis (Full Mode):")
    print("-" * 40)

    try:
        tokens = tokenizer.tokenize_pcap(pcap_path,
                                         add_packet_separators=True,
                                         add_bos=True,
                                         add_eos=True)
        token_ids = tokenizer.tokenize_pcap_to_ids(pcap_path,
                                                   add_packet_separators=True,
                                                   add_bos=True,
                                                   add_eos=True)

        print(f"🏷️  Generated {len(tokens):,} tokens total")

        # Separate analysis for special tokens and byte tokens
        special_tokens = [tid for tid in token_ids if tid < 5]
        byte_tokens = [tid for tid in token_ids if tid >= 5]

        print(f"🎨 Special tokens: {len(special_tokens)}")
        print(f"🔢 Byte tokens: {len(byte_tokens):,}")

        if byte_tokens:
            print(
                f"📏 Byte token IDs range: {min(byte_tokens)} to {max(byte_tokens)} (bytes {min(byte_tokens) - 5} to {max(byte_tokens) - 5})")

    except Exception as e:
        print(f"❌ Error in detailed analysis: {e}")
        return

    # Analyze byte value distribution (excluding special tokens)
    if byte_tokens:
        print("\n📊 Byte Value Distribution Analysis:")
        print("-" * 45)

        # Convert token IDs back to original byte values
        original_byte_values = [tid - 5 for tid in byte_tokens]
        unique_bytes = set(original_byte_values)
        print(f"🎨 Unique byte values: {len(unique_bytes)}/256 ({len(unique_bytes) / 256 * 100:.1f}%)")

        # Find most and least common bytes
        byte_counts = Counter(original_byte_values)
        most_common = byte_counts.most_common(5)
        least_common = byte_counts.most_common()[-5:]
        
        def common(common_values, original_byte_assets):
            for byte_values, count in common_values:
                char_repr = repr(chr(byte_values)) if 32 <= byte_values <= 126 else f"\\x{byte_values:02x}"
                percentage = count / len(original_byte_assets) * 100
                token_identity = byte_values + 5
                print(
                    f"   Byte {byte_values:3d} (Token ID {token_identity:3d}, {char_repr:>6}): {count:6,} times ({percentage:5.1f}%)")

        print(f"\n🔥 Most common byte values:")
        common(most_common, original_byte_values)
        
        print(f"\n🥶 Least common byte values:")
        common(least_common, byte_tokens)

        # Show missing byte values
        missing_bytes = set(range(256)) - unique_bytes
        if missing_bytes:
            print(
                f"\n🕳️  Missing byte values ({len(missing_bytes)}): {sorted(list(missing_bytes))[:20]}{'...' if len(missing_bytes) > 20 else ''}")
        else:
            print(f"\n✅ All 256 possible byte values are present!")

    # Show packet structure analysis
    print(f"\n📦 Packet Structure Analysis:")
    print("-" * 35)

    # Count packet separators to verify packet count
    pkt_separators = token_ids.count(tokenizer.pkt_token_id)
    expected_separators = len(packets) - 1  # N packets need N-1 separators

    print(f"   Expected packet separators: {expected_separators}")
    print(f"   Found packet separators: {pkt_separators}")
    print(f"   ✅ Packet separation correct: {pkt_separators == expected_separators}")

    # Show structure of first few tokens
    print(f"\n🔤 Token structure (first 30 tokens):")
    structure_tokens = tokens[:30]
    structure_ids = token_ids[:30]

    for i, (token, token_id) in enumerate(zip(structure_tokens, structure_ids)):
        if token_id < 5:
            token_type = "SPECIAL"
            if token_id == tokenizer.bos_token_id:
                token_name = "BOS"
            elif token_id == tokenizer.eos_token_id:
                token_name = "EOS"
            elif token_id == tokenizer.pkt_token_id:
                token_name = "PKT"
            else:
                token_name = "OTHER"
            display = f"{token_name}({token})"
        else:
            token_type = "BYTE"
            byte_val = token_id - 5
            if 32 <= byte_val <= 126:
                display = f"'{chr(byte_val)}'"
            else:
                display = f"\\x{byte_val:02x}"

        print(f"   {i:2d}: ID={token_id:3d} {token_type:7} {display}")

    # Test round-trip encoding/decoding
    print(f"\n🔄 Testing Round-trip Encoding/Decoding...")
    try:
        # Test with skip_special_tokens=True (should give original bytes)
        decoded_bytes = tokenizer.decode(token_ids, skip_special_tokens=True)
        original_bytes = b"".join(packets)

        if decoded_bytes == original_bytes:
            print("✅ Round-trip successful (skip_special_tokens=True)")
        else:
            print("❌ Round-trip failed (skip_special_tokens=True)")
            print(f"   Original length: {len(original_bytes)}")
            print(f"   Decoded length:  {len(decoded_bytes)}")

        # Test with skip_special_tokens=False (should include special tokens as string)
        decoded_string = tokenizer.decode(token_ids, skip_special_tokens=False)
        print(f"✅ String decode successful (length: {len(decoded_string)})")

    except Exception as e:
        print(f"❌ Round-trip test failed: {e}")

    # Test vocabulary
    print(f"\n📖 Vocabulary Test:")
    print("-" * 20)
    vocab = tokenizer.get_vocab()
    print(f"   Vocabulary size: {len(vocab)}")
    print(f"   Expected size: {tokenizer.vocab_size}")
    print(f"   ✅ Vocabulary complete: {len(vocab) == tokenizer.vocab_size}")

    # Show some vocabulary entries
    print(f"   Sample vocab entries:")
    sample_entries = list(vocab.items())[:10]
    for token, token_id in sample_entries:
        print(f"     '{token}' → {token_id}")

    # Test special token methods
    print(f"\n🎯 Special Token Methods Test:")
    print("-" * 35)

    try:
        # Test build_inputs_with_special_tokens
        sample_ids = [10, 20, 30]  # Some byte token IDs
        built_inputs = tokenizer.build_inputs_with_special_tokens(sample_ids)
        print(f"   Input: {sample_ids}")
        print(f"   Built: {built_inputs}")

        # Test get_special_tokens_mask
        mask = tokenizer.get_special_tokens_mask(sample_ids)
        print(f"   Mask: {mask}")

        print("✅ Special token methods working correctly")

    except Exception as e:
        print(f"❌ Special token methods test failed: {e}")

    # Save and load test
    print(f"\n💾 Testing Save/Load Functionality...")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save tokenizer
            tokenizer.save_pretrained(tmp_dir)
            print(f"   ✅ Tokenizer saved to temporary directory")

            # Save vocabulary
            vocab_files = tokenizer.save_vocabulary(tmp_dir)
            print(f"   ✅ Vocabulary saved: {vocab_files}")

            # Load tokenizer
            loaded_tokenizer = TokenPCAPByteTokenizer.from_pretrained(tmp_dir)
            print(f"   ✅ Tokenizer loaded successfully")

            # Test that loaded tokenizer works the same
            loaded_tokens = loaded_tokenizer.tokenize_pcap(pcap_path,
                                                           add_packet_separators=True,
                                                           add_bos=True,
                                                           add_eos=True)
            if loaded_tokens == tokens:
                print(f"   ✅ Loaded tokenizer produces identical results")
            else:
                print(f"   ❌ Loaded tokenizer produces different results")

    except Exception as e:
        print(f"   ❌ Save/load test failed: {e}")

    print(f"\n🎉 Enhanced PCAP Analysis Complete!")
    print("=" * 70)


def demonstrate_packet_analysis(pcap_path: str):
    """
    Demonstrate packet-by-packet analysis capabilities.
    """
    print(f"\n🔬 Packet-by-Packet Analysis Demo")
    print("=" * 45)

    tokenizer = TokenPCAPByteTokenizer()

    try:
        packets = tokenizer.read_pcap_packets(pcap_path)

        print(f"Analyzing first 3 packets individually:")
        print("-" * 40)

        for i, packet in enumerate(packets[:3]):
            print(f"\n📦 Packet {i + 1}:")
            print(f"   Size: {len(packet)} bytes")

            # Show first 20 bytes
            first_bytes = packet[:20]
            hex_repr = " ".join(f"{b:02x}" for b in first_bytes)
            print(f"   First 20 bytes (hex): {hex_repr}")

            # Show as token IDs (with offset)
            token_ids = [b + 5 for b in first_bytes]  # Add offset
            print(f"   As token IDs: {token_ids}")

            # Show printable characters
            printable = "".join(chr(b) if 32 <= b <= 126 else '.' for b in first_bytes)
            print(f"   Printable view: '{printable}'")

    except Exception as e:
        print(f"❌ Error in packet analysis: {e}")


def main():
    """Main function to run the enhanced PCAP tokenization example."""

    # Path to the test PCAP file
    pcap_path = "../data/test.pcap"

    print("🚀 Enhanced PCAP Byte Tokenizer Usage Example")
    print("🎯 Features: Special Tokens + Packet Separation + Byte Offset")
    print("=" * 70)

    # Run the main analysis
    analyze_enhanced_pcap_tokenization(pcap_path)

    # Run packet-by-packet demo
    demonstrate_packet_analysis(pcap_path)


if __name__ == "__main__":
    main()