#!/usr/bin/env python3
"""
Usage example for Enhanced PCAP Byte Tokenizer with field separation capabilities.

This script demonstrates how to use the enhanced TokenPCAPByteTokenizer to tokenize
network packet data from a PCAP file with special tokens, packet separation, and
protocol field separation using <sep> tokens.
"""

import json
import os
from collections import Counter

from src.byte.raw.field_aware_pcap_byte_tokenizer import FieldAwarePCAPByteTokenizer, SeparatorConfig


def analyze_enhanced_pcap_tokenization(pcap_path: str):
    """
    Analyze a PCAP file using the Enhanced PCAP Byte Tokenizer with field separation.

    Args:
        pcap_path: Path to the PCAP file to analyze
    """
    print(f"🔍 Analyzing PCAP file with Field Separation: {pcap_path}")
    print("=" * 80)

    # Check if file exists
    if not os.path.exists(pcap_path):
        print(f"❌ Error: PCAP file not found at {pcap_path}")
        return

    # Create different separator configurations for testing
    configs = {
        "Conservative": SeparatorConfig(
            policy="conservative",
            max_depth=4,
            insert_ethernet_fields=True,
            insert_ip_fields=True,
            insert_transport_fields=True
        ),
        "Hybrid": SeparatorConfig(
            policy="hybrid",
            max_depth=3,
            insert_ethernet_fields=True,
            insert_ip_fields=False,
            insert_transport_fields=False
        ),
        "Aggressive": SeparatorConfig(
            policy="aggressive",
            max_depth=4,
            insert_ethernet_fields=True,
            insert_ip_fields=True,
            insert_transport_fields=True
        )
    }

    # Test each configuration
    for config_name, config in configs.items():
        print(f"\n🛠️  Testing {config_name} Configuration:")
        print("-" * 50)

        # Initialize tokenizer with specific configuration
        malformed_log = f"malformed_{config_name.lower()}.log"
        tokenizer = FieldAwarePCAPByteTokenizer(
            separator_config=config,
            malformed_log_path=malformed_log
        )

        # Display special token information
        print(f"📚 Special Token Information:")
        print(f"   PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        print(f"   UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
        print(f"   BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
        print(f"   EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
        print(f"   PKT token: '<pkt>' (ID: {tokenizer.pkt_token_id})")
        print(f"   SEP token: '<sep>' (ID: {tokenizer.sep_token_id})")
        print(f"   Byte offset: {6} (byte values start from ID 6)")
        print(f"   Vocabulary size: {tokenizer.vocab_size}")

        # Get file and packet info
        try:
            file_size = os.path.getsize(pcap_path)
            packets = tokenizer.read_pcap_packets(pcap_path)

            print(f"\n📁 File Information:")
            print(f"   File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
            print(f"   Total packets: {len(packets)}")

            if len(packets) == 0:
                print("⚠️  Warning: No packets found in PCAP file")
                continue

            # Packet size statistics
            packet_sizes = [len(packet) for packet in packets]
            total_bytes = sum(packet_sizes)
            print(f"   Total packet bytes: {total_bytes:,}")
            print(f"   Average packet size: {total_bytes / len(packets):.1f} bytes")
            print(f"   Size range: {min(packet_sizes)} - {max(packet_sizes)} bytes")

        except Exception as e:
            print(f"❌ Error reading PCAP: {e}")
            continue

        # Test tokenization modes with field separators
        print(f"\n🎯 Tokenization Modes ({config_name}):")
        print("-" * 40)

        modes = [
            ("Basic (no separators)", {
                "add_packet_separators": False,
                "add_field_separators": False,
                "add_bos": False,
                "add_eos": False
            }),
            ("Packet separators only", {
                "add_packet_separators": True,
                "add_field_separators": False,
                "add_bos": False,
                "add_eos": False
            }),
            ("Field separators only", {
                "add_packet_separators": False,
                "add_field_separators": True,
                "add_bos": False,
                "add_eos": False
            }),
            ("Full separation", {
                "add_packet_separators": True,
                "add_field_separators": True,
                "add_bos": True,
                "add_eos": True
            }),
        ]

        for mode_name, mode_kwargs in modes:
            try:
                tokens = tokenizer.tokenize_pcap(pcap_path, **mode_kwargs)
                token_ids = tokenizer.tokenize_pcap_to_ids(pcap_path, **mode_kwargs)

                # Count different token types
                special_token_counts = {
                    "BOS": token_ids.count(tokenizer.bos_token_id),
                    "EOS": token_ids.count(tokenizer.eos_token_id),
                    "PKT": token_ids.count(tokenizer.pkt_token_id),
                    "SEP": token_ids.count(tokenizer.sep_token_id),
                    "PAD": token_ids.count(tokenizer.pad_token_id),
                    "UNK": token_ids.count(tokenizer.unk_token_id),
                }

                total_special = sum(special_token_counts.values())
                byte_token_count = len(token_ids) - total_special

                print(f"   {mode_name}:")
                print(f"     Total tokens: {len(tokens):,}")
                print(f"     Byte tokens: {byte_token_count:,}")
                print(f"     Special tokens: {total_special} " +
                      f"(PKT:{special_token_counts['PKT']}, SEP:{special_token_counts['SEP']}, " +
                      f"BOS:{special_token_counts['BOS']}, EOS:{special_token_counts['EOS']})")

                # Show compression ratio
                compression_ratio = len(token_ids) / total_bytes if total_bytes > 0 else 0
                print(f"     Compression ratio: {compression_ratio:.3f} tokens/byte")

                # Show first few tokens for insight
                first_tokens = tokens[:15]
                first_ids = token_ids[:15]
                print(f"     First 15 tokens: {first_tokens}")
                print(f"     First 15 IDs: {first_ids}")
                print()

            except Exception as e:
                print(f"   ❌ Error in {mode_name}: {e}")

        # Check for malformed packet log
        if os.path.exists(malformed_log):
            with open(malformed_log, 'r') as f:
                malformed_entries = [json.loads(line.strip()) for line in f if line.strip()]

            if malformed_entries:
                print(f"⚠️  Malformed Packets Detected ({len(malformed_entries)} entries):")
                print("   Check log file for details:", malformed_log)

                # Show summary of error types
                error_types = Counter(entry['error_type'] for entry in malformed_entries)
                for error_type, count in error_types.items():
                    print(f"     {error_type}: {count} packets")
            else:
                print(f"✅ No malformed packets detected")
                # Clean up empty log file
                os.remove(malformed_log)
        else:
            print(f"✅ No malformed packets detected")

        print()


def demonstrate_field_separation_analysis(pcap_path: str):
    """
    Demonstrate detailed field separation analysis.
    """
    print(f"\n🔬 Field Separation Analysis Demo")
    print("=" * 50)

    # Use hybrid configuration for detailed analysis
    config = SeparatorConfig(
        policy="hybrid",
        max_depth=4,
        insert_ethernet_fields=True,
        insert_ip_fields=True,
        insert_transport_fields=True
    )

    tokenizer = FieldAwarePCAPByteTokenizer(
        separator_config=config,
        malformed_log_path="field_analysis_errors.log"
    )

    try:
        # Get tokens with full separation
        tokens = tokenizer.tokenize_pcap(
            pcap_path,
            add_packet_separators=True,
            add_field_separators=True,
            add_bos=True,
            add_eos=True
        )

        token_ids = tokenizer.tokenize_pcap_to_ids(
            pcap_path,
            add_packet_separators=True,
            add_field_separators=True,
            add_bos=True,
            add_eos=True
        )

        print(f"📊 Field Separation Statistics:")
        print(f"   Total tokens: {len(tokens):,}")

        # Analyze separator distribution
        sep_count = token_ids.count(tokenizer.sep_token_id)
        pkt_count = token_ids.count(tokenizer.pkt_token_id)

        packets = tokenizer.read_pcap_packets(pcap_path)
        print(f"   Packets: {len(packets)}")
        print(f"   Packet separators: {pkt_count} (expected: {len(packets) - 1})")
        print(f"   Field separators: {sep_count}")

        if len(packets) > 0:
            avg_seps_per_packet = sep_count / len(packets)
            print(f"   Average field separators per packet: {avg_seps_per_packet:.1f}")

        # Show structure of first packet with separators
        print(f"\n🔤 First Packet Structure Analysis:")
        print("-" * 40)

        # Find tokens between BOS and first PKT (or EOS if only one packet)
        start_idx = 0
        if tokens[0] == '<bos>':
            start_idx = 1

        end_idx = len(tokens)
        if '<pkt>' in tokens[start_idx:]:
            end_idx = start_idx + tokens[start_idx:].index('<pkt>')
        elif tokens[-1] == '<eos>':
            end_idx = len(tokens) - 1

        first_packet_tokens = tokens[start_idx:end_idx]
        first_packet_ids = token_ids[start_idx:end_idx]

        print(f"   First packet tokens ({len(first_packet_tokens)}): ")

        # Group tokens by fields (separated by <sep>)
        fields = []
        current_field = []

        for token, token_id in zip(first_packet_tokens, first_packet_ids):
            if token == '<sep>':
                if current_field:
                    fields.append(current_field)
                    current_field = []
            else:
                current_field.append((token, token_id))

        # Add last field
        if current_field:
            fields.append(current_field)

        for i, field in enumerate(fields[:8]):  # Show first 8 fields
            field_bytes = len(field)
            if field_bytes > 0:
                # Convert to hex representation
                byte_values = [tid - 6 for _, tid in field if tid >= 6]
                hex_repr = " ".join(f"{b:02x}" for b in byte_values[:8])
                if len(byte_values) > 8:
                    hex_repr += "..."

                print(f"     Field {i + 1}: {field_bytes} bytes - {hex_repr}")

        # Show protocol field boundaries
        print(f"\n🌐 Protocol Analysis:")
        print("-" * 25)

        # Analyze first few packets individually
        packets = tokenizer.read_pcap_packets(pcap_path)

        for i, packet in enumerate(packets[:3]):
            print(f"\n   Packet {i + 1} ({len(packet)} bytes):")

            # Manually tokenize this packet to see field boundaries
            packet_tokens = tokenizer._tokenize_packet_with_separators(
                packet, pcap_path, i
            )

            # Count separators in this packet
            packet_sep_count = packet_tokens.count('<sep>')
            print(f"     Field separators: {packet_sep_count}")

            # Show packet structure overview
            if len(packet) >= 14:  # Has Ethernet header
                eth_dst = packet[:6].hex()
                eth_src = packet[6:12].hex()
                eth_type = int.from_bytes(packet[12:14], 'big')
                print(f"     Ethernet: {eth_dst} → {eth_src} (type: 0x{eth_type:04x})")

                if eth_type == 0x0800 and len(packet) >= 34:  # IPv4
                    ip_src = ".".join(str(b) for b in packet[26:30])
                    ip_dst = ".".join(str(b) for b in packet[30:34])
                    ip_proto = packet[23]
                    print(f"     IPv4: {ip_src} → {ip_dst} (protocol: {ip_proto})")

    except Exception as e:
        print(f"❌ Error in field separation analysis: {e}")


def demonstrate_round_trip_with_separators(pcap_path: str):
    """
    Demonstrate round-trip encoding/decoding with field separators.
    """
    print(f"\n🔄 Round-trip Testing with Field Separators")
    print("=" * 50)

    config = SeparatorConfig(policy="hybrid", max_depth=4)
    tokenizer = FieldAwarePCAPByteTokenizer(separator_config=config)

    try:
        # Original packets
        original_packets = tokenizer.read_pcap_packets(pcap_path)
        original_bytes = b"".join(original_packets)

        print(f"📦 Original data: {len(original_bytes)} bytes from {len(original_packets)} packets")

        # Tokenize with separators
        token_ids_with_sep = tokenizer.tokenize_pcap_to_ids(
            pcap_path,
            add_packet_separators=True,
            add_field_separators=True,
            add_bos=True,
            add_eos=True
        )

        # Decode back to bytes (should skip special tokens)
        decoded_bytes = tokenizer.decode(token_ids_with_sep, skip_special_tokens=True)

        print(f"🔢 Tokenized: {len(token_ids_with_sep)} tokens")
        print(f"🔄 Decoded: {len(decoded_bytes)} bytes")

        # Verify round-trip accuracy
        if decoded_bytes == original_bytes:
            print("✅ Perfect round-trip: Original bytes recovered exactly")
        else:
            print("❌ Round-trip error: Decoded bytes differ from original")
            print(f"   Original length: {len(original_bytes)}")
            print(f"   Decoded length: {len(decoded_bytes)}")

            # Find first difference
            for i, (orig, dec) in enumerate(zip(original_bytes, decoded_bytes)):
                if orig != dec:
                    print(f"   First difference at byte {i}: {orig} vs {dec}")
                    break

        # Test string decode (with special tokens)
        decoded_string = tokenizer.decode(token_ids_with_sep, skip_special_tokens=False)
        print(f"📝 String decode length: {len(decoded_string)} characters")

        # Count special tokens in string
        special_count = decoded_string.count('<pkt>') + decoded_string.count('<sep>')
        print(f"   Special tokens in string: {special_count}")

    except Exception as e:
        print(f"❌ Round-trip test failed: {e}")


def demonstrate_vocabulary_analysis(pcap_path: str):
    """
    Demonstrate vocabulary and token distribution analysis.
    """
    print(f"\n📖 Vocabulary and Token Distribution Analysis")
    print("=" * 55)

    tokenizer = FieldAwarePCAPByteTokenizer()

    # Analyze vocabulary
    vocab = tokenizer.get_vocab()
    print(f"📚 Vocabulary Information:")
    print(f"   Total entries: {len(vocab)}")
    print(f"   Expected size: {tokenizer.vocab_size}")
    print(f"   ✅ Complete: {len(vocab) == tokenizer.vocab_size}")

    # Show special token mappings
    print(f"\n🎯 Special Token Mappings:")
    special_tokens = {
        '<pad>': tokenizer.pad_token_id,
        '<unk>': tokenizer.unk_token_id,
        '<bos>': tokenizer.bos_token_id,
        '<eos>': tokenizer.eos_token_id,
        '<pkt>': tokenizer.pkt_token_id,
        '<sep>': tokenizer.sep_token_id,
    }

    for token, token_id in special_tokens.items():
        print(f"   {token:>6} → ID {token_id}")

    # Analyze token usage in real data
    try:
        tokens = tokenizer.tokenize_pcap(
            pcap_path,
            add_packet_separators=True,
            add_field_separators=True
        )
        token_ids = tokenizer.tokenize_pcap_to_ids(
            pcap_path,
            add_packet_separators=True,
            add_field_separators=True
        )

        print(f"\n📊 Token Usage Analysis:")
        print(f"   Total tokens generated: {len(token_ids):,}")

        # Separate special and byte tokens
        special_ids = [tid for tid in token_ids if tid < 6]
        byte_ids = [tid for tid in token_ids if tid >= 6]

        print(f"   Special tokens: {len(special_ids):,}")
        print(f"   Byte tokens: {len(byte_ids):,}")

        # Special token breakdown
        special_breakdown = Counter(special_ids)
        for token_id, count in special_breakdown.items():
            token_name = tokenizer._convert_id_to_token(token_id)
            print(f"     {token_name}: {count:,}")

        # Byte token analysis
        if byte_ids:
            original_bytes = [tid - 6 for tid in byte_ids]
            unique_bytes = set(original_bytes)

            print(f"\n🔢 Byte Token Analysis:")
            print(f"   Unique byte values: {len(unique_bytes)}/256 ({len(unique_bytes) / 256 * 100:.1f}%)")
            print(f"   Token ID range: {min(byte_ids)} - {max(byte_ids)}")
            print(f"   Byte value range: {min(original_bytes)} - {max(original_bytes)}")

            # Most/least common bytes
            byte_counter = Counter(original_bytes)
            most_common = byte_counter.most_common(5)

            print(f"\n   Most common bytes:")
            for byte_val, count in most_common:
                percentage = count / len(original_bytes) * 100
                char_display = chr(byte_val) if 32 <= byte_val <= 126 else f"\\x{byte_val:02x}"
                print(f"     Byte {byte_val:3d} ('{char_display}'): {count:,} ({percentage:.1f}%)")

    except Exception as e:
        print(f"❌ Token analysis failed: {e}")


def main():
    """Main function to run the enhanced PCAP tokenization example."""

    # Path to the test PCAP file
    pcap_path = "../data/test.pcap"

    print("🚀 Enhanced PCAP Byte Tokenizer with Field Separation")
    print("🎯 Features: Special Tokens + Packet Separation + Field Separation + Malformed Handling")
    print("=" * 85)

    # Run different analyses
    analyze_enhanced_pcap_tokenization(pcap_path)
    demonstrate_field_separation_analysis(pcap_path)
    demonstrate_round_trip_with_separators(pcap_path)
    demonstrate_vocabulary_analysis(pcap_path)

    # Clean up log files
    log_files = [
        "malformed_conservative.log",
        "malformed_hybrid.log",
        "malformed_aggressive.log",
        "field_analysis_errors.log"
    ]

    cleaned_logs = []
    for log_file in log_files:
        if os.path.exists(log_file):
            os.remove(log_file)
            cleaned_logs.append(log_file)

    if cleaned_logs:
        print(f"\n🧹 Cleaned up log files: {', '.join(cleaned_logs)}")

    print(f"\n🎉 Enhanced PCAP Analysis with Field Separation Complete!")
    print("=" * 85)


if __name__ == "__main__":
    main()