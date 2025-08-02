import os
import shutil
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

import dpkt
from transformers.utils import cached_property, is_tf_available, is_torch_available

from src.byte.raw.token_pcap_byte_tokenizer import TokenPCAPByteTokenizer

if is_torch_available():
    FRAMEWORK = "pt"
elif is_tf_available():
    FRAMEWORK = "tf"
else:
    FRAMEWORK = "jax"


class TokenPCAPByteTokenizationTest(unittest.TestCase):
    tokenizer_class = TokenPCAPByteTokenizer
    test_rust_tokenizer = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp_dir_name = tempfile.mkdtemp()
        tokenizer = TokenPCAPByteTokenizer()
        tokenizer.save_pretrained(cls.tmp_dir_name)

        # Create test PCAP files
        cls.test_pcap_dir = tempfile.mkdtemp()
        cls._create_test_pcap_files()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        # Clean up temporary directories
        if hasattr(cls, 'tmp_dir_name'):
            shutil.rmtree(cls.tmp_dir_name)
        if hasattr(cls, 'test_pcap_dir'):
            shutil.rmtree(cls.test_pcap_dir)

    @classmethod
    def _create_test_pcap_files(cls):
        """Create test PCAP files with known content for testing."""

        # Create a simple PCAP file with known packet data
        cls.simple_pcap_path = os.path.join(cls.test_pcap_dir, "simple.pcap")
        cls._create_simple_pcap(cls.simple_pcap_path)

        # Create an empty PCAP file
        cls.empty_pcap_path = os.path.join(cls.test_pcap_dir, "empty.pcap")
        cls._create_empty_pcap(cls.empty_pcap_path)

        # Create a multi-packet PCAP file
        cls.multi_pcap_path = os.path.join(cls.test_pcap_dir, "multi.pcap")
        cls._create_multi_packet_pcap(cls.multi_pcap_path)

        # Create a single byte packet PCAP for testing edge cases
        cls.single_byte_pcap_path = os.path.join(cls.test_pcap_dir, "single_byte.pcap")
        cls._create_single_byte_pcap(cls.single_byte_pcap_path)

    @classmethod
    def _create_simple_pcap(cls, filepath):
        """Create a simple PCAP file with one known packet."""
        with open(filepath, 'wb') as f:
            writer = dpkt.pcap.Writer(f)

            # Create a simple Ethernet frame with known bytes
            eth_packet = (
                b'\x00\x01\x02\x03\x04\x05'  # dst MAC
                b'\x06\x07\x08\x09\x0a\x0b'  # src MAC  
                b'\x08\x00'  # ethertype (IP)
                b'Hello World'  # payload
            )

            writer.writepkt(eth_packet, ts=1234567890.0)

    @classmethod
    def _create_empty_pcap(cls, filepath):
        """Create an empty PCAP file (header only)."""
        with open(filepath, 'wb') as f:
            dpkt.pcap.Writer(f)
            # Writer creates header automatically, we just don't add packets

    @classmethod
    def _create_multi_packet_pcap(cls, filepath):
        """Create a PCAP file with multiple packets."""
        with open(filepath, 'wb') as f:
            writer = dpkt.pcap.Writer(f)

            # Packet 1: Simple text
            packet1 = b'Hello'
            writer.writepkt(packet1, ts=1234567890.0)

            # Packet 2: Mix of bytes
            packet2 = b'\x0a\x0b\x0c\x0d\x0e\x0f'
            writer.writepkt(packet2, ts=1234567891.0)

            # Packet 3: All byte values 0-255
            packet3 = bytes(range(256))
            writer.writepkt(packet3, ts=1234567892.0)

    @classmethod
    def _create_single_byte_pcap(cls, filepath):
        """Create a PCAP file with single-byte packets."""
        with open(filepath, 'wb') as f:
            writer = dpkt.pcap.Writer(f)

            # Three single-byte packets
            writer.writepkt(b'\x00', ts=1000000000.0)
            writer.writepkt(b'\x7F', ts=1000000001.0)
            writer.writepkt(b'\xFF', ts=1000000002.0)

    @cached_property
    def pcap_tokenizer(self):
        return TokenPCAPByteTokenizer.from_pretrained(self.tmp_dir_name)

    @classmethod
    @lru_cache(maxsize=64)
    def get_tokenizer(cls, pretrained_name=None, **kwargs) -> TokenPCAPByteTokenizer:
        pretrained_name = pretrained_name or cls.tmp_dir_name
        return cls.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    def get_clean_sequence(self, tokenizer, max_length=20, min_length=5) -> tuple[str, list]:
        """Override to work with PCAP file paths instead of text."""
        # Use our simple test PCAP file as input
        pcap_path = self.simple_pcap_path

        # Tokenize the PCAP file without special tokens for this method
        tokens = tokenizer._tokenize(pcap_path, add_packet_separators=False,
                                     add_bos=False, add_eos=False)
        token_ids = [tokenizer._convert_token_to_id(token) for token in tokens]

        # Limit length as requested
        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            tokens = tokens[:max_length]

        if min_length is not None and min_length > len(token_ids) > 0:
            # Repeat tokens to reach minimum length
            while len(token_ids) < min_length:
                token_ids = token_ids + token_ids
                tokens = tokens + tokens
            token_ids = token_ids[:min_length]  # Trim to exact length
            tokens = tokens[:min_length]

        # Convert back to string for compatibility
        output_txt = tokenizer.convert_tokens_to_string(tokens)

        return output_txt, token_ids

    def test_special_token_properties(self):
        """Test that special tokens have correct IDs and properties."""
        tokenizer = self.pcap_tokenizer

        # Check special token properties
        self.assertEqual(tokenizer.pad_token, "<pad>")
        self.assertEqual(tokenizer.unk_token, "<unk>")
        self.assertEqual(tokenizer.eos_token, "<eos>")
        self.assertEqual(tokenizer.bos_token, "<bos>")

        # Check special token IDs
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertEqual(tokenizer.unk_token_id, 1)
        self.assertEqual(tokenizer.eos_token_id, 2)
        self.assertEqual(tokenizer.bos_token_id, 3)
        self.assertEqual(tokenizer.pkt_token_id, 4)

        # Check vocabulary size (5 special tokens + 256 byte values)
        self.assertEqual(tokenizer.vocab_size, 261)

    def test_vocabulary_consistency(self):
        """Test that vocabulary is consistent and complete."""
        tokenizer = self.pcap_tokenizer

        # Vocabulary size should be 261 (5 special + 256 bytes)
        self.assertEqual(tokenizer.vocab_size, 261)

        vocab = tokenizer.get_vocab()
        self.assertEqual(len(vocab), 261)

        # Check special tokens are in vocab
        self.assertIn("<pad>", vocab)
        self.assertIn("<unk>", vocab)
        self.assertIn("<eos>", vocab)
        self.assertIn("<bos>", vocab)
        self.assertIn("<pkt>", vocab)

        # Check special token IDs
        self.assertEqual(vocab["<pad>"], 0)
        self.assertEqual(vocab["<unk>"], 1)
        self.assertEqual(vocab["<eos>"], 2)
        self.assertEqual(vocab["<bos>"], 3)
        self.assertEqual(vocab["<pkt>"], 4)

        # Check byte tokens (with offset)
        for i in range(256):
            char = chr(i)
            self.assertIn(char, vocab)
            self.assertEqual(vocab[char], i + 5)  # Offset of 5

    def test_token_id_conversion(self):
        """Test token to ID conversion for special tokens and bytes."""
        tokenizer = self.pcap_tokenizer

        # Test special tokens
        self.assertEqual(tokenizer._convert_token_to_id("<pad>"), 0)
        self.assertEqual(tokenizer._convert_token_to_id("<unk>"), 1)
        self.assertEqual(tokenizer._convert_token_to_id("<eos>"), 2)
        self.assertEqual(tokenizer._convert_token_to_id("<bos>"), 3)
        self.assertEqual(tokenizer._convert_token_to_id("<pkt>"), 4)

        # Test byte tokens (should have offset)
        self.assertEqual(tokenizer._convert_token_to_id("A"), 65 + 5)  # 'A' = byte 65, token ID 70
        self.assertEqual(tokenizer._convert_token_to_id("\x00"), 0 + 5)  # null byte, token ID 5
        self.assertEqual(tokenizer._convert_token_to_id("\xFF"), 255 + 5)  # max byte, token ID 260

        # Test round trip conversion
        for i in range(261):
            token = tokenizer._convert_id_to_token(i)
            converted_back = tokenizer._convert_token_to_id(token)
            self.assertEqual(i, converted_back)

    def test_tokenize_simple_pcap_basic_mode(self):
        """Test basic tokenization without special tokens."""
        tokenizer = self.pcap_tokenizer

        # Basic mode (no special tokens)
        tokens = tokenizer._tokenize(self.simple_pcap_path,
                                     add_packet_separators=False,
                                     add_bos=False,
                                     add_eos=False)

        # Should have tokens for each byte in the packet
        self.assertGreater(len(tokens), 0)

        # Each token should be a single character (no special tokens)
        for token in tokens:
            self.assertEqual(len(token), 1)
            self.assertIsInstance(token, str)
            # Should not contain special tokens
            self.assertNotIn(token, ["<pad>", "<unk>", "<eos>", "<bos>", "<pkt>"])

    def test_tokenize_with_packet_separators(self):
        """Test tokenization with packet separator tokens."""
        tokenizer = self.pcap_tokenizer

        # Use multi-packet PCAP file
        tokens = tokenizer._tokenize(self.multi_pcap_path,
                                     add_packet_separators=True,
                                     add_bos=False,
                                     add_eos=False)

        # Should contain <pkt> tokens
        pkt_count = tokens.count("<pkt>")

        # Read packets to check expected separator count
        packets = tokenizer.read_pcap_packets(self.multi_pcap_path)
        expected_separators = len(packets) - 1  # N packets need N-1 separators

        self.assertEqual(pkt_count, expected_separators)

    def test_tokenize_with_bos_eos(self):
        """Test tokenization with BOS and EOS tokens."""
        tokenizer = self.pcap_tokenizer

        tokens = tokenizer._tokenize(self.simple_pcap_path,
                                     add_packet_separators=False,
                                     add_bos=True,
                                     add_eos=True)

        # Should start with BOS and end with EOS
        self.assertEqual(tokens[0], "<bos>")
        self.assertEqual(tokens[-1], "<eos>")

    def test_tokenize_full_mode(self):
        """Test tokenization with all special tokens enabled."""
        tokenizer = self.pcap_tokenizer

        tokens = tokenizer._tokenize(self.multi_pcap_path,
                                     add_packet_separators=True,
                                     add_bos=True,
                                     add_eos=True)

        # Should have BOS at start
        self.assertEqual(tokens[0], "<bos>")

        # Should have EOS at end
        self.assertEqual(tokens[-1], "<eos>")

        # Should have packet separators
        pkt_count = tokens.count("<pkt>")
        packets = tokenizer.read_pcap_packets(self.multi_pcap_path)
        expected_separators = len(packets) - 1
        self.assertEqual(pkt_count, expected_separators)

    def test_tokenize_empty_pcap(self):
        """Test tokenization of an empty PCAP file."""
        tokenizer = self.pcap_tokenizer

        # Empty PCAP without special tokens
        tokens = tokenizer._tokenize(self.empty_pcap_path,
                                     add_packet_separators=False,
                                     add_bos=False,
                                     add_eos=False)
        self.assertEqual(len(tokens), 0)

        # Empty PCAP with BOS/EOS
        tokens_with_special = tokenizer._tokenize(self.empty_pcap_path,
                                                  add_packet_separators=False,
                                                  add_bos=True,
                                                  add_eos=True)
        self.assertEqual(len(tokens_with_special), 2)  # Just BOS and EOS
        self.assertEqual(tokens_with_special, ["<bos>", "<eos>"])

    def test_read_pcap_packets(self):
        """Test reading individual packets from PCAP file."""
        tokenizer = self.pcap_tokenizer

        packets = tokenizer.read_pcap_packets(self.multi_pcap_path)

        # Should have 3 packets
        self.assertEqual(len(packets), 3)

        # Check packet contents
        self.assertEqual(packets[0], b'Hello')
        self.assertEqual(packets[1], b'\x0a\x0b\x0c\x0d\x0e\x0f')
        self.assertEqual(packets[2], bytes(range(256)))

    def test_public_tokenization_methods(self):
        """Test the public tokenization methods."""
        tokenizer = self.pcap_tokenizer

        # Test tokenize_pcap
        tokens = tokenizer.tokenize_pcap(self.simple_pcap_path,
                                         add_packet_separators=True,
                                         add_bos=True,
                                         add_eos=True)
        self.assertGreater(len(tokens), 2)  # At least BOS + data + EOS
        self.assertEqual(tokens[0], "<bos>")
        self.assertEqual(tokens[-1], "<eos>")

        # Test tokenize_pcap_to_ids
        token_ids = tokenizer.tokenize_pcap_to_ids(self.simple_pcap_path,
                                                   add_packet_separators=True,
                                                   add_bos=True,
                                                   add_eos=True)
        self.assertEqual(len(token_ids), len(tokens))
        self.assertEqual(token_ids[0], 3)  # BOS token ID
        self.assertEqual(token_ids[-1], 2)  # EOS token ID

    def test_round_trip_encoding_decoding(self):
        """Test that encoding and decoding are inverse operations."""
        tokenizer = self.pcap_tokenizer

        # Test basic mode (no special tokens)
        tokens = tokenizer._tokenize(self.multi_pcap_path,
                                     add_packet_separators=False,
                                     add_bos=False,
                                     add_eos=False)
        token_ids = [tokenizer._convert_token_to_id(token) for token in tokens]

        # Decode back to bytes (skip special tokens should not affect basic mode)
        decoded_bytes = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Re-read the original PCAP bytes (concatenated)
        packets = tokenizer.read_pcap_packets(self.multi_pcap_path)
        original_bytes = b"".join(packets)

        # Should be identical
        self.assertEqual(decoded_bytes, original_bytes)

    def test_round_trip_with_special_tokens(self):
        """Test round-trip with special tokens included."""
        tokenizer = self.pcap_tokenizer

        # Tokenize with special tokens
        token_ids = tokenizer.tokenize_pcap_to_ids(self.simple_pcap_path,
                                                   add_packet_separators=True,
                                                   add_bos=True,
                                                   add_eos=True)

        # Decode skipping special tokens should give original bytes
        decoded_bytes = tokenizer.decode(token_ids, skip_special_tokens=True)
        packets = tokenizer.read_pcap_packets(self.simple_pcap_path)
        original_bytes = b"".join(packets)
        self.assertEqual(decoded_bytes, original_bytes)

        # Decode including special tokens should give string representation
        decoded_string = tokenizer.decode(token_ids, skip_special_tokens=False)
        self.assertIsInstance(decoded_string, str)
        self.assertIn("<bos>", decoded_string)
        self.assertIn("<eos>", decoded_string)

    def test_convert_tokens_to_string(self):
        """Test token to string conversion (filters special tokens)."""
        tokenizer = self.pcap_tokenizer

        # Test with mixed tokens including special tokens
        test_tokens = ["<bos>", "H", "e", "l", "l", "o", "<pkt>", "W", "o", "r", "l", "d", "<eos>"]

        result_string = tokenizer.convert_tokens_to_string(test_tokens)

        # Should filter out special tokens
        expected_string = "HelloWorld"
        self.assertEqual(result_string, expected_string)

    def test_build_inputs_with_special_tokens(self):
        """Test building inputs with special tokens."""
        tokenizer = self.pcap_tokenizer

        token_ids_0 = [10, 20, 30]  # Some byte token IDs
        token_ids_1 = [40, 50, 60]  # More byte token IDs

        # Single sequence should get BOS + sequence + EOS
        result_single = tokenizer.build_inputs_with_special_tokens(token_ids_0)
        expected_single = [3] + token_ids_0 + [2]  # BOS + sequence + EOS
        self.assertEqual(result_single, expected_single)

        # Pair should get BOS + seq1 + PKT + seq2 + EOS
        result_pair = tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
        expected_pair = [3] + token_ids_0 + [4] + token_ids_1 + [2]  # BOS + seq1 + PKT + seq2 + EOS
        self.assertEqual(result_pair, expected_pair)

    def test_get_special_tokens_mask(self):
        """Test special tokens mask generation."""
        tokenizer = self.pcap_tokenizer

        token_ids_0 = [10, 20, 30]
        token_ids_1 = [40, 50, 60]

        # Test mask for single sequence (not already with special tokens)
        mask_single = tokenizer.get_special_tokens_mask(token_ids_0)
        expected_single = [1] + [0] * 3 + [1]  # BOS + regular + EOS
        self.assertEqual(mask_single, expected_single)

        # Test mask for pair
        mask_pair = tokenizer.get_special_tokens_mask(token_ids_0, token_ids_1)
        expected_pair = [1] + [0] * 3 + [1] + [0] * 3 + [1]  # BOS + seq1 + PKT + seq2 + EOS
        self.assertEqual(mask_pair, expected_pair)

        # Test mask for already special tokens
        mixed_ids = [3, 10, 20, 4, 30, 40, 2]  # BOS + bytes + PKT + bytes + EOS
        mask_already = tokenizer.get_special_tokens_mask(mixed_ids, already_has_special_tokens=True)
        expected_already = [1, 0, 0, 1, 0, 0, 1]  # Special=1, Regular=0
        self.assertEqual(mask_already, expected_already)

    def test_unknown_token_handling(self):
        """Test handling of unknown tokens."""
        tokenizer = self.pcap_tokenizer

        # Unknown token should return UNK ID
        unknown_id = tokenizer._convert_token_to_id("unknown_token")
        self.assertEqual(unknown_id, tokenizer.unk_token_id)

        # Multi-character token should return UNK ID
        multi_char_id = tokenizer._convert_token_to_id("abc")
        self.assertEqual(multi_char_id, tokenizer.unk_token_id)

    def test_file_not_found_error(self):
        """Test handling of non-existent PCAP files."""
        tokenizer = self.pcap_tokenizer

        non_existent_file = "/path/that/does/not/exist.pcap"

        with self.assertRaises(FileNotFoundError):
            tokenizer._tokenize(non_existent_file)

        with self.assertRaises(FileNotFoundError):
            tokenizer.read_pcap_packets(non_existent_file)

    def test_save_and_load_tokenizer(self):
        """Test saving and loading the tokenizer."""
        tokenizer = self.get_tokenizer()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save tokenizer
            tokenizer.save_pretrained(tmp_dir)

            # Load tokenizer
            loaded_tokenizer = TokenPCAPByteTokenizer.from_pretrained(tmp_dir)

            # Should have same properties
            self.assertEqual(tokenizer.vocab_size, loaded_tokenizer.vocab_size)
            self.assertEqual(tokenizer.pad_token_id, loaded_tokenizer.pad_token_id)
            self.assertEqual(tokenizer.unk_token_id, loaded_tokenizer.unk_token_id)
            self.assertEqual(tokenizer.eos_token_id, loaded_tokenizer.eos_token_id)
            self.assertEqual(tokenizer.bos_token_id, loaded_tokenizer.bos_token_id)
            self.assertEqual(tokenizer.pkt_token_id, loaded_tokenizer.pkt_token_id)

            # Should tokenize the same way
            tokens_original = tokenizer.tokenize_pcap(self.simple_pcap_path)
            tokens_loaded = loaded_tokenizer.tokenize_pcap(self.simple_pcap_path)
            self.assertEqual(tokens_original, tokens_loaded)

    def test_save_vocabulary(self):
        """Test vocabulary saving."""
        tokenizer = self.pcap_tokenizer

        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_files = tokenizer.save_vocabulary(tmp_dir)
            self.assertEqual(len(vocab_files), 1)
            self.assertTrue(os.path.exists(vocab_files[0]))

            # Check that vocab file contains expected content
            import json
            with open(vocab_files[0], 'r') as f:
                saved_vocab = json.load(f)

            expected_vocab = tokenizer.get_vocab()
            self.assertEqual(saved_vocab, expected_vocab)

    def test_single_byte_packets(self):
        """Test handling of single-byte packets."""
        tokenizer = self.pcap_tokenizer

        tokens = tokenizer.tokenize_pcap(self.single_byte_pcap_path,
                                         add_packet_separators=True,
                                         add_bos=True,
                                         add_eos=True)

        # Should have: BOS + byte + PKT + byte + PKT + byte + EOS = 7 tokens
        self.assertEqual(len(tokens), 7)
        self.assertEqual(tokens[0], "<bos>")
        self.assertEqual(tokens[-1], "<eos>")
        self.assertEqual(tokens.count("<pkt>"), 2)  # 3 packets = 2 separators

    def test_decode_single_bytes(self):
        """Test decoding individual byte values with offset."""
        tokenizer = self.pcap_tokenizer

        # Test decoding byte tokens (with offset)
        for i in range(256):
            token_id = i + 5  # Byte values start at ID 5
            decoded = tokenizer.decode([token_id], skip_special_tokens=True)
            self.assertEqual(len(decoded), 1)
            self.assertEqual(decoded[0], i)

        # Test decoding special tokens should be skipped
        special_ids = [0, 1, 2, 3, 4]  # All special token IDs
        decoded_special = tokenizer.decode(special_ids, skip_special_tokens=True)
        self.assertEqual(len(decoded_special), 0)  # Should be empty

    def test_decode_single_int(self):
        """Test decoding a single integer."""
        tokenizer = self.pcap_tokenizer

        # Should handle single int input (byte token)
        decoded = tokenizer.decode(65 + 5, skip_special_tokens=True)  # 'A' with offset
        self.assertEqual(decoded, b'A')

        # Special token should be skipped
        decoded_special = tokenizer.decode(0, skip_special_tokens=True)  # PAD token
        self.assertEqual(decoded_special, b'')

    def test_path_handling(self):
        """Test handling of different path types."""
        tokenizer = self.pcap_tokenizer

        # Test with string path
        tokens_str = tokenizer._tokenize(str(self.simple_pcap_path))

        # Test with Path object
        tokens_path = tokenizer._tokenize(Path(self.simple_pcap_path))

        # Should produce same result
        self.assertEqual(tokens_str, tokens_path)

    def test_definitive_tokenization_verification(self):
        """
        Definitive test: Craft a PCAP with known bytes and verify exact token IDs.
        This accounts for the enhanced tokenizer's byte offset.
        """
        tokenizer = self.pcap_tokenizer

        # Create a temporary PCAP file with precisely known content
        test_pcap_path = os.path.join(self.test_pcap_dir, "definitive_enhanced_test.pcap")

        with open(test_pcap_path, 'wb') as f:
            writer = dpkt.pcap.Writer(f)

            # Packet 1: Simple ASCII sequence "Hi"
            packet1 = b'Hi'
            writer.writepkt(packet1, ts=1000000000.0)

            # Packet 2: Specific byte sequence
            packet2 = bytes([0x00, 0x01, 0x7F, 0xFF])
            writer.writepkt(packet2, ts=1000000001.0)

        # Test basic mode (no special tokens)
        tokens = tokenizer._tokenize(test_pcap_path,
                                     add_packet_separators=False,
                                     add_bos=False,
                                     add_eos=False)
        token_ids = [tokenizer._convert_token_to_id(token) for token in tokens]

        # Expected token IDs with offset
        # "Hi" = [72, 105] -> [77, 110] (with +5 offset)
        # [0, 1, 127, 255] -> [5, 6, 132, 260] (with +5 offset)
        expected_basic_ids = [72 + 5, 105 + 5, 0 + 5, 1 + 5, 127 + 5, 255 + 5]
        self.assertEqual(token_ids, expected_basic_ids)

        # Test with packet separators
        tokens_with_sep = tokenizer._tokenize(test_pcap_path,
                                              add_packet_separators=True,
                                              add_bos=False,
                                              add_eos=False)
        token_ids_with_sep = [tokenizer._convert_token_to_id(token) for token in tokens_with_sep]

        # Expected: packet1 + PKT + packet2
        expected_with_sep = [72 + 5, 105 + 5, 4, 0 + 5, 1 + 5, 127 + 5, 255 + 5]  # PKT token = ID 4
        self.assertEqual(token_ids_with_sep, expected_with_sep)

        # Test full mode (BOS + separators + EOS)
        tokens_full = tokenizer._tokenize(test_pcap_path,
                                          add_packet_separators=True,
                                          add_bos=True,
                                          add_eos=True)
        token_ids_full = [tokenizer._convert_token_to_id(token) for token in tokens_full]

        # Expected: BOS + packet1 + PKT + packet2 + EOS
        expected_full = [3, 72 + 5, 105 + 5, 4, 0 + 5, 1 + 5, 127 + 5, 255 + 5, 2]  # BOS=3, PKT=4, EOS=2
        self.assertEqual(token_ids_full, expected_full)

        print(f"\nDEFINITIVE ENHANCED TOKENIZATION TEST:")
        print(f"Basic mode: {token_ids}")
        print(f"With separators: {token_ids_with_sep}")
        print(f"Full mode: {token_ids_full}")
        print("✓ All enhanced tokenization verification checks passed!")

        # Test round-trip decoding
        decoded_bytes = tokenizer.decode(token_ids_full, skip_special_tokens=True)
        packets = tokenizer.read_pcap_packets(test_pcap_path)
        original_bytes = b"".join(packets)
        self.assertEqual(decoded_bytes, original_bytes)

        print("✓ Round-trip encoding/decoding verified!")

    def test_edge_cases(self):
        """Test various edge cases."""
        tokenizer = self.pcap_tokenizer

        # Test ID bounds
        with self.assertRaises(ValueError):
            tokenizer._convert_id_to_token(-1)

        with self.assertRaises(ValueError):
            tokenizer._convert_id_to_token(261)  # vocab_size

        # Test valid boundary IDs
        self.assertEqual(tokenizer._convert_id_to_token(0), "<pad>")
        self.assertEqual(tokenizer._convert_id_to_token(4), "<pkt>")
        self.assertEqual(tokenizer._convert_id_to_token(5), chr(0))  # First byte
        self.assertEqual(tokenizer._convert_id_to_token(260), chr(255))  # Last byte

    # Skip tests that don't apply to our enhanced tokenizer
    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths, not pretokenized inputs")
    def test_pretokenized_inputs(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer doesn't support text input for __call__")
    def test_call_with_text_input(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths as input")
    def test_batch_encoding(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths, not text")
    def test_call(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths, not text")
    def test_batch_encode_plus_batch_sequence_length(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths, not text")
    def test_batch_encode_plus_padding(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths, not text")
    def test_encode_plus_with_padding(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths, not text")
    def test_internal_consistency(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths, not text")
    def test_maximum_encoding_length_single_input(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths, not text")
    def test_maximum_encoding_length_pair_input(self):
        pass

    @unittest.skip(reason="Covered by test_get_special_tokens_mask")
    def test_special_tokens_mask(self):
        pass

    @unittest.skip(reason="Covered by test_get_special_tokens_mask")
    def test_special_tokens_mask_input_pairs(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer doesn't work with text padding")
    def test_right_and_left_padding(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer doesn't work with text truncation")
    def test_right_and_left_truncation(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer doesn't use attention masks in the same way")
    def test_padding_with_attention_mask(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths, not text")
    def test_rust_and_python_full_tokenizers(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths, not text")
    def test_encode_decode_with_spaces(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths, not text")
    def test_prepare_for_model(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer uses file paths, not text")
    def test_prepare_seq2seq_batch(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer doesn't support text")
    def test_added_tokens_do_lower_case(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer doesn't support text")
    def test_add_tokens_tokenizer(self):
        pass

    @unittest.skip(reason="EnhancedPCAPByteTokenizer special tokens are predefined")
    def test_add_special_tokens(self):
        pass


if __name__ == '__main__':
    unittest.main()