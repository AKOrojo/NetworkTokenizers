import os
import shutil
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

import dpkt
from transformers.utils import cached_property, is_tf_available, is_torch_available

from src.byte.raw.pcap_byte_tokenizer import PCAPByteTokenizer

if is_torch_available():
    FRAMEWORK = "pt"
elif is_tf_available():
    FRAMEWORK = "tf"
else:
    FRAMEWORK = "jax"


class PCAPByteTokenizationTest(unittest.TestCase):
    tokenizer_class = PCAPByteTokenizer
    test_rust_tokenizer = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp_dir_name = tempfile.mkdtemp()
        tokenizer = PCAPByteTokenizer()
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

            # Packet 1
            packet1 = b'\x01\x02\x03\x04\x05'
            writer.writepkt(packet1, ts=1234567890.0)

            # Packet 2
            packet2 = b'\x0a\x0b\x0c\x0d\x0e\x0f'
            writer.writepkt(packet2, ts=1234567891.0)

            # Packet 3 with all byte values 0-255
            packet3 = bytes(range(256))
            writer.writepkt(packet3, ts=1234567892.0)

    @cached_property
    def pcap_tokenizer(self):
        return PCAPByteTokenizer.from_pretrained(self.tmp_dir_name)

    @classmethod
    @lru_cache(maxsize=64)
    def get_tokenizer(cls, pretrained_name=None, **kwargs) -> PCAPByteTokenizer:
        pretrained_name = pretrained_name or cls.tmp_dir_name
        return cls.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    def get_clean_sequence(self, tokenizer, max_length=20, min_length=5) -> tuple[str, list]:
        """Override to work with PCAP file paths instead of text."""
        # Use our simple test PCAP file as input
        pcap_path = self.simple_pcap_path

        # Tokenize the PCAP file
        tokens = tokenizer._tokenize(pcap_path)
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

    def test_tokenize_simple_pcap(self):
        """Test basic tokenization of a simple PCAP file."""
        tokenizer = self.pcap_tokenizer

        # Tokenize the simple PCAP file
        tokens = tokenizer._tokenize(self.simple_pcap_path)

        # Should have tokens for each byte in the packet
        self.assertGreater(len(tokens), 0)

        # Each token should be a single character
        for token in tokens:
            self.assertEqual(len(token), 1)
            self.assertIsInstance(token, str)

    def test_tokenize_empty_pcap(self):
        """Test tokenization of an empty PCAP file."""
        tokenizer = self.pcap_tokenizer

        # Empty PCAP should produce empty token list
        tokens = tokenizer._tokenize(self.empty_pcap_path)
        self.assertEqual(len(tokens), 0)

    def test_tokenize_multi_packet_pcap(self):
        """Test tokenization of multi-packet PCAP file."""
        tokenizer = self.pcap_tokenizer

        tokens = tokenizer._tokenize(self.multi_pcap_path)
        token_ids = [tokenizer._convert_token_to_id(token) for token in tokens]

        # Should have tokens from all packets concatenated
        self.assertGreater(len(tokens), 256)  # At least from the full byte range packet

        # Should include all possible byte values (0-255) from the third packet
        unique_ids = set(token_ids)
        self.assertEqual(len(unique_ids), 256)  # All possible byte values should be present

    def test_vocabulary_consistency(self):
        """Test that vocabulary is consistent and complete."""
        tokenizer = self.pcap_tokenizer

        # Vocabulary size should be exactly 256
        self.assertEqual(tokenizer.vocab_size, 256)

        # Should be able to convert all IDs 0-255 to tokens
        for i in range(256):
            token = tokenizer._convert_id_to_token(i)
            self.assertEqual(len(token), 1)

            # Round trip should work
            converted_back = tokenizer._convert_token_to_id(token)
            self.assertEqual(i, converted_back)

    def test_round_trip_encoding_decoding(self):
        """Test that encoding and decoding are inverse operations."""
        tokenizer = self.pcap_tokenizer

        # Use multi-packet PCAP to test all byte values
        tokens = tokenizer._tokenize(self.multi_pcap_path)
        token_ids = [tokenizer._convert_token_to_id(token) for token in tokens]

        # Decode back to bytes
        decoded_bytes = tokenizer.decode(token_ids)

        # Re-read the original PCAP bytes
        original_bytes = tokenizer._read_pcap_bytes(self.multi_pcap_path)

        # Should be identical
        self.assertEqual(decoded_bytes, original_bytes)

    def test_convert_tokens_to_string(self):
        """Test token to string conversion."""
        tokenizer = self.pcap_tokenizer

        # Test with known byte sequence
        test_bytes = b'Hello\x00\xff\x80World'
        tokens = [chr(b) for b in test_bytes]

        result_string = tokenizer.convert_tokens_to_string(tokens)

        # Should reconstruct the original string
        expected_string = ''.join(chr(b) for b in test_bytes)
        self.assertEqual(result_string, expected_string)

    def test_special_tokens_are_none(self):
        """Test that all special tokens are None as expected."""
        tokenizer = self.pcap_tokenizer

        self.assertIsNone(tokenizer.pad_token)
        self.assertIsNone(tokenizer.eos_token)
        self.assertIsNone(tokenizer.unk_token)
        self.assertIsNone(tokenizer.bos_token)
        self.assertIsNone(tokenizer.sep_token)
        self.assertIsNone(tokenizer.cls_token)
        self.assertIsNone(tokenizer.mask_token)

    def test_build_inputs_with_special_tokens(self):
        """Test building inputs without special tokens."""
        tokenizer = self.pcap_tokenizer

        token_ids_0 = [1, 2, 3, 4, 5]
        token_ids_1 = [6, 7, 8, 9, 10]

        # Should just concatenate without adding special tokens
        result_single = tokenizer.build_inputs_with_special_tokens(token_ids_0)
        self.assertEqual(result_single, token_ids_0)

        result_pair = tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
        self.assertEqual(result_pair, token_ids_0 + token_ids_1)

    def test_get_special_tokens_mask(self):
        """Test special tokens mask (should be all zeros)."""
        tokenizer = self.pcap_tokenizer

        token_ids_0 = [1, 2, 3, 4, 5]
        token_ids_1 = [6, 7, 8, 9, 10]

        # Should return all zeros since no special tokens
        mask_single = tokenizer.get_special_tokens_mask(token_ids_0)
        self.assertEqual(mask_single, [0] * len(token_ids_0))

        mask_pair = tokenizer.get_special_tokens_mask(token_ids_0, token_ids_1)
        self.assertEqual(mask_pair, [0] * (len(token_ids_0) + len(token_ids_1)))

    def test_file_not_found_error(self):
        """Test handling of non-existent PCAP files."""
        tokenizer = self.pcap_tokenizer

        non_existent_file = "/path/that/does/not/exist.pcap"

        with self.assertRaises(FileNotFoundError):
            tokenizer._tokenize(non_existent_file)

    def test_save_and_load_tokenizer(self):
        """Test saving and loading the tokenizer."""
        tokenizer = self.get_tokenizer()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save tokenizer
            tokenizer.save_pretrained(tmp_dir)

            # Load tokenizer
            loaded_tokenizer = PCAPByteTokenizer.from_pretrained(tmp_dir)

            # Should have same properties
            self.assertEqual(tokenizer.vocab_size, loaded_tokenizer.vocab_size)
            self.assertEqual(tokenizer.model_input_names, loaded_tokenizer.model_input_names)

            # Should tokenize the same way
            tokens_original = tokenizer._tokenize(self.simple_pcap_path)
            tokens_loaded = loaded_tokenizer._tokenize(self.simple_pcap_path)
            self.assertEqual(tokens_original, tokens_loaded)

    def test_get_vocab(self):
        """Test vocabulary dictionary generation."""
        tokenizer = self.pcap_tokenizer

        vocab = tokenizer.get_vocab()

        # Should have exactly 256 entries
        self.assertEqual(len(vocab), 256)

        # Each byte value should map to its character representation
        for i in range(256):
            char = chr(i)
            self.assertIn(char, vocab)
            self.assertEqual(vocab[char], i)

    def test_decode_single_bytes(self):
        """Test decoding individual byte values."""
        tokenizer = self.pcap_tokenizer

        # Test decoding each possible byte value
        for i in range(256):
            decoded = tokenizer.decode([i])
            self.assertEqual(len(decoded), 1)
            self.assertEqual(decoded[0], i)

        # Test decoding multiple bytes
        test_ids = [72, 101, 108, 108, 111]  # "Hello" in ASCII
        decoded = tokenizer.decode(test_ids)
        self.assertEqual(decoded, b'Hello')

    def test_decode_single_int(self):
        """Test decoding a single integer."""
        tokenizer = self.pcap_tokenizer

        # Should handle single int input
        decoded = tokenizer.decode(65)
        self.assertEqual(decoded, b'A')

    def test_path_handling(self):
        """Test handling of different path types."""
        tokenizer = self.pcap_tokenizer

        # Test with string path
        tokens_str = tokenizer._tokenize(str(self.simple_pcap_path))

        # Test with Path object
        tokens_path = tokenizer._tokenize(Path(self.simple_pcap_path))

        # Should produce same result
        self.assertEqual(tokens_str, tokens_path)

    def test_save_vocabulary(self):
        """Test vocabulary saving (should return empty tuple)."""
        tokenizer = self.pcap_tokenizer

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = tokenizer.save_vocabulary(tmp_dir)
            self.assertEqual(result, ())

    def test_definitive_tokenization_verification(self):
        """
        Definitive test: Craft a PCAP with known bytes and verify exact token IDs.
        This is the ultimate test to ensure tokenization is working correctly.
        """
        tokenizer = self.pcap_tokenizer

        # Create a temporary PCAP file with precisely known content
        test_pcap_path = os.path.join(self.test_pcap_dir, "definitive_test.pcap")

        with open(test_pcap_path, 'wb') as f:
            writer = dpkt.pcap.Writer(f)

            # Packet 1: Simple ASCII sequence "Hello"
            packet1 = b'Hello'
            writer.writepkt(packet1, ts=1000000000.0)

            # Packet 2: Specific byte sequence with known values
            packet2 = bytes([0x00, 0x01, 0x0A, 0x10, 0x7F, 0x80, 0xFF])
            writer.writepkt(packet2, ts=1000000001.0)

            # Packet 3: All byte values 0-255 in order
            packet3 = bytes(range(256))
            writer.writepkt(packet3, ts=1000000002.0)

            # Packet 4: Some common network bytes (Ethernet header-like)
            packet4 = bytes([
                0x00, 0x11, 0x22, 0x33, 0x44, 0x55,  # dst MAC
                0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,  # src MAC
                0x08, 0x00,  # ethertype (IP)
                0x45, 0x00  # IP version/IHL, DSCP/ECN
            ])
            writer.writepkt(packet4, ts=1000000003.0)

        # Calculate expected token IDs
        # Each byte value directly maps to its token ID
        expected_token_ids = []

        # Packet 1: "Hello" = [72, 101, 108, 108, 111]
        expected_token_ids.extend([72, 101, 108, 108, 111])

        # Packet 2: [0, 1, 10, 16, 127, 128, 255]
        expected_token_ids.extend([0, 1, 10, 16, 127, 128, 255])

        # Packet 3: [0, 1, 2, ..., 255]
        expected_token_ids.extend(list(range(256)))

        # Packet 4: The specific bytes we defined
        expected_token_ids.extend([
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55,  # dst MAC
            0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,  # src MAC
            0x08, 0x00,  # ethertype
            0x45, 0x00  # IP header start
        ])

        # Tokenize the PCAP file
        tokens = tokenizer._tokenize(test_pcap_path)
        actual_token_ids = [tokenizer._convert_token_to_id(token) for token in tokens]

        # Verify the results
        print(f"\nDEFINITIVE TOKENIZATION TEST:")
        print(f"Expected {len(expected_token_ids)} token IDs")
        print(f"Got {len(actual_token_ids)} token IDs")
        print(f"Expected: {expected_token_ids[:20]}... (showing first 20)")
        print(f"Actual:   {actual_token_ids[:20]}... (showing first 20)")

        # Test length matches
        self.assertEqual(len(actual_token_ids), len(expected_token_ids),
                         f"Token count mismatch: expected {len(expected_token_ids)}, got {len(actual_token_ids)}")

        # Test exact sequence matches
        self.assertEqual(actual_token_ids, expected_token_ids,
                         "Token ID sequence does not match expected values")

        # Additional verification: check that all 256 possible byte values are represented
        unique_token_ids = set(actual_token_ids)
        self.assertEqual(len(unique_token_ids), 256,
                         f"Expected all 256 byte values to be present, but got {len(unique_token_ids)} unique values")

        # Verify specific subsequences
        # Check "Hello" at the beginning
        hello_ids = actual_token_ids[:5]
        self.assertEqual(hello_ids, [72, 101, 108, 108, 111], "Hello sequence incorrect")

        # Check the specific byte sequence
        specific_start_idx = 5  # After "Hello"
        specific_ids = actual_token_ids[specific_start_idx:specific_start_idx + 7]
        self.assertEqual(specific_ids, [0, 1, 10, 16, 127, 128, 255], "Specific byte sequence incorrect")

        # Check that the full 0-255 sequence is present
        range_start_idx = 5 + 7  # After "Hello" + specific bytes
        range_ids = actual_token_ids[range_start_idx:range_start_idx + 256]
        self.assertEqual(range_ids, list(range(256)), "0-255 sequence incorrect")

        # Check the Ethernet-like header at the end
        eth_start_idx = 5 + 7 + 256  # After all previous packets
        eth_ids = actual_token_ids[eth_start_idx:eth_start_idx + 16]
        expected_eth_ids = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x08, 0x00, 0x45,
                            0x00]
        self.assertEqual(eth_ids, expected_eth_ids, "Ethernet header sequence incorrect")

        print("✓ All tokenization verification checks passed!")

        # Test round-trip decoding
        decoded_bytes = tokenizer.decode(actual_token_ids)
        original_bytes = tokenizer._read_pcap_bytes(test_pcap_path)
        self.assertEqual(decoded_bytes, original_bytes, "Round-trip decode failed")

        print("✓ Round-trip encoding/decoding verified!")

    # Skip tests that don't apply to our tokenizer
    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not pretokenized inputs")
    def test_pretokenized_inputs(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer doesn't support text input for __call__")
    def test_call_with_text_input(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths as input")
    def test_batch_encoding(self):
        pass

    # Skip more incompatible tests
    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_call(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_batch_encode_plus_batch_sequence_length(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_batch_encode_plus_padding(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_encode_plus_with_padding(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_internal_consistency(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_maximum_encoding_length_single_input(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_maximum_encoding_length_pair_input(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_special_tokens_mask(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_special_tokens_mask_input_pairs(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer doesn't work with text padding")
    def test_right_and_left_padding(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer doesn't work with text truncation")
    def test_right_and_left_truncation(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer doesn't use attention masks")
    def test_padding_with_attention_mask(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_rust_and_python_full_tokenizers(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_encode_decode_with_spaces(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_prepare_for_model(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer uses file paths, not text")
    def test_prepare_seq2seq_batch(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer doesn't support text")
    def test_added_tokens_do_lower_case(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer doesn't support text")
    def test_add_tokens_tokenizer(self):
        pass

    @unittest.skip(reason="PCAPByteTokenizer doesn't support text")
    def test_add_special_tokens(self):
        pass