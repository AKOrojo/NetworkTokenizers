# test_pcap_text_tokenizer.py
import shutil
import tempfile
from pathlib import Path

import pytest
from scapy.all import Raw
from scapy.all import wrpcap
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether

from others.pcap_text_tokenizer import PCAPTextTokenizer


class TestPCAPTextTokenizer:
    """Comprehensive test suite for PCAPTextTokenizer"""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance for testing"""
        return PCAPTextTokenizer()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_packets(self):
        """Create sample network packets for testing"""
        packets = []

        # HTTP packet
        http_packet = (Ether(dst="aa:bb:cc:dd:ee:ff", src="11:22:33:44:55:66") /
                       IP(src="192.168.1.100", dst="8.8.8.8") /
                       TCP(sport=12345, dport=80) /
                       Raw(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"))
        packets.append(http_packet)

        # DNS packet
        dns_packet = (Ether() /
                      IP(src="192.168.1.100", dst="8.8.8.8") /
                      UDP(sport=54321, dport=53) /
                      Raw(b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"))
        packets.append(dns_packet)

        # HTTPS packet
        https_packet = (Ether() /
                        IP(src="192.168.1.100", dst="1.1.1.1") /
                        TCP(sport=55555, dport=443) /
                        Raw(b"\x16\x03\x01\x00\x01\x01\x00"))
        packets.append(https_packet)

        return packets

    @pytest.fixture
    def sample_pcap_file(self, sample_packets, temp_dir):
        """Create a sample PCAP file for testing"""
        pcap_path = temp_dir / "test_sample.pcap"
        wrpcap(str(pcap_path), sample_packets)
        return pcap_path

    def test_tokenizer_initialization(self):
        """Test tokenizer initialization with default and custom parameters"""
        # Default initialization
        tokenizer = PCAPTextTokenizer()
        assert tokenizer.pcap_start_token == "<pcap>"
        assert tokenizer.pcap_end_token == "</pcap>"
        assert tokenizer.packet_sep_token == "<pkt>"
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.eos_token == "</s>"
        assert tokenizer.unk_token == "<unk>"

        # Custom initialization
        custom_tokenizer = PCAPTextTokenizer(
            pcap_start_token="[PCAP_START]",
            pcap_end_token="[PCAP_END]",
            packet_sep_token="[PKT_SEP]",
            extra_ids=50
        )
        assert custom_tokenizer.pcap_start_token == "[PCAP_START]"
        assert custom_tokenizer.pcap_end_token == "[PCAP_END]"
        assert custom_tokenizer.packet_sep_token == "[PKT_SEP]"

    def test_vocabulary_properties(self, tokenizer):
        """Test vocabulary size and structure"""
        # Test vocab size (should be 256 for UTF-8 bytes)
        assert tokenizer.vocab_size == 256

        # Test vocab mapping
        vocab = tokenizer.get_vocab()
        assert len(vocab) >= 256 + tokenizer.offset

        # Test special tokens are in vocab
        assert tokenizer.pcap_start_token in vocab
        assert tokenizer.pcap_end_token in vocab
        assert tokenizer.packet_sep_token in vocab

    def test_text_tokenization_basic(self, tokenizer):
        """Test basic text tokenization functionality"""
        # Simple ASCII text
        text = "Hello"
        tokens = tokenizer.tokenize(text)
        expected = ['H', 'e', 'l', 'l', 'o']
        assert tokens == expected

        # Unicode text
        unicode_text = "Hello 🌍"
        tokens = tokenizer.tokenize(unicode_text)
        # Should handle UTF-8 encoding properly
        assert len(tokens) > len(unicode_text)  # emoji takes multiple bytes
        assert all(len(token) == 1 for token in tokens)

    def test_text_tokenization_edge_cases(self, tokenizer):
        """Test text tokenization edge cases"""
        # Empty string
        assert tokenizer.tokenize("") == []

        # Whitespace
        tokens = tokenizer.tokenize(" \t\n")
        assert len(tokens) == 3

        # Special characters
        special_text = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        tokens = tokenizer.tokenize(special_text)
        assert len(tokens) == len(special_text)

    def test_token_id_conversion(self, tokenizer):
        """Test conversion between tokens and IDs"""
        text = "Test123"

        # Tokenize and convert to IDs
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Convert back to tokens
        recovered_tokens = tokenizer.convert_ids_to_tokens(token_ids)

        assert tokens == recovered_tokens

        # Test individual conversion
        assert tokenizer._convert_token_to_id('A') == ord('A') + tokenizer.offset
        assert tokenizer._convert_id_to_token(ord('A') + tokenizer.offset) == 'A'

    def test_pcap_file_parsing(self, sample_pcap_file, sample_packets):
        """Test PCAP file parsing functionality"""
        # Parse complete packets
        packet_bytes = PCAPTextTokenizer.parse_pcap_file(sample_pcap_file)

        assert len(packet_bytes) == len(sample_packets)
        assert all(isinstance(data, bytes) for data in packet_bytes)

        # Verify packet content matches
        original_bytes = [bytes(pkt) for pkt in sample_packets]
        assert packet_bytes == original_bytes

    def test_pcap_parsing_modes(self, sample_pcap_file):
        """Test different PCAP parsing modes"""
        # Full packet parsing
        full_packets = PCAPTextTokenizer.parse_pcap_file(sample_pcap_file)

        # Payload-only parsing
        payload_packets = PCAPTextTokenizer.parse_pcap_file(
            sample_pcap_file, extract_payload=True
        )

        # Skip link layer parsing
        no_link_packets = PCAPTextTokenizer.parse_pcap_file(
            sample_pcap_file, skip_link_layer=True
        )

        # Payload-only should be smaller than full packets
        assert all(len(payload) <= len(full)
                   for payload, full in zip(payload_packets, full_packets))

        # No-link-layer should be smaller than full packets
        assert all(len(no_link) < len(full)
                   for no_link, full in zip(no_link_packets, full_packets))

    def test_pcap_tokenization(self, tokenizer, sample_pcap_file):
        """Test PCAP data tokenization"""
        tokens = tokenizer.tokenize_pcap(sample_pcap_file)

        # Should start and end with special tokens
        assert tokens[0] == tokenizer.pcap_start_token
        assert tokens[-1] == tokenizer.pcap_end_token

        # Should contain packet separator tokens
        sep_count = tokens.count(tokenizer.packet_sep_token)
        # For n packets, we should have n-1 separators
        packet_count = len(PCAPTextTokenizer.parse_pcap_file(sample_pcap_file))
        assert sep_count == packet_count - 1

        # All non-special tokens should be single characters
        special_tokens = {tokenizer.pcap_start_token, tokenizer.pcap_end_token,
                          tokenizer.packet_sep_token}
        for token in tokens:
            if token not in special_tokens:
                assert len(token) == 1

    def test_pcap_tokenization_modes(self, tokenizer, sample_pcap_file):
        """Test PCAP tokenization with different extraction modes"""
        full_tokens = tokenizer.tokenize_pcap(sample_pcap_file)
        payload_tokens = tokenizer.tokenize_pcap(sample_pcap_file, extract_payload=True)
        no_link_tokens = tokenizer.tokenize_pcap(sample_pcap_file, skip_link_layer=True)

        # Payload-only should have fewer tokens
        assert len(payload_tokens) <= len(full_tokens)

        # No-link-layer should have fewer tokens
        assert len(no_link_tokens) < len(full_tokens)

        # All should have proper structure
        for tokens in [full_tokens, payload_tokens, no_link_tokens]:
            assert tokens[0] == tokenizer.pcap_start_token
            assert tokens[-1] == tokenizer.pcap_end_token

    def test_mixed_tokenization(self, tokenizer, sample_pcap_file):
        """Test mixed text and PCAP tokenization"""
        text = "Network traffic analysis: "

        # Test with both text and PCAP
        mixed_tokens = tokenizer.tokenize_mixed(text=text, pcap_data=sample_pcap_file)

        # Should start with text tokens
        text_tokens = tokenizer.tokenize(text)
        assert mixed_tokens[:len(text_tokens)] == text_tokens

        # Should contain PCAP tokens after text
        assert tokenizer.pcap_start_token in mixed_tokens
        assert tokenizer.pcap_end_token in mixed_tokens

        # Test with only text
        text_only = tokenizer.tokenize_mixed(text=text)
        assert text_only == text_tokens

        # Test with only PCAP
        pcap_only = tokenizer.tokenize_mixed(pcap_data=sample_pcap_file)
        pcap_tokens = tokenizer.tokenize_pcap(sample_pcap_file)
        assert pcap_only == pcap_tokens

    def test_packet_decoding(self, tokenizer, sample_pcap_file, sample_packets):
        """Test decoding packets back from tokens"""
        # Tokenize PCAP data
        tokens = tokenizer.tokenize_pcap(sample_pcap_file)

        # Decode packets back
        decoded_packets = tokenizer.decode_packets_from_tokens(tokens)

        # Should match original packet count
        assert len(decoded_packets) == len(sample_packets)

        # Should match original packet bytes
        original_bytes = [bytes(pkt) for pkt in sample_packets]
        assert decoded_packets == original_bytes

    def test_packet_decoding_edge_cases(self, tokenizer):
        """Test packet decoding with edge cases"""
        # Empty token list
        assert tokenizer.decode_packets_from_tokens([]) == []

        # Tokens without PCAP markers
        text_tokens = tokenizer.tokenize("Hello world")
        assert tokenizer.decode_packets_from_tokens(text_tokens) == []

        # Incomplete PCAP structure
        incomplete_tokens = [tokenizer.pcap_start_token, 'A', 'B', 'C']  # Missing end token
        decoded = tokenizer.decode_packets_from_tokens(incomplete_tokens)
        assert len(decoded) == 0  # Should handle gracefully

    def test_string_conversion(self, tokenizer):
        """Test token to string conversion"""
        text = "Hello 🌍 World!"
        tokens = tokenizer.tokenize(text)

        # Convert back to string
        recovered_text = tokenizer.convert_tokens_to_string(tokens)
        assert recovered_text == text

    def test_special_tokens_handling(self, tokenizer):
        """Test special tokens mask and handling"""
        text = "Test"
        token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

        # Test special tokens mask
        mask = tokenizer.get_special_tokens_mask(token_ids)
        assert len(mask) == len(token_ids) + 1  # +1 for EOS

        # Test building inputs with special tokens
        inputs = tokenizer.build_inputs_with_special_tokens(token_ids)
        assert inputs[-1] == tokenizer.eos_token_id

    def test_large_text_handling(self, tokenizer):
        """Test handling of large text inputs"""
        # Large text string
        large_text = "A" * 10000
        tokens = tokenizer.tokenize(large_text)
        assert len(tokens) == 10000

        # Convert to IDs
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        recovered_text = tokenizer.decode(token_ids, skip_special_tokens=True)

        assert recovered_text == large_text

    def test_unicode_edge_cases(self, tokenizer):
        """Test various Unicode edge cases"""
        test_cases = [
            "🚀🌟💻",  # Emojis
            "café naïve résumé",  # Accented characters
            "こんにちは",  # Japanese
            "Здравствуй",  # Cyrillic
            "\u200b\u200c\u200d",  # Zero-width characters
        ]

        for test_text in test_cases:
            tokens = tokenizer.tokenize(test_text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            # 1. Test that the conversion from IDs back to tokens is correct
            tokens_from_ids = tokenizer.convert_ids_to_tokens(token_ids)
            assert tokens_from_ids == tokens

            # 2. Test that the conversion from tokens back to a string is correct
            recovered_text = tokenizer.convert_tokens_to_string(tokens)
            assert recovered_text == test_text

    def test_vocabulary_consistency(self, tokenizer):
        """Test vocabulary consistency across operations"""
        vocab = tokenizer.get_vocab()

        # Test that all byte values are mappable
        for i in range(256):
            char = chr(i)
            token_id = tokenizer._convert_token_to_id(char)
            assert token_id is not None

            recovered_char = tokenizer._convert_id_to_token(token_id)
            assert recovered_char == char

            # Verify it's in vocabulary
            assert char in vocab or token_id in vocab.values()

    def test_empty_pcap_file(self, tokenizer, temp_dir):
        """Test handling of empty PCAP files"""
        empty_pcap = temp_dir / "empty.pcap"
        # Create an empty PCAP file
        wrpcap(str(empty_pcap), [])

        tokens = tokenizer.tokenize_pcap(empty_pcap)
        # Should still have start and end tokens
        assert tokens == [tokenizer.pcap_start_token, tokenizer.pcap_end_token]

    def test_single_packet_pcap(self, tokenizer, temp_dir):
        """Test PCAP file with single packet"""
        single_packet = Ether() / IP() / TCP() / Raw(b"Hello")
        single_pcap = temp_dir / "single.pcap"
        wrpcap(str(single_pcap), single_packet)

        tokens = tokenizer.tokenize_pcap(single_pcap)

        # Should have start/end tokens but no separator
        assert tokens[0] == tokenizer.pcap_start_token
        assert tokens[-1] == tokenizer.pcap_end_token
        assert tokenizer.packet_sep_token not in tokens

        # Decode should work correctly
        decoded = tokenizer.decode_packets_from_tokens(tokens)
        assert len(decoded) == 1
        assert decoded[0] == bytes(single_packet)

    @pytest.mark.parametrize("extract_payload,skip_link_layer", [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ])
    def test_all_parsing_combinations(self, tokenizer, sample_pcap_file,
                                      extract_payload, skip_link_layer):
        """Test all combinations of parsing parameters"""
        tokens = tokenizer.tokenize_pcap(
            sample_pcap_file,
            extract_payload=extract_payload,
            skip_link_layer=skip_link_layer
        )

        # Basic structure should always be maintained
        assert tokens[0] == tokenizer.pcap_start_token
        assert tokens[-1] == tokenizer.pcap_end_token

        # Should be able to decode
        decoded = tokenizer.decode_packets_from_tokens(tokens)
        assert len(decoded) > 0  # Should have some packets

    def test_tokenizer_state_consistency(self, tokenizer):
        """Test that tokenizer maintains consistent state"""
        # Multiple operations shouldn't affect each other
        text1 = "First text"
        text2 = "Second text"

        tokens1_first = tokenizer.tokenize(text1)
        tokens2 = tokenizer.tokenize(text2)
        tokens1_second = tokenizer.tokenize(text1)

        # Should get same results
        assert tokens1_first == tokens1_second
        assert tokens1_first != tokens2

    def test_performance_characteristics(self, tokenizer, temp_dir):
        """Basic performance and memory usage tests"""
        # Create a larger PCAP file for testing
        large_packets = []
        for i in range(100):
            packet = (Ether() / IP(dst=f"192.168.1.{i}") /
                      TCP(dport=80) / Raw(b"Data" * 100))
            large_packets.append(packet)

        large_pcap = temp_dir / "large.pcap"
        wrpcap(str(large_pcap), large_packets)

        # Tokenize and verify basic properties
        tokens = tokenizer.tokenize_pcap(large_pcap)

        # Should handle large files
        assert len(tokens) > 1000
        assert tokens[0] == tokenizer.pcap_start_token
        assert tokens[-1] == tokenizer.pcap_end_token

        # Separator count should match packet count
        sep_count = tokens.count(tokenizer.packet_sep_token)
        assert sep_count == len(large_packets) - 1



    class TestExactExpectedOutputs:
        """Tests with exact expected outputs for given inputs"""

        @pytest.fixture
        def tokenizer(self):
            return PCAPTextTokenizer()

        def test_exact_text_tokenization(self, tokenizer):
            """Test exact expected token outputs for text inputs"""

            test_cases = [
                # Input -> Exact expected tokens
                ("hello", ["h", "e", "l", "l", "o"]),
                ("world", ["w", "o", "r", "l", "d"]),
                ("helloworld", ["h", "e", "l", "l", "o", "w", "o", "r", "l", "d"]),
                ("Hello World", ["H", "e", "l", "l", "o", " ", "W", "o", "r", "l", "d"]),
                ("123", ["1", "2", "3"]),
                ("test!", ["t", "e", "s", "t", "!"]),
                ("a b c", ["a", " ", "b", " ", "c"]),
                ("ABC", ["A", "B", "C"]),
                ("", []),
                ("x", ["x"]),
            ]

            for input_text, expected_tokens in test_cases:
                actual_tokens = tokenizer.tokenize(input_text)

                assert actual_tokens == expected_tokens, (
                    f"\nINPUT: '{input_text}'"
                    f"\nEXPECTED: {expected_tokens}"
                    f"\nACTUAL:   {actual_tokens}"
                    f"\nMISMATCH!"
                )

                print(f"✓ '{input_text}' -> {actual_tokens}")

        def test_exact_token_to_id_conversion(self, tokenizer):
            """Test exact expected token ID outputs"""
            # The tokenizer has offset=3 (pad=0, eos=1, unk=2)
            # So token IDs = ord(char) + 3

            test_cases = [
                # (token, expected_id)
                ("a", 97 + 3),  # ord('a') = 97, +3 = 100
                ("b", 98 + 3),  # ord('b') = 98, +3 = 101
                ("A", 65 + 3),  # ord('A') = 65, +3 = 68
                ("B", 66 + 3),  # ord('B') = 66, +3 = 69
                ("0", 48 + 3),  # ord('0') = 48, +3 = 51
                ("1", 49 + 3),  # ord('1') = 49, +3 = 52
                (" ", 32 + 3),  # ord(' ') = 32, +3 = 35
                ("!", 33 + 3),  # ord('!') = 33, +3 = 36
            ]

            for token, expected_id in test_cases:
                actual_id = tokenizer._convert_token_to_id(token)

                assert actual_id == expected_id, (
                    f"\nTOKEN: '{token}'"
                    f"\nEXPECTED ID: {expected_id}"
                    f"\nACTUAL ID:   {actual_id}"
                    f"\nMISMATCH!"
                )

                print(f"✓ '{token}' -> ID {actual_id}")

        def test_exact_text_to_ids_full_pipeline(self, tokenizer):
            """Test complete text->tokens->ids pipeline with exact expected outputs"""

            test_cases = [
                # (input_text, expected_tokens, expected_ids)
                ("hi",
                 ["h", "i"],
                 [104 + 3, 105 + 3]),  # h=104, i=105, +3 each

                ("AI",
                 ["A", "I"],
                 [65 + 3, 73 + 3]),  # A=65, I=73, +3 each

                ("123",
                 ["1", "2", "3"],
                 [49 + 3, 50 + 3, 51 + 3]),  # 1=49, 2=50, 3=51, +3 each
            ]

            for input_text, expected_tokens, expected_ids in test_cases:
                # Test tokenization
                actual_tokens = tokenizer.tokenize(input_text)
                assert actual_tokens == expected_tokens, (
                    f"\nTEXT: '{input_text}'"
                    f"\nEXPECTED TOKENS: {expected_tokens}"
                    f"\nACTUAL TOKENS:   {actual_tokens}"
                )

                # Test ID conversion
                actual_ids = tokenizer.convert_tokens_to_ids(actual_tokens)
                assert actual_ids == expected_ids, (
                    f"\nTEXT: '{input_text}'"
                    f"\nEXPECTED IDS: {expected_ids}"
                    f"\nACTUAL IDS:   {actual_ids}"
                )

                print(f"✓ '{input_text}' -> {actual_tokens} -> {actual_ids}")

        def test_exact_pcap_tokenization_simple_bytes(self, tokenizer):
            """Test exact PCAP tokenization with known byte sequences"""

            # Get a unique temporary file name and close the handle immediately
            temp_f = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
            temp_path = Path(temp_f.name)
            temp_f.close()

            try:
                # Create a proper network packet with Ethernet header + payload
                packet = Ether(dst="aa:bb:cc:dd:ee:ff", src="11:22:33:44:55:66") / Raw(b"AB")
                wrpcap(str(temp_path), packet)

                # Get the actual packet bytes to determine expected tokens
                packet_bytes = bytes(packet)
                expected_tokens = ["<pcap>"] + [chr(b) for b in packet_bytes] + ["</pcap>"]

                actual_tokens = tokenizer.tokenize_pcap(temp_path)

                assert actual_tokens == expected_tokens, (
                    f"\nPCAP packet bytes: {packet_bytes.hex()}"
                    f"\nExpected {len(expected_tokens)} tokens, got {len(actual_tokens)}"
                    f"\nFirst 10 expected: {expected_tokens[:10]}"
                    f"\nFirst 10 actual:   {actual_tokens[:10]}"
                )

                print(f"✓ PCAP({len(packet_bytes)} bytes) -> {len(actual_tokens)} tokens")

            finally:
                temp_path.unlink()

        def test_exact_real_http_packet_structure(self, tokenizer):
            """Test with a realistic HTTP packet to show exact structure"""

            # Get a unique temporary file name and close the handle immediately
            temp_f = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
            temp_path = Path(temp_f.name)
            temp_f.close()

            try:
                # Create a realistic HTTP packet
                http_payload = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
                packet = (Ether(dst="aa:bb:cc:dd:ee:ff", src="11:22:33:44:55:66") /
                          IP(src="192.168.1.100", dst="8.8.8.8") /
                          TCP(sport=12345, dport=80) /
                          Raw(http_payload))

                wrpcap(str(temp_path), packet)

                # Analyze different extraction modes
                full_tokens = tokenizer.tokenize_pcap(temp_path, extract_payload=False)
                payload_tokens = tokenizer.tokenize_pcap(temp_path, extract_payload=True)
                no_link_tokens = tokenizer.tokenize_pcap(temp_path, skip_link_layer=True)

                # Get actual byte counts for verification
                full_bytes = bytes(packet)
                payload_bytes = http_payload
                no_link_bytes = bytes(packet.payload)  # IP layer and above

                # Verify token counts match byte counts (plus special tokens)
                assert len(full_tokens) == len(full_bytes) + 2, "Full: tokens != bytes + 2 special tokens"
                assert len(payload_tokens) == len(payload_bytes) + 2, "Payload: tokens != bytes + 2 special tokens"
                assert len(no_link_tokens) == len(no_link_bytes) + 2, "No-link: tokens != bytes + 2 special tokens"

                print("✓ HTTP Packet Analysis:")
                print(f"  Full packet: {len(full_bytes)} bytes -> {len(full_tokens)} tokens")
                print(f"  Payload only: {len(payload_bytes)} bytes -> {len(payload_tokens)} tokens")
                print(f"  No link layer: {len(no_link_bytes)} bytes -> {len(no_link_tokens)} tokens")

                # Show actual payload content in tokens
                payload_content = payload_tokens[1:-1]  # Remove <pcap> and </pcap>
                reconstructed_payload = ''.join(payload_content)

                print(f"  Payload content: {repr(reconstructed_payload)}")
                assert reconstructed_payload == http_payload.decode('utf-8', errors='replace')

            finally:
                # The file is guaranteed to be closed here, so unlink will succeed
                temp_path.unlink()

        def test_exact_pcap_tokenization_multi_packet(self, tokenizer):
            """Test exact multi-packet PCAP tokenization"""

            temp_f = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
            temp_path = Path(temp_f.name)
            temp_f.close()

            try:
                # Create two proper network packets
                packet1 = Ether() / IP(dst="192.168.1.1") / Raw(b"X")
                packet2 = Ether() / IP(dst="192.168.1.2") / Raw(b"Y")
                packets = [packet1, packet2]
                wrpcap(str(temp_path), packets)


                actual_tokens = tokenizer.tokenize_pcap(temp_path)

                # Expected structure: <pcap> + packet1_bytes + <pkt> + packet2_bytes + </pcap>
                packet1_bytes = bytes(packet1)
                packet2_bytes = bytes(packet2)

                expected_tokens = (["<pcap>"] +
                                   [chr(b) for b in packet1_bytes] +
                                   ["<pkt>"] +
                                   [chr(b) for b in packet2_bytes] +
                                   ["</pcap>"])

                assert actual_tokens == expected_tokens, (
                    f"\nPacket1 bytes: {len(packet1_bytes)}, Packet2 bytes: {len(packet2_bytes)}"
                    f"\nExpected {len(expected_tokens)} tokens, got {len(actual_tokens)}"
                    f"\nExpected separators: {expected_tokens.count('<pkt>')}"
                    f"\nActual separators: {actual_tokens.count('<pkt>')}"
                )

                print(
                    f"✓ PCAP(2 packets: {len(packet1_bytes)}+{len(packet2_bytes)} bytes) -> {len(actual_tokens)} tokens")

            finally:
                temp_path.unlink()

        def test_exact_mixed_tokenization(self, tokenizer):
            """Test exact mixed text + PCAP tokenization"""

            temp_f = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
            temp_path = Path(temp_f.name)
            temp_f.close()

            try:

                # Create a proper packet with known payload
                packet = Ether() / Raw(b"!")
                wrpcap(str(temp_path), packet)

                text = "hi"
                actual_tokens = tokenizer.tokenize_mixed(text=text, pcap_data=temp_path)

                # Expected: text tokens + PCAP tokens
                packet_bytes = bytes(packet)
                expected_tokens = (["h", "i"] +
                                   ["<pcap>"] +
                                   [chr(b) for b in packet_bytes] +
                                   ["</pcap>"])

                assert actual_tokens == expected_tokens, (
                    f"\nMIXED: text='hi' + PCAP({len(packet_bytes)} bytes)"
                    f"\nExpected {len(expected_tokens)} tokens, got {len(actual_tokens)}"
                    f"\nText part: {actual_tokens[:2]}"
                    f"\nPCAP start: {actual_tokens[2:5]}"
                )

                print(f"✓ MIXED('hi' + PCAP({len(packet_bytes)} bytes)) -> {len(actual_tokens)} tokens")

            finally:
                temp_path.unlink()

        def test_exact_packet_decoding(self, tokenizer):
            """Test exact packet decoding from tokens"""

            test_cases = [
                # (input_tokens, expected_decoded_packets)
                (
                    ["<pcap>", "A", "B", "</pcap>"],
                    [b"AB"]
                ),
                (
                    ["<pcap>", "X", "<pkt>", "Y", "Z", "</pcap>"],
                    [b"X", b"YZ"]
                ),
                (
                    ["<pcap>", "</pcap>"],  # Empty PCAP
                    []
                ),
            ]

            for input_tokens, expected_packets in test_cases:
                actual_packets = tokenizer.decode_packets_from_tokens(input_tokens)

                assert actual_packets == expected_packets, (
                    f"\nTOKENS: {input_tokens}"
                    f"\nEXPECTED PACKETS: {[p.hex() for p in expected_packets]}"
                    f"\nACTUAL PACKETS:   {[p.hex() for p in actual_packets]}"
                )

                packet_hex = [p.hex() for p in actual_packets]
                print(f"✓ {input_tokens} -> packets {packet_hex}")

        def test_exact_string_roundtrip(self, tokenizer):
            """Test exact string conversion roundtrip"""

            test_cases = [
                "hello",
                "world",
                "test123",
                "Hello World!",
                "",
                "a",
            ]

            for input_text in test_cases:
                # Forward: text -> tokens -> string
                tokens = tokenizer.tokenize(input_text)
                recovered_text = tokenizer.convert_tokens_to_string(tokens)

                assert recovered_text == input_text, (
                    f"\nINPUT TEXT: '{input_text}'"
                    f"\nTOKENS:     {tokens}"
                    f"\nRECOVERED:  '{recovered_text}'"
                    f"\nROUNDTRIP FAILED!"
                )

                print(f"✓ '{input_text}' -> {tokens} -> '{recovered_text}'")

        def test_exact_unicode_tokenization(self, tokenizer):
            """Test exact Unicode tokenization with known byte sequences"""

            test_cases = [
                # The first token for 'é' should be the character for its first UTF-8 byte (0xC3 -> 'Ã')
                ("é", 2, chr("é".encode('utf-8')[0])),
                ("🙂", 4, None),  # 🙂 = 4 bytes in UTF-8, no first token check needed
            ]

            for input_text, expected_byte_count, expected_first in test_cases:
                tokens = tokenizer.tokenize(input_text)

                # Check total token count matches UTF-8 byte count
                utf8_bytes = input_text.encode('utf-8')
                assert len(tokens) == expected_byte_count, (
                    f"\nTEXT: '{input_text}'"
                    f"\nUTF-8 BYTES: {utf8_bytes.hex()}"
                    f"\nEXPECTED BYTE COUNT: {expected_byte_count}"
                    f"\nACTUAL TOKEN COUNT:  {len(tokens)}"
                )

                # Check first token if specified
                if expected_first and tokens:
                    assert tokens[0] == expected_first, (
                        f"\nTEXT: '{input_text}'"
                        f"\nEXPECTED FIRST TOKEN: '{expected_first}'"
                        f"\nACTUAL FIRST TOKEN:   '{tokens[0]}'"
                    )

                print(f"✓ '{input_text}' -> {len(tokens)} tokens: {tokens}")

        def test_exact_special_token_ids(self, tokenizer):
            """Test exact special token ID values"""

            expected_special_ids = {
                # Default special token IDs based on tokenizer setup
                "pad_token_id": 0,
                "eos_token_id": 1,
                "unk_token_id": 2,
            }

            for attr_name, expected_id in expected_special_ids.items():
                actual_id = getattr(tokenizer, attr_name)

                assert actual_id == expected_id, (
                    f"\nSPECIAL TOKEN: {attr_name}"
                    f"\nEXPECTED ID: {expected_id}"
                    f"\nACTUAL ID:   {actual_id}"
                )

                print(f"✓ {attr_name} = {actual_id}")

        def test_exact_build_inputs_with_eos(self, tokenizer):
            """Test exact behavior of building inputs with EOS token"""

            # Input: "hi" -> tokens ["h", "i"] -> IDs [107, 108]
            text = "hi"
            tokens = tokenizer.tokenize(text)  # ["h", "i"]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)  # [107, 108]

            # Build inputs should add EOS (ID=1)
            inputs_with_eos = tokenizer.build_inputs_with_special_tokens(token_ids)
            expected_inputs = token_ids + [1]  # Add EOS token ID

            assert inputs_with_eos == expected_inputs, (
                f"\nINPUT TEXT: '{text}'"
                f"\nTOKEN IDS: {token_ids}"
                f"\nEXPECTED WITH EOS: {expected_inputs}"
                f"\nACTUAL WITH EOS:   {inputs_with_eos}"
            )

            print(f"✓ '{text}' -> IDs {token_ids} -> with EOS {inputs_with_eos}")

def print_tokenizer_info():
    """Print basic tokenizer configuration for reference"""
    tokenizer = PCAPTextTokenizer()

    print("\n" + "=" * 50)
    print("TOKENIZER CONFIGURATION:")
    print("=" * 50)
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"offset: {tokenizer.offset}")
    print(f"pad_token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"eos_token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"unk_token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    print(f"pcap_start_token: '{tokenizer.pcap_start_token}'")
    print(f"pcap_end_token: '{tokenizer.pcap_end_token}'")
    print(f"packet_sep_token: '{tokenizer.packet_sep_token}'")
    print("=" * 50)


if __name__ == "__main__":
    print_tokenizer_info()
    pytest.main([__file__, "-v", "-s", "--tb=short"])