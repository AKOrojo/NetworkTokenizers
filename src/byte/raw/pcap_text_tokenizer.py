# coding=utf-8
"""
PCAP + Text Tokenizer based on ByT5
Tokenizes both text (UTF-8 bytes) and raw network packet bytes the same way
"""
import warnings
from pathlib import Path
from typing import Optional, Union, List

from scapy.all import rdpcap, Raw
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class PCAPTextTokenizer(PreTrainedTokenizer):
    """
    A tokenizer that can handle both text and PCAP network packet data.

    For text: Uses UTF-8 byte encoding like ByT5
    For PCAP: Extracts raw packet bytes and tokenizes them the same way

    This tokenizer inherits from PreTrainedTokenizer and uses the same approach as ByT5
    for byte-level tokenization.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            pcap_start_token="<pcap>",
            pcap_end_token="</pcap>",
            packet_sep_token="<pkt>",
            extra_ids=125,
            additional_special_tokens=None,
            **kwargs,
    ) -> None:
        """
        Initialize the PCAP + Text tokenizer.

        Args:
            eos_token: End of sequence token
            unk_token: Unknown token
            pad_token: Padding token
            pcap_start_token: Token to mark start of PCAP data
            pcap_end_token: Token to mark end of PCAP data
            packet_sep_token: Token to separate individual packets
            extra_ids: Number of extra sentinel tokens
            additional_special_tokens: Additional special tokens
        """

        # Build the complete list of special tokens
        pcap_special_tokens = [pcap_start_token, pcap_end_token, packet_sep_token]

        # Add extra_ids to the special token list
        if extra_ids > 0:
            extra_id_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
            if additional_special_tokens is None:
                additional_special_tokens = extra_id_tokens + pcap_special_tokens
            else:
                additional_special_tokens = list(additional_special_tokens) + extra_id_tokens + pcap_special_tokens
        else:
            if additional_special_tokens is None:
                additional_special_tokens = pcap_special_tokens
            else:
                additional_special_tokens = list(additional_special_tokens) + pcap_special_tokens

        # Set up special tokens with proper formatting
        pad_token = AddedToken(pad_token, lstrip=True, rstrip=True) if isinstance(pad_token, str) else pad_token
        eos_token = AddedToken(eos_token, lstrip=True, rstrip=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=True, rstrip=True) if isinstance(unk_token, str) else unk_token

        # Store PCAP-specific tokens
        self.pcap_start_token = pcap_start_token
        self.pcap_end_token = pcap_end_token
        self.packet_sep_token = packet_sep_token

        # Set up the core token mapping
        self._added_tokens_decoder = {0: pad_token, 1: eos_token, 2: unk_token}
        self.offset = len(self._added_tokens_decoder)
        self._utf_vocab_size = 2 ** 8

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=0,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return self._utf_vocab_size

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size + self.offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    @staticmethod
    def parse_pcap_file(pcap_path: Union[str, Path],
                        extract_payload: bool = False,
                        skip_link_layer: bool = False) -> List[bytes]:
        """
        Parse a PCAP file and extract raw packet bytes using Scapy.

        Args:
            pcap_path: Path to the PCAP file
            extract_payload: If True, extract only application payload data
            skip_link_layer: If True, skip link layer headers (e.g., Ethernet)

        Returns:
            List of raw packet data as bytes objects
        """


        try:
            # Read PCAP file using Scapy
            packets = rdpcap(str(pcap_path))
            packet_bytes = []

            for packet in packets:
                if extract_payload:
                    # Extract only the payload (application data)
                    if Raw in packet:
                        data = bytes(packet[Raw])
                    else:
                        # If no Raw layer, try to get the highest layer payload
                        data = bytes(packet.payload) if hasattr(packet, 'payload') else bytes(packet)
                elif skip_link_layer:
                    # Skip link layer (Ethernet) but keep everything else
                    if hasattr(packet, 'payload') and packet.payload:
                        data = bytes(packet.payload)
                    else:
                        data = bytes(packet)
                else:
                    # Get the complete packet including link layer
                    data = bytes(packet)

                if data:  # Only add non-empty packets
                    packet_bytes.append(data)

            return packet_bytes

        except Exception as e:
            raise ValueError(f"Error parsing PCAP file {pcap_path}: {e}")

    @staticmethod
    def _tokenize_text(text_input: str) -> List[str]:
        """Tokenize text using UTF-8 byte encoding"""
        return [chr(i) for i in text_input.encode("utf-8")]

    def _tokenize_pcap_data(self, pcap_data: Union[str, Path],
                            extract_payload: bool = False,
                            skip_link_layer: bool = False) -> List[str]:
        """
        Tokenize PCAP data.

        Args:
            pcap_data: Can be:
                - Path to PCAP file (str or Path)
            extract_payload: If True, extract only application payload data
            skip_link_layer: If True, skip link layer headers when parsing files

        Returns:
            List of character tokens representing the packet bytes
        """
        tokens = [self.pcap_start_token]
        packets = []

        if isinstance(pcap_data, (str, Path)):
            # Parse PCAP file
            packets = self.parse_pcap_file(pcap_data, extract_payload, skip_link_layer)

        for i, packet_bytes in enumerate(packets):
            if i > 0:
                tokens.append(self.packet_sep_token)

            # Convert packet bytes to character tokens
            packet_tokens = [chr(b) for b in packet_bytes]
            tokens.extend(packet_tokens)

        tokens.append(self.pcap_end_token)
        return tokens

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize input text using UTF-8 byte encoding.
        Note: For PCAP data, use tokenize_pcap() method instead.
        """
        return self._tokenize_text(text)

    def tokenize_pcap(self, pcap_data: Union[str, Path],
                      extract_payload: bool = False,
                      skip_link_layer: bool = False) -> List[str]:
        """
        Public method to tokenize PCAP data.

        Args:
            pcap_data: PCAP file path
            extract_payload: If True, extract only application payload data
            skip_link_layer: If True, skip link layer headers when parsing files

        Returns:
            List of tokens
        """
        return self._tokenize_pcap_data(pcap_data, extract_payload, skip_link_layer)

    def tokenize_mixed(self, text: str = None,
                       pcap_data: Union[str, Path] = None,
                       extract_payload: bool = False,
                       skip_link_layer: bool = False) -> List[str]:
        """
        Tokenize both text and PCAP data together.

        Args:
            text: Text to tokenize
            pcap_data: PCAP data to tokenize
            extract_payload: If True, extract only application payload data
            skip_link_layer: If True, skip link layer headers when parsing files

        Returns:
            Combined list of tokens
        """
        tokens = []

        if text is not None:
            tokens.extend(self._tokenize_text(text))

        if pcap_data is not None:
            tokens.extend(self._tokenize_pcap_data(pcap_data, extract_payload, skip_link_layer))

        return tokens

    def _convert_token_to_id(self, token):
        """Convert a token (str) to an id using the vocab."""
        if len(token) != 1:
            return None
        else:
            return ord(token) + self.offset

    def _convert_id_to_token(self, index):
        """Convert an index (integer) to a token (str) using the vocab."""
        return chr(index - self.offset)

    def convert_tokens_to_string(self, tokens):
        """Convert a sequence of tokens back to a string."""
        byte_string = b""

        for token in tokens:
            if token in self.added_tokens_encoder:
                tok_string = token.encode("utf-8")
            else:
                tok_string = bytes([ord(token)])

            byte_string += tok_string

        return byte_string.decode("utf-8", errors="ignore")

    def decode_packets_from_tokens(self, tokens: List[str]) -> List[bytes]:
        """
        Extract and decode raw packet bytes from tokens.

        Args:
            tokens: List of tokens that may contain PCAP data

        Returns:
            List of raw packet bytes
        """
        packets = []
        current_packet = []
        in_pcap = False

        for token in tokens:
            if token == self.pcap_start_token:
                in_pcap = True
                current_packet = []
            elif token == self.pcap_end_token:
                if current_packet:
                    packets.append(bytes([ord(t) for t in current_packet]))
                in_pcap = False
                current_packet = []
            elif token == self.packet_sep_token:
                if current_packet:
                    packets.append(bytes([ord(t) for t in current_packet]))
                current_packet = []
            elif in_pcap and len(token) == 1:
                current_packet.append(token)

        return packets

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """Create special tokens mask"""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Add EOS token if not already present."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs with special tokens"""
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create token type IDs (returns zeros like ByT5)."""
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """ByT5-style tokenizer has no vocab file to save."""
        return ()