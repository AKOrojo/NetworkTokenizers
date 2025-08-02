"""
A byte-level PCAP tokenizer with special tokens and packet separation.
"""

from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple

import dpkt
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Special token definitions
_PAD_TOKEN = "<pad>"
_UNK_TOKEN = "<unk>"
_EOS_TOKEN = "<eos>"
_BOS_TOKEN = "<bos>"
_PKT_TOKEN = "<pkt>"

# Special token IDs (occupy the lowest values)
_PAD_ID = 0
_UNK_ID = 1
_EOS_ID = 2
_BOS_ID = 3
_PKT_ID = 4

# Number of special tokens
_NUM_SPECIAL_TOKENS = 5

# Offset for byte values (they start after special tokens)
_BYTE_OFFSET = _NUM_SPECIAL_TOKENS

# Total vocabulary size (special tokens + 256 byte values)
_VOCAB_SIZE = _NUM_SPECIAL_TOKENS + 256


class TokenPCAPByteTokenizer(PreTrainedTokenizer):
    """
    A byte-level tokenizer for raw network packet data from PCAP files with special tokens.

    This tokenizer treats each byte (0-255) as a distinct token, offset by the number of
    special tokens. It includes standard special tokens (PAD, UNK, EOS, BOS) plus a
    new <pkt> token to separate individual packets. Special tokens occupy the lowest
    token IDs (0-4), and byte values are mapped to IDs 5-260.

    Uses dpkt to read the original captured packet bytes and inserts <pkt> tokens
    between individual packets.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs) -> None:
        """
        Initializes the EnhancedPCAPByteTokenizer with special tokens.
        """
        # Set special tokens
        kwargs.setdefault('pad_token', _PAD_TOKEN)
        kwargs.setdefault('unk_token', _UNK_TOKEN)
        kwargs.setdefault('eos_token', _EOS_TOKEN)
        kwargs.setdefault('bos_token', _BOS_TOKEN)

        # Add custom special tokens
        additional_special_tokens = kwargs.get('additional_special_tokens', [])
        if _PKT_TOKEN not in additional_special_tokens:
            additional_special_tokens.append(_PKT_TOKEN)
        kwargs['additional_special_tokens'] = additional_special_tokens

        super().__init__(**kwargs)

        # Store special token IDs
        self._pad_id = _PAD_ID
        self._unk_id = _UNK_ID
        self._eos_id = _EOS_ID
        self._bos_id = _BOS_ID
        self._pkt_id = _PKT_ID

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary (special tokens + 256 byte values).
        """
        return _VOCAB_SIZE

    @property
    def pad_token_id(self) -> int:
        return self._pad_id

    @property
    def unk_token_id(self) -> int:
        return self._unk_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_id

    @property
    def bos_token_id(self) -> int:
        return self._bos_id

    @property
    def pkt_token_id(self) -> int:
        """Returns the ID of the packet separator token."""
        return self._pkt_id

    def get_vocab(self) -> Dict[str, int]:
        """
        Constructs the vocabulary dictionary including special tokens and byte tokens.
        """
        # Add special tokens
        vocab = {_PAD_TOKEN: _PAD_ID, _UNK_TOKEN: _UNK_ID, _EOS_TOKEN: _EOS_ID, _BOS_TOKEN: _BOS_ID,
                 _PKT_TOKEN: _PKT_ID}


        # Add byte tokens (with offset)
        for i in range(256):
            token = self._convert_id_to_token(i + _BYTE_OFFSET)
            vocab[token] = i + _BYTE_OFFSET

        return vocab

    def tokenize_pcap(self, pcap_path: Union[str, Path],
                     add_packet_separators: bool = True,
                     add_bos: bool = False,
                     add_eos: bool = False) -> List[str]:
        """
        Public method to tokenize a PCAP file.

        Args:
            pcap_path: Path to the PCAP file to tokenize
            add_packet_separators: Whether to add <pkt> tokens between packets
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token

        Returns:
            List of tokens representing the PCAP file
        """
        return self._tokenize(str(pcap_path),
                            add_packet_separators=add_packet_separators,
                            add_bos=add_bos,
                            add_eos=add_eos)

    def read_pcap_packets(self, pcap_path: Union[str, Path]) -> List[bytes]:
        """
        Public method to read individual packets from a PCAP file.

        Args:
            pcap_path: Path to the PCAP file to read

        Returns:
            List of raw packet bytes
        """
        return self._read_pcap_packets(pcap_path)

    def tokenize_pcap_to_ids(self, pcap_path: Union[str, Path],
                           add_packet_separators: bool = True,
                           add_bos: bool = False,
                           add_eos: bool = False) -> List[int]:
        """
        Public method to tokenize a PCAP file directly to token IDs.

        Args:
            pcap_path: Path to the PCAP file to tokenize
            add_packet_separators: Whether to add <pkt> tokens between packets
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token

        Returns:
            List of token IDs
        """
        tokens = self.tokenize_pcap(pcap_path, add_packet_separators, add_bos, add_eos)
        return [self._convert_token_to_id(token) for token in tokens]

    @staticmethod
    def _read_pcap_packets(pcap_path: Union[str, Path]) -> List[bytes]:
        """
        Reads a PCAP file and returns a list of individual packet bytes.

        Args:
            pcap_path: The path to the PCAP file.

        Returns:
            A list of bytes objects, one for each packet.

        Raises:
            FileNotFoundError: If the PCAP file does not exist.
            ValueError: If there is an error reading the PCAP file.
        """
        pcap_path = Path(pcap_path)
        if not pcap_path.is_file():
            raise FileNotFoundError(f"PCAP file not found at: {pcap_path}")

        try:
            packets = []
            with open(pcap_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                # buf contains the raw packet bytes as captured on the wire
                for ts, buf in pcap:
                    packets.append(buf)
            return packets
        except Exception as e:
            raise ValueError(f"Error reading PCAP file {pcap_path}: {e}")

    def _tokenize(self, text: str,
                 add_packet_separators: bool = True,
                 add_bos: bool = False,
                 add_eos: bool = False,
                 **kwargs) -> List[str]:
        """
        Tokenizes the content of a PCAP file specified by its path.

        Args:
            text: The file path to the PCAP file to tokenize.
            add_packet_separators: Whether to add <pkt> tokens between packets
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token

        Returns:
            A list of tokens including special tokens and byte tokens.
        """
        # Interpret the input 'text' as the path to the PCAP file
        pcap_file_path = text
        packets = self._read_pcap_packets(pcap_file_path)

        tokens = []

        # Add beginning-of-sequence token if requested
        if add_bos:
            tokens.append(_BOS_TOKEN)

        # Process each packet
        for i, packet_bytes in enumerate(packets):
            # Add packet separator before each packet (except the first)
            if add_packet_separators and i > 0:
                tokens.append(_PKT_TOKEN)

            # Convert each byte to its character representation
            packet_tokens = [chr(b) for b in packet_bytes]
            tokens.extend(packet_tokens)

        # Add end-of-sequence token if requested
        if add_eos:
            tokens.append(_EOS_TOKEN)

        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        """
        Converts a token to its ID, handling special tokens and byte tokens with offset.
        """
        # Handle special tokens
        if token == _PAD_TOKEN:
            return _PAD_ID
        elif token == _UNK_TOKEN:
            return _UNK_ID
        elif token == _EOS_TOKEN:
            return _EOS_ID
        elif token == _BOS_TOKEN:
            return _BOS_ID
        elif token == _PKT_TOKEN:
            return _PKT_ID

        # Handle byte tokens
        if len(token) == 1:
            byte_value = ord(token)
            if 0 <= byte_value <= 255:
                return byte_value + _BYTE_OFFSET

        # Return UNK token ID for unknown tokens
        return self._unk_id

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts an ID to its token, handling special tokens and byte tokens with offset.
        """
        if not (0 <= index < self.vocab_size):
            raise ValueError(f"ID must be between 0 and {self.vocab_size - 1}.")

        # Handle special tokens
        if index == _PAD_ID:
            return _PAD_TOKEN
        elif index == _UNK_ID:
            return _UNK_TOKEN
        elif index == _EOS_ID:
            return _EOS_TOKEN
        elif index == _BOS_ID:
            return _BOS_TOKEN
        elif index == _PKT_ID:
            return _PKT_TOKEN

        # Handle byte tokens (subtract offset)
        byte_value = index - _BYTE_OFFSET
        if 0 <= byte_value <= 255:
            return chr(byte_value)

        # This should not happen if index is in valid range
        return _UNK_TOKEN

    @staticmethod
    def convert_tokens_to_string(tokens: List[str], **kwargs) -> str:
        """
        Converts a sequence of tokens back into a string, filtering out special tokens.
        """
        # Filter out special tokens and convert byte tokens to string
        byte_tokens = [t for t in tokens if t not in {_PAD_TOKEN, _UNK_TOKEN, _EOS_TOKEN, _BOS_TOKEN, _PKT_TOKEN}]
        return "".join(byte_tokens)

    def decode(self, token_ids: Union[List[int], int],
              skip_special_tokens: bool = True,
              **kwargs) -> Union[bytes, str]:
        """
        Converts a sequence of token IDs back into bytes or string.

        Args:
            token_ids: A list of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            The decoded bytes object (if skip_special_tokens=True) or string representation.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        if skip_special_tokens:
            # Filter out special token IDs and convert to bytes
            byte_ids = []
            for token_id in token_ids:
                if _BYTE_OFFSET <= token_id < _BYTE_OFFSET + 256:
                    byte_ids.append(token_id - _BYTE_OFFSET)
            return bytes(byte_ids)
        else:
            # Convert all tokens to string representation
            tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
            return "".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int],
                                       token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs by adding BOS and EOS tokens if configured.
        """
        # Add BOS at the beginning
        result = [self._bos_id] + token_ids_0

        # Add second sequence if provided
        if token_ids_1 is not None:
            result = result + [self._pkt_id] + token_ids_1

        # Add EOS at the end
        result = result + [self._eos_id]

        return result

    def get_special_tokens_mask(self, token_ids_0: List[int],
                              token_ids_1: Optional[List[int]] = None,
                              already_has_special_tokens: bool = False) -> List[int]:
        """
        Returns a mask indicating which tokens are special tokens.
        """
        if already_has_special_tokens:
            return [1 if token_id < _BYTE_OFFSET else 0 for token_id in token_ids_0]

        # Mask for build_inputs_with_special_tokens output
        mask = [1]  # BOS
        mask.extend([0] * len(token_ids_0))  # Regular tokens

        if token_ids_1 is not None:
            mask.append(1)  # PKT separator
            mask.extend([0] * len(token_ids_1))  # Regular tokens

        mask.append(1)  # EOS

        return mask

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary to a file.
        """
        import json

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        filename = "vocab.json"
        if filename_prefix is not None:
            filename = f"{filename_prefix}-{filename}"

        vocab_file = save_directory / filename

        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_vocab(), f, ensure_ascii=False, indent=2)

        return (str(vocab_file),)