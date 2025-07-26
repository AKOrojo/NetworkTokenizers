"""
A pure byte-level PCAP tokenizer with no special tokens.
"""

from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple

from scapy.all import rdpcap
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

# The vocabulary size is fixed to the number of possible byte values.
_VOCAB_SIZE = 256

class PCAPByteTokenizer(PreTrainedTokenizer):
    """
    A byte-level tokenizer for raw network packet data from PCAP files.

    This tokenizer treats each byte (0-255) as a distinct token. It does not use any
    special tokens (e.g., PAD, EOS, UNK, or separators). It reads a PCAP file,
    concatenates the raw bytes of all packets into a single sequence, and maps
    each byte to its integer value as the token ID.

    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs) -> None:
        """
        Initializes the PCAPByteTokenizer.

        The vocabulary is fixed and consists of 256 tokens, corresponding to all
        possible byte values (0-255). No special tokens are added.
        """

        super().__init__(
            pad_token=None,
            eos_token=None,
            unk_token=None,
            bos_token=None,
            sep_token=None,
            cls_token=None,
            mask_token=None,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary, which is always 256 for the byte values.
        """
        return _VOCAB_SIZE

    def get_vocab(self) -> Dict[str, int]:
        """
        Constructs the vocabulary dictionary on the fly. The tokens are represented as
        characters corresponding to their byte value.
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    @staticmethod
    def _read_pcap_bytes(pcap_path: Union[str, Path]) -> bytes:
        """
        Reads a PCAP file and returns the concatenated raw bytes of all packets.

        Args:
            pcap_path: The path to the PCAP file.

        Returns:
            A single bytes object containing all packet data from the file.

        Raises:
            FileNotFoundError: If the PCAP file does not exist.
            ValueError: If there is an error parsing the PCAP file with Scapy.
        """
        pcap_path = Path(pcap_path)
        if not pcap_path.is_file():
            raise FileNotFoundError(f"PCAP file not found at: {pcap_path}")

        try:
            packets = rdpcap(str(pcap_path))
            all_bytes = b"".join(bytes(p) for p in packets)
            return all_bytes
        except Exception as e:
            raise ValueError(f"Error parsing PCAP file {pcap_path}: {e}")

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenizes the content of a PCAP file specified by its path.

        The `text` argument is interpreted as the file path to the PCAP.

        Args:
            text: The file path to the PCAP file to tokenize.

        Returns:
            A list of single-character strings, where each character represents a byte.
        """
        # Interpret the input 'text' as the path to the PCAP file.
        pcap_file_path = text
        raw_bytes = self._read_pcap_bytes(pcap_file_path)

        # Convert each byte into its character representation. This list of
        # characters serves as the "tokens".
        tokens = [chr(b) for b in raw_bytes]
        return tokens

    @staticmethod
    def _convert_token_to_id(token: str, **kwargs) -> int:
        """
        Converts a single-character token back to its byte value (ID).
        """

        if len(token) != 1:
            raise ValueError("Token must be a single character.")
        return ord(token)

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts an ID (a byte value from 0-255) to its single-character token.
        """

        if not (0 <= index < self.vocab_size):
            raise ValueError(f"ID must be between 0 and {self.vocab_size - 1}.")
        return chr(index)

    @staticmethod
    def convert_tokens_to_string(tokens: List[str], **kwargs) -> str:
        """
        Converts a sequence of single-character tokens back into a single string.
        """
        return "".join(tokens)

    @staticmethod
    def decode(token_ids: Union[List[int], int], **kwargs) -> bytes:
        """
        Converts a sequence of token IDs back into a raw bytes object.

        Args:
            token_ids: A list of token IDs (integers 0-255) to decode.

        Returns:
            The decoded `bytes` object.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        # Convert IDs directly to a bytes object.
        return bytes(token_ids)

    @staticmethod
    def build_inputs_with_special_tokens(token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, **kwargs) -> List[int]:
        """
        This tokenizer does not use special tokens, so it simply returns the
        input sequence(s) concatenated.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    @staticmethod
    def get_special_tokens_mask(token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, **kwargs) -> List[int]:
        """
        Returns a mask of all zeros since there are no special tokens.
        """
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * (len(token_ids_0) + len(token_ids_1))

    @staticmethod
    def save_vocabulary(**kwargs) -> Tuple:
        """
        This tokenizer has a fixed, algorithmically-defined vocabulary and does
        not save a vocabulary file.
        """
        return ()