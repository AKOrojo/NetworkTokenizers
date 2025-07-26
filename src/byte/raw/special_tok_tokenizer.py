# coding=utf-8
"""
A pure byte-level PCAP tokenizer with standard special tokens and a packet
delimiter token.

This tokenizer processes PCAP files by treating each byte as a unique token
and inserting a special <pkt> token between each packet. It also includes
standard special tokens (<pad>, </s>, <unk>) for compatibility with
transformer models.

"""

from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple

from scapy.all import rdpcap
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class PCAPByteTokenizer(PreTrainedTokenizer):
    """
    A byte-level tokenizer for raw network packet data from PCAP files,
    including standard special tokens and a packet delimiter.

    This tokenizer treats each byte (0-255) as a distinct token and inserts a
    special `<pkt>` token between the raw bytes of each packet. It reads a PCAP
    file, processes packets sequentially, and maps each byte to its integer
    value. It also includes `<pad>`, `</s>`, and `<unk>` tokens required for
    most transformer model training pipelines.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        pkt_token="<pkt>",
        **kwargs,
    ) -> None:
        """
        Initializes the PCAPByteTokenizer.

        The vocabulary consists of 256 byte tokens plus the specified special
        tokens.

        Args:
            pad_token (str): The padding token.
            eos_token (str): The end-of-sequence token.
            unk_token (str): The unknown token.
            pkt_token (str): The token used to separate packets.
        """
        # Define special tokens as AddedToken objects for proper handling
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pkt_token = AddedToken(pkt_token, lstrip=False, rstrip=False) if isinstance(pkt_token, str) else pkt_token

        # The core vocabulary size is fixed to the number of possible byte values.
        self._byte_vocab_size = 256

        super().__init__(
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pkt_token=pkt_token,
            **kwargs,
        )

        # The byte vocabulary is mapped to token IDs starting after the initial
        # special tokens. The exact number is handled by the base class.
        self.byte_offset = len(self.all_special_ids)

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the core vocabulary (256 bytes).
        The total vocabulary size including special tokens is `len(tokenizer)`.
        """
        return self._byte_vocab_size

    def __len__(self):
        """Returns the total size of the vocabulary including all special tokens."""
        return self.vocab_size + len(self.all_special_ids)

    def get_vocab(self) -> Dict[str, int]:
        """
        Constructs and returns the full vocabulary dictionary, including byte
        tokens and all special tokens.
        """
        # Start with the special tokens from the base class method
        vocab = self.get_added_vocab()
        # Add the byte vocabulary
        for i in range(self.vocab_size):
            vocab[chr(i)] = i + self.byte_offset
        return vocab

    def _tokenize(self, pcap_path: str, **kwargs) -> List[str]:
        """
        Performs the main tokenization logic. Reads a PCAP file, inserts a
        <pkt> token between each packet, and converts the byte stream into a
        list of character tokens.

        Note: This tokenizer expects the input string to be a path to a PCAP file.
        """
        pcap_path_obj = Path(pcap_path)
        if not pcap_path_obj.is_file():
            logger.warning(f"PCAP file not found at: {pcap_path}. Returning empty list.")
            return []

        try:
            packets = rdpcap(pcap_path)
        except Exception as e:
            logger.error(f"Error parsing PCAP file {pcap_path}: {e}")
            return []

        if not packets:
            return []

        all_tokens = []
        # self.pkt_token is the AddedToken object; .content gets the string
        pkt_separator_token = self.pkt_token.content

        # Add tokens for the first packet
        all_tokens.extend(chr(b) for b in bytes(packets[0]))

        # For all subsequent packets, add the separator then the packet tokens
        for packet in packets[1:]:
            all_tokens.append(pkt_separator_token)
            all_tokens.extend(chr(b) for b in bytes(packet))

        return all_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an ID using the vocab."""
        # The added_tokens_encoder correctly maps all special tokens to their IDs.
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        # If it's not a special token, assume it's a byte token.
        if len(token) == 1:
            return ord(token) + self.byte_offset
        # If it's an unknown multi-character string, return the unk_token_id.
        return self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocab."""
        # The added_tokens_decoder maps special token IDs back to their string representation.
        if index in self.added_tokens_decoder:
            return self.added_tokens_decoder[index].content
        # Check if the index corresponds to a valid byte token.
        if 0 <= (index - self.byte_offset) < self._byte_vocab_size:
            return chr(index - self.byte_offset)
        # Otherwise, it's an unknown token.
        return self.unk_token

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens back into a single string. Special tokens
        are filtered out, resulting in a concatenation of byte characters.
        """
        return "".join(token for token in tokens if token not in self.all_special_tokens)

    def decode(
        self,
        token_ids: Union[List[int], int],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> bytes:
        """
        Converts a sequence of token IDs back into a raw bytes object.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        if skip_special_tokens:
            # Filter out IDs corresponding to special tokens
            filtered_ids = [
                _id for _id in token_ids if _id not in self.all_special_ids
            ]
        else:
            filtered_ids = token_ids

        # Convert the remaining IDs (which should be byte tokens) to bytes.
        # We subtract the offset to get the true byte value.
        byte_values = [
            (i - self.byte_offset)
            for i in filtered_ids
            if 0 <= (i - self.byte_offset) < self._byte_vocab_size
        ]
        return bytes(byte_values)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Builds a sequence by adding an EOS token at the end. The <pkt> tokens
        are assumed to already be present from the _tokenize step.
        """
        if token_ids_1:
            return token_ids_0 + token_ids_1 + [self.eos_token_id]
        return token_ids_0 + [self.eos_token_id]

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Returns a mask indicating the position of special tokens.
        If `already_has_special_tokens` is True, it will identify all special
        tokens in the input IDs, including the <pkt> tokens.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # This part handles the case where special tokens are being added by `build_inputs_with_special_tokens`
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]  # For the final EOS
        return ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple:
        """
        This tokenizer's vocabulary is algorithmically defined and does not need
        to be saved to a file. The special tokens configuration will be saved.
        """
        # The base class method will save the `tokenizer_config.json`, which
        # contains the definitions of our special tokens.
        return super().save_vocabulary(save_directory, filename_prefix)