"""
A field-aware and packet-aware byte-level PCAP tokenizer.

This tokenizer processes PCAP files by treating each byte as a token, but it
also inserts special tokens to mark boundaries:
- <pkt>: Separates individual packets.
- <sep>: Separates protocol fields within a packet.

It also includes standard special tokens (<pad>, </s>, <unk>) for compatibility
with transformer models.
"""

from typing import List, Dict, Optional, Tuple

from scapy.all import rdpcap, Packet, Raw
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class PCAPFieldTokenizer(PreTrainedTokenizer):
    """
    A field-aware and packet-aware, byte-level tokenizer for PCAP files.

    This tokenizer operates at the byte level but is aware of the underlying
    protocol structure. It parses each packet, inserts a <sep> token between
    each protocol field, and a <pkt> token between each packet.

    Example tokenization:
    [pkt1_f1]<sep>[pkt1_f2]<pkt>[pkt2_f1]<sep>[pkt2_f2]...

    The vocabulary consists of:
    1. 256 tokens for all possible byte values (0-255).
    2. Special tokens: <pad>, </s>, <unk>, <sep>, and <pkt>.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="<sep>",
        pkt_sep_token="<pkt>",
        **kwargs,
    ) -> None:
        """
        Initializes the PCAPFieldTokenizer.

        Args:
            pad_token (str): The padding token.
            eos_token (str): The end-of-sequence token.
            unk_token (str): The unknown token.
            sep_token (str): The token to separate protocol fields.
            pkt_sep_token (str): The token to separate packets.
        """
        # Define special tokens as AddedToken objects for proper handling
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        pkt_sep_token = AddedToken(pkt_sep_token, lstrip=False, rstrip=False) if isinstance(pkt_sep_token, str) else pkt_sep_token

        self._byte_vocab_size = 256

        super().__init__(
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            additional_special_tokens=[sep_token, pkt_sep_token],
            **kwargs,
        )

        self.sep_token = str(sep_token)
        self.pkt_sep_token = str(pkt_sep_token)

        # The offset for byte tokens is the total number of special tokens defined.
        self.byte_offset = len(self.all_special_ids)

    @property
    def vocab_size(self) -> int:
        """Returns the size of the core vocabulary (256 bytes)."""
        return self._byte_vocab_size

    def __len__(self):
        """Returns the total size of the vocabulary including all special tokens."""
        return self.vocab_size + len(self.all_special_ids)

    def get_vocab(self) -> Dict[str, int]:
        """Constructs the full vocabulary dictionary."""
        vocab = self.get_added_vocab()  # Gets all special tokens
        for i in range(self.vocab_size):
            vocab[chr(i)] = i + self.byte_offset
        return vocab

    @staticmethod
    def _get_field_bytes(layer: Packet, field) -> bytes:
        """Extracts the raw byte representation of a single field."""
        field_val = layer.getfieldval(field.name)
        return field.addfield(None, b"", field_val)

    def _tokenize_packet_fields(self, packet: Packet) -> List[bytes]:
        """Tokenizes a single packet into a list of its field bytes."""
        field_byte_list = []
        current_layer = packet

        while current_layer:
            for field in current_layer.fields_desc:
                if field.name in current_layer.fields:
                    field_bytes = self._get_field_bytes(current_layer, field)
                    field_byte_list.append(field_bytes)

            if isinstance(current_layer.payload, Packet):
                current_layer = current_layer.payload
            elif isinstance(current_layer.payload, Raw):
                field_byte_list.append(bytes(current_layer.payload))
                current_layer = None
            else:
                current_layer = None

        return field_byte_list

    def _tokenize(self, pcap_path: str, **kwargs) -> List[str]:
        """Performs the main tokenization of a PCAP file."""
        try:
            packets = rdpcap(pcap_path)
        except Exception as e:
            logger.error(f"Failed to read or parse PCAP file at {pcap_path}: {e}")
            return []

        all_tokens = []
        sep_token_as_list = [self.sep_token]
        pkt_sep_token_as_list = [self.pkt_sep_token]
        num_packets = len(packets)

        for i, packet in enumerate(packets):
            field_bytes_list = self._tokenize_packet_fields(packet)
            num_fields = len(field_bytes_list)

            for j, field_bytes in enumerate(field_bytes_list):
                field_char_tokens = [chr(b) for b in field_bytes]
                all_tokens.extend(field_char_tokens)

                if j < num_fields - 1:
                    all_tokens.extend(sep_token_as_list)

            if i < num_packets - 1:
                all_tokens.extend(pkt_sep_token_as_list)

        return all_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an ID using the vocab."""
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        if len(token) == 1:
            return ord(token) + self.byte_offset
        return self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocab."""
        if index in self.added_tokens_decoder:
            return self.added_tokens_decoder[index].content
        if 0 <= (index - self.byte_offset) < self._byte_vocab_size:
            return chr(index - self.byte_offset)
        return self.unk_token

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Builds a sequence by adding an EOS token at the end."""
        if token_ids_1:
            return token_ids_0 + token_ids_1 + [self.eos_token_id]
        return token_ids_0 + [self.eos_token_id]

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Returns a mask indicating the position of special tokens."""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]

        return ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple:
        """This tokenizer does not save a vocabulary file."""
        return ()
