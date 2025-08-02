"""
A field-aware and packet-aware byte-level PCAP tokenizer using pyshark.

This tokenizer processes PCAP files by working with raw bytes and using
pyshark's Wireshark dissectors to determine field boundaries.
It inserts special tokens to mark boundaries:
- <pkt>: Separates individual packets.
- <sep>: Separates protocol fields within a packet.

Unlike Scapy-based approaches, this preserves the exact wire format.
"""

from typing import List, Dict, Optional, Tuple
import pyshark
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class PySharkFieldTokenizer(PreTrainedTokenizer):
    """
    A field-aware and packet-aware, byte-level tokenizer for PCAP files using pyshark.

    This tokenizer operates at the byte level using the exact wire format
    and uses pyshark's Wireshark dissectors to identify field boundaries.
    It inserts a <sep> token between each protocol field, and a <pkt> token
    between each packet.

    Example tokenization:
    [raw_field1_bytes]<sep>[raw_field2_bytes]<pkt>[raw_field1_bytes]<sep>...

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
        Initializes the PySharkFieldTokenizer.

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

        # Store these before calling super().__init__
        self.sep_token = str(sep_token)
        self.pkt_sep_token = str(pkt_sep_token)

        # Similar to ByT5: we have special tokens first, then byte tokens
        self._added_tokens_decoder = {0: pad_token, 1: eos_token, 2: unk_token, 3: sep_token, 4: pkt_sep_token}
        self.offset = len(self._added_tokens_decoder)  # Bytes start after special tokens
        self._byte_vocab_size = 256  # 256 possible byte values

        super().__init__(
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            additional_special_tokens=[sep_token, pkt_sep_token],
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """Returns the size of the core vocabulary (256 bytes)."""
        return self._byte_vocab_size

    def get_vocab(self) -> Dict[str, int]:
        """Constructs the full vocabulary dictionary."""
        # Start with added tokens (special tokens)
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size + self.offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _get_field_offsets(self, packet) -> List[Tuple[int, int, str]]:
        """
        Extract field boundary information from pyshark packet.
        Returns list of (start_offset, end_offset, field_name) tuples.
        """
        field_offsets = []

        try:
            # Iterate through all layers in the packet
            for layer in packet.layers:
                layer_name = layer.layer_name

                # Get all fields in this layer
                for field_name in layer.field_names:
                    try:
                        field = getattr(layer, field_name, None)
                        if field is None:
                            continue

                        # Try to get field position information
                        # pyshark sometimes provides offset info in the field
                        if hasattr(field, 'pos'):
                            start_pos = int(field.pos)
                            # Try to determine field size
                            if hasattr(field, 'size'):
                                field_size = int(field.size)
                            elif hasattr(field, 'raw_value') and field.raw_value:
                                # Calculate size from raw_value (hex string)
                                field_size = len(field.raw_value) // 2
                            else:
                                # Default to 1 byte if we can't determine size
                                field_size = 1

                            end_pos = start_pos + field_size
                            field_offsets.append((start_pos, end_pos, f"{layer_name}.{field_name}"))

                    except (AttributeError, ValueError, TypeError):
                        # Skip fields where we can't determine position
                        continue

        except Exception as e:
            logger.warning(f"Error extracting field offsets: {e}")

        # Sort by start position
        field_offsets.sort(key=lambda x: x[0])
        return field_offsets

    def _tokenize_packet_with_fields(self, packet) -> List[str]:
        """
        Tokenize a single packet using raw bytes and field boundary information.
        """
        try:
            # Get the raw packet bytes
            if hasattr(packet, 'get_raw_packet'):
                raw_bytes = packet.get_raw_packet()
            elif hasattr(packet, 'frame_raw'):
                # Convert hex string to bytes
                hex_string = packet.frame_raw.value.replace(':', '')
                raw_bytes = bytes.fromhex(hex_string)
            else:
                # Fallback: try to reconstruct from layers
                logger.warning("Could not get raw packet bytes, using fallback")
                return self._fallback_tokenize_packet(packet)

            # Get field boundary information
            field_offsets = self._get_field_offsets(packet)

            if not field_offsets:
                # No field information available, just tokenize raw bytes
                return [chr(b) for b in raw_bytes.decode('latin-1')]

            tokens = []
            last_end = 0

            for start, end, field_name in field_offsets:
                # Ensure we don't go beyond the packet bounds
                start = max(start, last_end)
                end = min(end, len(raw_bytes))

                if start >= len(raw_bytes):
                    break

                # Add any gap between fields as raw bytes
                if start > last_end:
                    gap_bytes = raw_bytes[last_end:start]
                    gap_string = gap_bytes.decode('latin-1')
                    tokens.extend([char for char in gap_string])

                # Add the field bytes
                if end > start:
                    field_bytes = raw_bytes[start:end]
                    field_string = field_bytes.decode('latin-1')
                    tokens.extend([char for char in field_string])

                    # Add field separator (except for the last field)
                    tokens.append(self.sep_token)

                last_end = end

            # Add any remaining bytes after the last field
            if last_end < len(raw_bytes):
                remaining_bytes = raw_bytes[last_end:]
                remaining_string = remaining_bytes.decode('latin-1')
                tokens.extend([char for char in remaining_string])

            # Remove trailing separator if present
            if tokens and tokens[-1] == self.sep_token:
                tokens.pop()

            return tokens

        except Exception as e:
            logger.warning(f"Error tokenizing packet with fields: {e}")
            return self._fallback_tokenize_packet(packet)

    def _fallback_tokenize_packet(self, packet) -> List[str]:
        """Fallback method when field extraction fails."""
        try:
            # Try to get raw bytes through different methods
            if hasattr(packet, 'get_raw_packet'):
                raw_bytes = packet.get_raw_packet()
            elif hasattr(packet, 'frame_raw'):
                hex_string = packet.frame_raw.value.replace(':', '')
                raw_bytes = bytes.fromhex(hex_string)
            else:
                # Last resort: empty packet
                return []

            # Just tokenize as raw bytes without field separation
            raw_string = raw_bytes.decode('latin-1')
            return [char for char in raw_string]

        except Exception as e:
            logger.error(f"Fallback tokenization failed: {e}")
            return []

    def _tokenize(self, pcap_path: str, **kwargs) -> List[str]:
        """Performs the main tokenization of a PCAP file."""
        try:
            # Open PCAP file with pyshark
            capture = pyshark.FileCapture(pcap_path)

            all_tokens = []
            pkt_sep_token_as_list = [self.pkt_sep_token]

            packet_count = 0
            for packet in capture:
                packet_count += 1

                try:
                    # Tokenize this packet with field awareness
                    packet_tokens = self._tokenize_packet_with_fields(packet)
                    all_tokens.extend(packet_tokens)

                    # Add packet separator (except after the last packet)
                    # We don't know if this is the last packet, so we'll add it
                    # and remove the final one at the end if needed
                    all_tokens.extend(pkt_sep_token_as_list)

                except Exception as e:
                    logger.warning(f"Error processing packet {packet_count}: {e}")
                    continue

            # Remove trailing packet separator if present
            if all_tokens and all_tokens[-1] == self.pkt_sep_token:
                all_tokens.pop()

            capture.close()
            logger.info(f"Processed {packet_count} packets into {len(all_tokens)} tokens")
            return all_tokens

        except Exception as e:
            logger.error(f"Failed to read or parse PCAP file at {pcap_path}: {e}")
            return []

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an ID using the vocab."""
        # Check if it's a special token first
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]

        # For regular byte tokens (single characters)
        if len(token) == 1:
            return ord(token) + self.offset

        # Unknown token
        return self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocab."""
        # Check if it's a special token first
        if index in self.added_tokens_decoder:
            return self.added_tokens_decoder[index].content

        # For regular byte tokens
        if self.offset <= index < (self.offset + self._byte_vocab_size):
            return chr(index - self.offset)

        # Unknown token
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

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens back to a string (for debugging purposes)."""
        # This is mainly for debugging - converts tokens back to byte string
        bstring = b""
        for token in tokens:
            if token in [self.sep_token, self.pkt_sep_token, self.pad_token, self.eos_token, self.unk_token]:
                # Skip special tokens when reconstructing byte string
                continue
            else:
                # Regular byte token - convert back to byte
                try:
                    bstring += bytes([ord(token)])
                except:
                    continue
        
        # Return as latin-1 string for debugging
        return bstring.decode('latin-1', errors='replace')

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple:
        """This tokenizer does not save a vocabulary file."""
        return ()