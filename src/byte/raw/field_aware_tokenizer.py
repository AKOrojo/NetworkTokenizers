"""
A field-aware and packet-aware byte-level PCAP tokenizer.

This tokenizer processes PCAP files by treating each byte as a token, but it
also inserts special tokens to mark boundaries:
- <pkt>: Separates individual packets.
- <sep>: Separates protocol fields within a packet.

It also includes standard special tokens (<pad>, </s>, <unk>) for compatibility
with transformer models.

Based on ByT5 tokenizer design but for raw PCAP bytes instead of text bytes.
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

        # Store these before calling super().__init__
        self.sep_token = str(sep_token)
        self.pkt_sep_token = str(pkt_sep_token)
        
        # Similar to ByT5: we have special tokens first, then byte tokens
        # ByT5 uses: pad_token=0, eos_token=1, unk_token=2, then bytes start at offset=3
        # We have: pad, eos, unk, sep, pkt = 5 special tokens
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

    @staticmethod
    def _get_field_bytes(layer: Packet, field, debug=False) -> bytes:
        """Extracts the raw byte representation of a single field."""
        if debug:
            print(f"    Processing field: {field.name}")
            
        try:
            field_val = layer.getfieldval(field.name)
            if debug:
                print(f"      Field value: {field_val} (type: {type(field_val)})")
            
            # Method 1: Try the standard field.addfield approach first
            try:
                result = field.addfield(None, b"", field_val)
                if isinstance(result, bytes) and result:
                    if debug:
                        print(f"      ✅ Method 1 success: {len(result)} bytes")
                    return result
            except Exception as e:
                if debug:
                    print(f"      ❌ Method 1 failed: {e}")
                pass  # Try fallback methods
            
            # Method 2: Handle different field value types directly
            if isinstance(field_val, bytes):
                if debug:
                    print(f"      ✅ Method 2 (bytes): {len(field_val)} bytes")
                return field_val
            elif isinstance(field_val, str):
                result = field_val.encode('latin-1', errors='replace')
                if debug:
                    print(f"      ✅ Method 2 (string): {len(result)} bytes")
                return result
            elif isinstance(field_val, int):
                # Convert integer to bytes (handle different sizes)
                try:
                    # Try to determine field size from the field definition
                    if hasattr(field, 'sz') and field.sz:
                        byte_length = field.sz
                    elif hasattr(field, 'fmt') and field.fmt:
                        # Common struct format sizes
                        fmt_sizes = {'B': 1, 'H': 2, 'I': 4, 'Q': 8, 'b': 1, 'h': 2, 'i': 4, 'q': 8}
                        byte_length = fmt_sizes.get(field.fmt.replace('!', '').replace('<', '').replace('>', ''), 4)
                    else:
                        # Default: use minimum bytes needed for the value
                        byte_length = max(1, (field_val.bit_length() + 7) // 8) if field_val > 0 else 1
                    
                    result = field_val.to_bytes(byte_length, byteorder='big', signed=False)
                    if debug:
                        print(f"      ✅ Method 2 (int): {len(result)} bytes")
                    return result
                except (OverflowError, ValueError):
                    # Fallback for very large integers or negative values
                    result = str(field_val).encode('latin-1', errors='replace')
                    if debug:
                        print(f"      ✅ Method 2 (int->string): {len(result)} bytes")
                    return result
            elif field_val is None:
                # For None values, try to get default or compute field size
                try:
                    # Try to get the field's default bytes representation
                    default_val = field.default if hasattr(field, 'default') else 0
                    if isinstance(default_val, int):
                        byte_length = getattr(field, 'sz', 1) or 1
                        result = default_val.to_bytes(byte_length, byteorder='big', signed=False)
                        if debug:
                            print(f"      ✅ Method 2 (None->default): {len(result)} bytes")
                        return result
                    else:
                        if debug:
                            print(f"      ✅ Method 2 (None): 1 null byte")
                        return b'\x00'  # Single null byte for None fields
                except Exception:
                    if debug:
                        print(f"      ✅ Method 2 (None fallback): 1 null byte")
                    return b'\x00'
            else:
                # Method 3: Try to serialize other field types
                try:
                    # Convert to string and encode
                    result = str(field_val).encode('latin-1', errors='replace')
                    if debug:
                        print(f"      ✅ Method 3 (other->string): {len(result)} bytes")
                    return result
                except Exception:
                    if debug:
                        print(f"      ✅ Method 3 (fallback): 1 null byte")
                    return b'\x00'
                    
        except Exception as e:
            # Method 4: Last resort - try to extract from raw packet bytes
            try:
                # Get the field's position in the packet if possible
                if hasattr(layer, 'fields') and field.name in layer.fields:
                    # Try to get some bytes for this field, even if it's just a placeholder
                    if debug:
                        print(f"      ✅ Method 4 (placeholder): 1 null byte - {e}")
                    return b'\x00'  # At least mark that this field exists
                else:
                    if debug:
                        print(f"      ⚠️  Field not present: 0 bytes")
                    return b''  # Field not present
            except Exception:
                if debug:
                    print(f"      ✅ Complete fallback: 1 null byte - {e}")
                return b'\x00'  # Always return something to maintain field structure

    def _tokenize_packet_fields(self, packet: Packet) -> List[bytes]:
        """Tokenizes a single packet into a list of its field bytes."""
        field_byte_list = []
        current_layer = packet

        while current_layer:
            try:
                for field in current_layer.fields_desc:
                    if field.name in current_layer.fields:
                        field_bytes = self._get_field_bytes(current_layer, field)
                        # Always include the field, even if it's just a null byte
                        # This preserves the field structure for learning
                        field_byte_list.append(field_bytes)

                if isinstance(current_layer.payload, Packet):
                    current_layer = current_layer.payload
                elif isinstance(current_layer.payload, Raw):
                    payload_bytes = bytes(current_layer.payload)
                    if payload_bytes:  # Only add non-empty payload
                        field_byte_list.append(payload_bytes)
                    current_layer = None
                else:
                    current_layer = None
            except Exception as e:
                logger.warning(f"Error processing layer {current_layer.__class__.__name__}: {e}")
                # Try to get raw bytes from the layer if field processing fails
                try:
                    layer_bytes = bytes(current_layer)
                    if layer_bytes:
                        field_byte_list.append(layer_bytes)
                except Exception:
                    # If we can't get any bytes, add a placeholder to maintain structure
                    field_byte_list.append(b'\x00')
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
            try:
                field_bytes_list = self._tokenize_packet_fields(packet)
                num_fields = len(field_bytes_list)

                for j, field_bytes in enumerate(field_bytes_list):
                    try:
                        # Convert bytes to tokens using latin-1 (like ByT5 but for raw bytes)
                        # This ensures every byte value 0-255 maps to a valid character
                        field_string = field_bytes.decode('latin-1')
                        field_char_tokens = [char for char in field_string]
                        all_tokens.extend(field_char_tokens)

                        if j < num_fields - 1:
                            all_tokens.extend(sep_token_as_list)
                    except Exception as e:
                        logger.warning(f"Error processing field bytes in packet {i}, field {j}: {e}")
                        continue

                if i < num_packets - 1:
                    all_tokens.extend(pkt_sep_token_as_list)
            except Exception as e:
                logger.warning(f"Error processing packet {i}: {e}")
                continue

        return all_tokens

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
            if token in self.added_tokens_decoder.values():
                # Skip special tokens when reconstructing byte string
                continue
            elif token in [self.sep_token, self.pkt_sep_token, self.pad_token, self.eos_token, self.unk_token]:
                # Skip our known special tokens
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