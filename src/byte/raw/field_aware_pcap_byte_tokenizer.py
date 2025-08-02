"""
A byte-level PCAP tokenizer with special tokens, packet separation, and protocol field separation.
"""

from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple, NamedTuple
import json
import datetime
from dataclasses import dataclass

import dpkt
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging as transformers_logging

logger = transformers_logging.get_logger(__name__)

# Special token definitions
_PAD_TOKEN = "<pad>"
_UNK_TOKEN = "<unk>"
_EOS_TOKEN = "<eos>"
_BOS_TOKEN = "<bos>"
_PKT_TOKEN = "<pkt>"
_SEP_TOKEN = "<sep>"

# Special token IDs (occupy the lowest values)
_PAD_ID = 0
_UNK_ID = 1
_EOS_ID = 2
_BOS_ID = 3
_PKT_ID = 4
_SEP_ID = 5

# Number of special tokens
_NUM_SPECIAL_TOKENS = 6

# Offset for byte values (they start after special tokens)
_BYTE_OFFSET = _NUM_SPECIAL_TOKENS

# Total vocabulary size (special tokens + 256 byte values)
_VOCAB_SIZE = _NUM_SPECIAL_TOKENS + 256


@dataclass
class SeparatorConfig:
    """Configuration for separator insertion behavior."""
    insert_ethernet_fields: bool = True
    insert_ip_fields: bool = True
    insert_transport_fields: bool = True
    max_depth: int = 4  # L2, L3, L4, Payload
    separate_payload: bool = True
    policy: str = "hybrid"  # "conservative", "aggressive", "hybrid"


class ProtocolOffset(NamedTuple):
    """Represents a protocol field boundary."""
    offset: int
    field_name: str
    protocol: str


class MalformedPacketError(Exception):
    """Exception for packets that cannot be parsed properly."""
    pass


class FieldAwarePCAPByteTokenizer(PreTrainedTokenizer):
    """
    A byte-level tokenizer for raw network packet data from PCAP files with special tokens
    and protocol field separation.

    This tokenizer treats each byte (0-255) as a distinct token, offset by the number of
    special tokens. It includes standard special tokens (PAD, UNK, EOS, BOS) plus
    <pkt> tokens to separate packets and <sep> tokens to separate protocol fields.
    Special tokens occupy the lowest token IDs (0-5), and byte values are mapped to IDs 6-261.

    Uses dpkt to analyze protocol structure and inserts <sep> tokens at field boundaries
    while preserving the exact raw bytes as captured on the wire.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self,
                 separator_config: Optional[SeparatorConfig] = None,
                 malformed_log_path: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initializes the TokenPCAPByteTokenizer with special tokens and separator configuration.

        Args:
            separator_config: Configuration for how to insert field separators
            malformed_log_path: Path to log malformed packets (default: tokenizer_malformed.log)
        """
        # Set special tokens
        kwargs.setdefault('pad_token', _PAD_TOKEN)
        kwargs.setdefault('unk_token', _UNK_TOKEN)
        kwargs.setdefault('eos_token', _EOS_TOKEN)
        kwargs.setdefault('bos_token', _BOS_TOKEN)

        # Add custom special tokens
        additional_special_tokens = kwargs.get('additional_special_tokens', [])
        for token in [_PKT_TOKEN, _SEP_TOKEN]:
            if token not in additional_special_tokens:
                additional_special_tokens.append(token)
        kwargs['additional_special_tokens'] = additional_special_tokens

        super().__init__(**kwargs)

        # Store special token IDs
        self._pad_id = _PAD_ID
        self._unk_id = _UNK_ID
        self._eos_id = _EOS_ID
        self._bos_id = _BOS_ID
        self._pkt_id = _PKT_ID
        self._sep_id = _SEP_ID

        # Configuration
        self.separator_config = separator_config or SeparatorConfig()
        self.malformed_log_path = malformed_log_path or "tokenizer_malformed.log"

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary (special tokens + 256 byte values)."""
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
        return self._pkt_id

    @property
    def sep_token_id(self) -> int:
        """Returns the ID of the field separator token."""
        return self._sep_id

    def get_vocab(self) -> Dict[str, int]:
        """Constructs the vocabulary dictionary including special tokens and byte tokens."""
        # Add special tokens
        vocab = {
            _PAD_TOKEN: _PAD_ID,
            _UNK_TOKEN: _UNK_ID,
            _EOS_TOKEN: _EOS_ID,
            _BOS_TOKEN: _BOS_ID,
            _PKT_TOKEN: _PKT_ID,
            _SEP_TOKEN: _SEP_ID
        }

        # Add byte tokens (with offset)
        for i in range(256):
            token = self._convert_id_to_token(i + _BYTE_OFFSET)
            vocab[token] = i + _BYTE_OFFSET

        return vocab

    def tokenize_pcap(self, pcap_path: Union[str, Path],
                      add_packet_separators: bool = True,
                      add_field_separators: bool = False,
                      add_bos: bool = False,
                      add_eos: bool = False) -> List[str]:
        """
        Public method to tokenize a PCAP file.

        Args:
            pcap_path: Path to the PCAP file to tokenize
            add_packet_separators: Whether to add <pkt> tokens between packets
            add_field_separators: Whether to add <sep> tokens between protocol fields
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token

        Returns:
            List of tokens representing the PCAP file
        """
        return self._tokenize(str(pcap_path),
                              add_packet_separators=add_packet_separators,
                              add_field_separators=add_field_separators,
                              add_bos=add_bos,
                              add_eos=add_eos)

    def read_pcap_packets(self, pcap_path: Union[str, Path]) -> List[bytes]:
        """Public method to read individual packets from a PCAP file."""
        return self._read_pcap_packets(pcap_path)

    def tokenize_pcap_to_ids(self, pcap_path: Union[str, Path],
                             add_packet_separators: bool = True,
                             add_field_separators: bool = False,
                             add_bos: bool = False,
                             add_eos: bool = False) -> List[int]:
        """Public method to tokenize a PCAP file directly to token IDs."""
        tokens = self.tokenize_pcap(pcap_path, add_packet_separators,
                                    add_field_separators, add_bos, add_eos)
        return [self._convert_token_to_id(token) for token in tokens]

    @staticmethod
    def _read_pcap_packets(pcap_path: Union[str, Path]) -> List[bytes]:
        """Reads a PCAP file and returns a list of individual packet bytes."""
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

    def _log_malformed_packet(self, pcap_filename: str, packet_index: int,
                              packet_bytes: bytes, error: Exception) -> None:
        """Log details about malformed packets for investigation."""
        error_info = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file': pcap_filename,
            'packet_index': packet_index,
            'packet_length': len(packet_bytes),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'first_bytes_hex': packet_bytes[:20].hex() if len(packet_bytes) >= 20 else packet_bytes.hex(),
            'packet_hex': packet_bytes.hex()
        }

        try:
            with open(self.malformed_log_path, 'a') as f:
                f.write(json.dumps(error_info) + '\n')
        except Exception as log_error:
            logger.warning(f"Failed to log malformed packet: {log_error}")

    def _calculate_ethernet_offsets(self, packet_bytes: bytes) -> List[ProtocolOffset]:
        """Calculate byte offsets for Ethernet fields."""
        offsets = []
        if len(packet_bytes) < 14:  # Minimum Ethernet frame size
            return offsets

        if self.separator_config.insert_ethernet_fields:
            # Ethernet: dst(6) + src(6) + type(2) = 14 bytes minimum
            offsets.extend([
                ProtocolOffset(6, "eth_src", "ethernet"),  # After dst MAC
                ProtocolOffset(12, "eth_type", "ethernet"),  # After src MAC
                ProtocolOffset(14, "eth_payload", "ethernet")  # After EtherType
            ])
        else:
            # Just mark end of Ethernet header
            offsets.append(ProtocolOffset(14, "eth_end", "ethernet"))

        return offsets

    def _calculate_ip_offsets(self, ip_packet: bytes, eth_offset: int = 14) -> List[ProtocolOffset]:
        """Calculate byte offsets for IP fields."""
        offsets = []

        if len(ip_packet) < eth_offset + 20:  # Minimum IP header
            return offsets

        # Check IP version
        ip_version = (ip_packet[eth_offset] >> 4) & 0xF

        if ip_version == 4:  # IPv4
            return self._calculate_ipv4_offsets(ip_packet, eth_offset)
        elif ip_version == 6:  # IPv6
            return self._calculate_ipv6_offsets(ip_packet, eth_offset)

        return offsets

    def _calculate_ipv4_offsets(self, packet_bytes: bytes, ip_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for IPv4 fields."""
        offsets = []

        if len(packet_bytes) < ip_start + 20:
            return offsets

        # Get IP header length (IHL field)
        ihl = packet_bytes[ip_start] & 0x0F
        ip_header_len = ihl * 4

        if len(packet_bytes) < ip_start + ip_header_len:
            return offsets

        if self.separator_config.insert_ip_fields:
            # IPv4 fields: ver+ihl(1) + tos(1) + len(2) + id(2) + flags+frag(2) + ttl(1) + proto(1) + csum(2) + src(4) + dst(4)
            base = ip_start
            offsets.extend([
                ProtocolOffset(base + 2, "ip_tos", "ipv4"),  # After version+IHL + ToS
                ProtocolOffset(base + 4, "ip_len", "ipv4"),  # After total length
                ProtocolOffset(base + 8, "ip_flags", "ipv4"),  # After ID + flags+fragment
                ProtocolOffset(base + 12, "ip_src", "ipv4"),  # After TTL + protocol + checksum
                ProtocolOffset(base + 16, "ip_dst", "ipv4"),  # After source IP
                ProtocolOffset(base + 20, "ip_options", "ipv4")  # After destination IP
            ])

            # Add options boundary if present
            if ip_header_len > 20:
                offsets.append(ProtocolOffset(ip_start + ip_header_len, "ip_payload", "ipv4"))
        else:
            # Just mark end of IP header
            offsets.append(ProtocolOffset(ip_start + ip_header_len, "ip_end", "ipv4"))

        return offsets

    def _calculate_ipv6_offsets(self, packet_bytes: bytes, ip_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for IPv6 fields."""
        offsets = []

        if len(packet_bytes) < ip_start + 40:  # IPv6 header is fixed 40 bytes
            return offsets

        if self.separator_config.insert_ip_fields:
            # IPv6 fields: ver+tc+fl(4) + len(2) + next(1) + hop(1) + src(16) + dst(16)
            base = ip_start
            offsets.extend([
                ProtocolOffset(base + 4, "ip6_len", "ipv6"),  # After version+traffic class+flow label
                ProtocolOffset(base + 6, "ip6_next", "ipv6"),  # After payload length
                ProtocolOffset(base + 8, "ip6_src", "ipv6"),  # After next header + hop limit
                ProtocolOffset(base + 24, "ip6_dst", "ipv6"),  # After source IP
                ProtocolOffset(base + 40, "ip6_payload", "ipv6")  # After destination IP
            ])
        else:
            offsets.append(ProtocolOffset(ip_start + 40, "ip6_end", "ipv6"))

        return offsets

    def _calculate_transport_offsets(self, packet_bytes: bytes, transport_start: int,
                                     protocol: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for transport layer fields."""
        offsets = []

        if protocol == 6:  # TCP
            return self._calculate_tcp_offsets(packet_bytes, transport_start)
        elif protocol == 17:  # UDP
            return self._calculate_udp_offsets(packet_bytes, transport_start)
        elif protocol == 1:  # ICMP
            return self._calculate_icmp_offsets(packet_bytes, transport_start)

        return offsets

    def _calculate_tcp_offsets(self, packet_bytes: bytes, tcp_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for TCP fields."""
        offsets = []

        if len(packet_bytes) < tcp_start + 20:  # Minimum TCP header
            return offsets

        # Get TCP header length from data offset field
        data_offset = (packet_bytes[tcp_start + 12] >> 4) & 0xF
        tcp_header_len = data_offset * 4

        if len(packet_bytes) < tcp_start + tcp_header_len:
            return offsets

        if self.separator_config.insert_transport_fields:
            # TCP fields: src(2) + dst(2) + seq(4) + ack(4) + hlen +flags(2) + win(2) + csum(2) + urg(2)
            base = tcp_start
            offsets.extend([
                ProtocolOffset(base + 2, "tcp_dst", "tcp"),  # After source port
                ProtocolOffset(base + 4, "tcp_seq", "tcp"),  # After destination port
                ProtocolOffset(base + 8, "tcp_ack", "tcp"),  # After sequence number
                ProtocolOffset(base + 12, "tcp_flags", "tcp"),  # After acknowledgment
                ProtocolOffset(base + 14, "tcp_win", "tcp"),  # After header length + flags
                ProtocolOffset(base + 18, "tcp_urg", "tcp"),  # After window + checksum
                ProtocolOffset(base + 20, "tcp_options", "tcp")  # After urgent pointer
            ])

            # Add options boundary if present
            if tcp_header_len > 20:
                offsets.append(ProtocolOffset(tcp_start + tcp_header_len, "tcp_payload", "tcp"))
        else:
            offsets.append(ProtocolOffset(tcp_start + tcp_header_len, "tcp_end", "tcp"))

        return offsets

    def _calculate_udp_offsets(self, packet_bytes: bytes, udp_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for UDP fields."""
        offsets = []

        if len(packet_bytes) < udp_start + 8:  # UDP header is fixed 8 bytes
            return offsets

        if self.separator_config.insert_transport_fields:
            # UDP fields: src(2) + dst(2) + len(2) + csum(2)
            base = udp_start
            offsets.extend([
                ProtocolOffset(base + 2, "udp_dst", "udp"),  # After source port
                ProtocolOffset(base + 4, "udp_len", "udp"),  # After destination port
                ProtocolOffset(base + 6, "udp_csum", "udp"),  # After length
                ProtocolOffset(base + 8, "udp_payload", "udp")  # After checksum
            ])
        else:
            offsets.append(ProtocolOffset(udp_start + 8, "udp_end", "udp"))

        return offsets

    def _calculate_icmp_offsets(self, packet_bytes: bytes, icmp_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for ICMP fields."""
        offsets = []

        if len(packet_bytes) < icmp_start + 8:  # Minimum ICMP header
            return offsets

        if self.separator_config.insert_transport_fields:
            # ICMP fields: type(1) + code(1) + csum(2) + data(4+)
            base = icmp_start
            offsets.extend([
                ProtocolOffset(base + 1, "icmp_code", "icmp"),  # After type
                ProtocolOffset(base + 2, "icmp_csum", "icmp"),  # After code
                ProtocolOffset(base + 4, "icmp_data", "icmp"),  # After checksum
                ProtocolOffset(base + 8, "icmp_payload", "icmp")  # After header data
            ])
        else:
            offsets.append(ProtocolOffset(icmp_start + 8, "icmp_end", "icmp"))

        return offsets

    def _calculate_protocol_offsets(self, packet_bytes: bytes, pcap_filename: str,
                                    packet_index: int) -> List[ProtocolOffset]:
        """
        Calculate all protocol field offsets for a packet using hybrid parsing approach.

        Args:
            packet_bytes: Raw packet bytes
            pcap_filename: Name of PCAP file (for error logging)
            packet_index: Index of packet in file (for error logging)

        Returns:
            List of ProtocolOffset objects sorted by byte position
        """
        offsets = []

        try:
            # Parse with dpkt to get protocol information
            eth = dpkt.ethernet.Ethernet(packet_bytes)

            # Layer 2 - Ethernet
            if self.separator_config.max_depth >= 2:
                eth_offsets = self._calculate_ethernet_offsets(packet_bytes)
                offsets.extend(eth_offsets)

            # Layer 3 - IP
            if self.separator_config.max_depth >= 3 and isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                ip_offsets = self._calculate_ip_offsets(packet_bytes)
                offsets.extend(ip_offsets)

                # Layer 4 - Transport
                if self.separator_config.max_depth >= 4:
                    if isinstance(eth.data, dpkt.ip.IP):
                        protocol = eth.data.p
                        ip_header_len = (packet_bytes[14] & 0x0F) * 4
                        transport_start = 14 + ip_header_len
                    elif isinstance(eth.data, dpkt.ip6.IP6):
                        protocol = eth.data.nxt
                        transport_start = 14 + 40  # IPv6 header is fixed 40 bytes
                    else:
                        protocol = None
                        transport_start = None

                    if protocol is not None and transport_start is not None:
                        transport_offsets = self._calculate_transport_offsets(
                            packet_bytes, transport_start, protocol)
                        offsets.extend(transport_offsets)

        except Exception as e:
            # Handle malformed packets based on policy
            if self.separator_config.policy == "conservative":
                # Don't add any separators for malformed packets
                self._log_malformed_packet(pcap_filename, packet_index, packet_bytes, e)
                return []
            elif self.separator_config.policy == "aggressive":
                # Try to extract what we can, even if parsing fails
                try:
                    basic_offsets = self._calculate_ethernet_offsets(packet_bytes)
                    offsets.extend(basic_offsets)
                except:
                    pass
                self._log_malformed_packet(pcap_filename, packet_index, packet_bytes, e)
            else:  # hybrid
                # Insert basic separators if we can determine them
                if len(packet_bytes) >= 14:  # Has Ethernet header
                    try:
                        eth_offsets = self._calculate_ethernet_offsets(packet_bytes)
                        offsets.extend(eth_offsets)
                    except:
                        pass
                self._log_malformed_packet(pcap_filename, packet_index, packet_bytes, e)

        # Sort offsets by position and remove duplicates
        offsets = sorted(set(offsets), key=lambda x: x.offset)

        # Ensure offsets are within packet bounds
        valid_offsets = [offset for offset in offsets if offset.offset <= len(packet_bytes)]

        return valid_offsets

    def _tokenize_packet_with_separators(self, packet_bytes: bytes, pcap_filename: str,
                                         packet_index: int) -> List[str]:
        """
        Tokenize a single packet with field separators inserted at appropriate positions.

        Args:
            packet_bytes: Raw packet bytes
            pcap_filename: Name of PCAP file (for error logging)
            packet_index: Index of packet in file (for error logging)

        Returns:
            List of tokens including byte tokens and separator tokens
        """
        # Calculate where to insert separators
        offsets = self._calculate_protocol_offsets(packet_bytes, pcap_filename, packet_index)

        if not offsets:
            # No separators to insert, return raw bytes
            return [chr(b) for b in packet_bytes]

        # Insert separators at calculated positions
        tokens = []
        last_pos = 0

        for offset_info in offsets:
            offset = offset_info.offset

            # Add bytes up to this boundary
            if last_pos < offset <= len(packet_bytes):
                byte_tokens = [chr(b) for b in packet_bytes[last_pos:offset]]
                tokens.extend(byte_tokens)

                # Add separator (except at the very end of packet)
                if offset < len(packet_bytes):
                    tokens.append(_SEP_TOKEN)

                last_pos = offset

        # Add any remaining bytes after the last separator
        if last_pos < len(packet_bytes):
            remaining_tokens = [chr(b) for b in packet_bytes[last_pos:]]
            tokens.extend(remaining_tokens)

        return tokens

    def _tokenize(self, text: str,
                  add_packet_separators: bool = True,
                  add_field_separators: bool = False,
                  add_bos: bool = False,
                  add_eos: bool = False,
                  **kwargs) -> List[str]:
        """
        Tokenizes the content of a PCAP file specified by its path.

        Args:
            text: The file path to the PCAP file to tokenize.
            add_packet_separators: Whether to add <pkt> tokens between packets
            add_field_separators: Whether to add <sep> tokens between protocol fields
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

            # Tokenize packet with or without field separators
            if add_field_separators:
                packet_tokens = self._tokenize_packet_with_separators(
                    packet_bytes, pcap_file_path, i)
            else:
                # Convert each byte to its character representation
                packet_tokens = [chr(b) for b in packet_bytes]

            tokens.extend(packet_tokens)

        # Add end-of-sequence token if requested
        if add_eos:
            tokens.append(_EOS_TOKEN)

        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token to its ID, handling special tokens and byte tokens with offset."""
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
        elif token == _SEP_TOKEN:
            return _SEP_ID

        # Handle byte tokens
        if len(token) == 1:
            byte_value = ord(token)
            if 0 <= byte_value <= 255:
                return byte_value + _BYTE_OFFSET

        # Return UNK token ID for unknown tokens
        return self._unk_id

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an ID to its token, handling special tokens and byte tokens with offset."""
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
        elif index == _SEP_ID:
            return _SEP_TOKEN

        # Handle byte tokens (subtract offset)
        byte_value = index - _BYTE_OFFSET
        if 0 <= byte_value <= 255:
            return chr(byte_value)

        # This should not happen if index is in valid range
        return _UNK_TOKEN

    @staticmethod
    def convert_tokens_to_string(tokens: List[str], **kwargs) -> str:
        """Converts a sequence of tokens back into a string, filtering out special tokens."""
        # Filter out special tokens and convert byte tokens to string
        byte_tokens = [t for t in tokens if t not in {_PAD_TOKEN, _UNK_TOKEN, _EOS_TOKEN,
                                                      _BOS_TOKEN, _PKT_TOKEN, _SEP_TOKEN}]
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
        """Build model inputs by adding BOS and EOS tokens if configured."""
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
        """Returns a mask indicating which tokens are special tokens."""
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
        """Save the vocabulary to a file."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        filename = "vocab.json"
        if filename_prefix is not None:
            filename = f"{filename_prefix}-{filename}"

        vocab_file = save_directory / filename

        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_vocab(), f, ensure_ascii=False, indent=2)

        return (str(vocab_file),)