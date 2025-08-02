"""
A byte-level PCAP tokenizer with special tokens, packet separation, and protocol field separation
supporting up to Layer 5+ (Application Layer) protocols.
"""

import datetime
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple, NamedTuple

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
    insert_application_fields: bool = True  # New: Layer 5+ support
    max_depth: int = 5  # L2, L3, L4, L5, Payload
    separate_payload: bool = True
    policy: str = "hybrid"  # "conservative", "aggressive", "hybrid"

    # Application layer specific settings
    parse_http_headers: bool = True
    parse_dns_sections: bool = True
    parse_tls_records: bool = True
    parse_dhcp_options: bool = True
    detailed_app_parsing: bool = False  # Enable detailed field-level parsing


class ProtocolOffset(NamedTuple):
    """Represents a protocol field boundary."""
    offset: int
    field_name: str
    protocol: str
    layer: int = 4  # Default to Layer 4


class MalformedPacketError(Exception):
    """Exception for packets that cannot be parsed properly."""
    pass


class ApplicationProtocolDetector:
    """Detects application layer protocols based on ports and content analysis."""

    # Well-known port mappings
    PORT_PROTOCOLS = {
        # Web protocols
        80: "http", 8080: "http", 8000: "http", 3000: "http",
        443: "https", 8443: "https",

        # Network services
        53: "dns",
        67: "dhcp-server", 68: "dhcp-client",
        123: "ntp",
        1812: "radius", 1813: "radius",
        3478: "stun", 3479: "stun",

        # File transfer & communication
        69: "tftp",
        23: "telnet",
        5060: "sip", 5061: "sips",

        # Network management
        111: "rpc",
        135: "rpc", 445: "smb", 139: "netbios",

        # Messaging & chat
        5190: "aim", 1863: "aim",
        4000: "qq", 5050: "yahoo",

        # Other protocols
        5900: "rfb",  # VNC
        1720: "h225",
        1521: "tns",  # Oracle
        102: "tpkt",

        # Additional common ports
        21: "ftp", 22: "ssh", 25: "smtp",
        110: "pop3", 143: "imap", 993: "imaps", 995: "pop3s",
        161: "snmp", 162: "snmp-trap",
        194: "irc", 6667: "irc",
        514: "syslog",
    }

    @classmethod
    def detect_protocol(cls, src_port: int, dst_port: int, payload: bytes) -> Optional[str]:
        """Detect application protocol based on ports and payload analysis."""
        # Check well-known ports
        if src_port in cls.PORT_PROTOCOLS:
            return cls.PORT_PROTOCOLS[src_port]
        if dst_port in cls.PORT_PROTOCOLS:
            return cls.PORT_PROTOCOLS[dst_port]

        # Content-based detection for common protocols
        if len(payload) > 0:
            return cls._detect_by_content(payload)

        return None

    @classmethod
    def _detect_by_content(cls, payload: bytes) -> Optional[str]:
        """Detect protocol by analyzing payload content."""
        if len(payload) < 4:
            return None

        # HTTP detection
        http_methods = [b'GET ', b'POST', b'PUT ', b'HEAD', b'DELE', b'OPTI', b'TRAC', b'CONN']
        http_responses = [b'HTTP/1.0', b'HTTP/1.1', b'HTTP/2.0']

        for method in http_methods:
            if payload.startswith(method):
                return "http"
        for response in http_responses:
            if payload.startswith(response):
                return "http"

        # TLS/SSL detection (TLS handshake)
        if len(payload) >= 5:
            if payload[0] == 0x16 and payload[1:3] in [b'\x03\x00', b'\x03\x01', b'\x03\x02', b'\x03\x03']:
                return "tls"

        # DNS detection (simplified)
        if len(payload) >= 12:
            # Check if it looks like a DNS packet (basic heuristic)
            try:
                # DNS header: ID(2) + Flags(2) + QDCOUNT(2) + ANCOUNT(2) + NSCOUNT(2) + ARCOUNT(2)
                flags = struct.unpack('!H', payload[2:4])[0]
                qr = (flags >> 15) & 1  # Query/Response bit
                if qr in [0, 1]:  # Valid QR bit
                    return "dns"
            except:
                pass

        # DHCP detection
        if len(payload) >= 240:  # Minimum DHCP packet size
            if payload[0] in [1, 2]:  # BOOTREQUEST or BOOTREPLY
                magic_cookie = payload[236:240]
                if magic_cookie == b'\x63\x82\x53\x63':  # DHCP magic cookie
                    return "dhcp"

        return None


class FieldAwarePCAPByteTokenizer(PreTrainedTokenizer):
    """
    Enhanced byte-level tokenizer for raw network packet data from PCAP files with special tokens
    and protocol field separation supporting up to Layer 5+ (Application Layer).
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self,
                 separator_config: Optional[SeparatorConfig] = None,
                 malformed_log_path: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initializes the Enhanced FieldAwarePCAPByteTokenizer with extended protocol support.

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
                ProtocolOffset(6, "eth_src", "ethernet", 2),  # After dst MAC
                ProtocolOffset(12, "eth_type", "ethernet", 2),  # After src MAC
                ProtocolOffset(14, "eth_payload", "ethernet", 2)  # After EtherType
            ])
        else:
            # Just mark end of Ethernet header
            offsets.append(ProtocolOffset(14, "eth_end", "ethernet", 2))

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
                ProtocolOffset(base + 2, "ip_tos", "ipv4", 3),  # After version+IHL + ToS
                ProtocolOffset(base + 4, "ip_len", "ipv4", 3),  # After total length
                ProtocolOffset(base + 8, "ip_flags", "ipv4", 3),  # After ID + flags+fragment
                ProtocolOffset(base + 12, "ip_src", "ipv4", 3),  # After TTL + protocol + checksum
                ProtocolOffset(base + 16, "ip_dst", "ipv4", 3),  # After source IP
                ProtocolOffset(base + 20, "ip_options", "ipv4", 3)  # After destination IP
            ])

            # Add options boundary if present
            if ip_header_len > 20:
                offsets.append(ProtocolOffset(ip_start + ip_header_len, "ip_payload", "ipv4", 3))
        else:
            # Just mark end of IP header
            offsets.append(ProtocolOffset(ip_start + ip_header_len, "ip_end", "ipv4", 3))

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
                ProtocolOffset(base + 4, "ip6_len", "ipv6", 3),  # After version+traffic class+flow label
                ProtocolOffset(base + 6, "ip6_next", "ipv6", 3),  # After payload length
                ProtocolOffset(base + 8, "ip6_src", "ipv6", 3),  # After next header + hop limit
                ProtocolOffset(base + 24, "ip6_dst", "ipv6", 3),  # After source IP
                ProtocolOffset(base + 40, "ip6_payload", "ipv6", 3)  # After destination IP
            ])
        else:
            offsets.append(ProtocolOffset(ip_start + 40, "ip6_end", "ipv6", 3))

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
                ProtocolOffset(base + 2, "tcp_dst", "tcp", 4),  # After source port
                ProtocolOffset(base + 4, "tcp_seq", "tcp", 4),  # After destination port
                ProtocolOffset(base + 8, "tcp_ack", "tcp", 4),  # After sequence number
                ProtocolOffset(base + 12, "tcp_flags", "tcp", 4),  # After acknowledgment
                ProtocolOffset(base + 14, "tcp_win", "tcp", 4),  # After header length + flags
                ProtocolOffset(base + 18, "tcp_urg", "tcp", 4),  # After window + checksum
                ProtocolOffset(base + 20, "tcp_options", "tcp", 4)  # After urgent pointer
            ])

            # Add options boundary if present
            if tcp_header_len > 20:
                offsets.append(ProtocolOffset(tcp_start + tcp_header_len, "tcp_payload", "tcp", 4))
        else:
            offsets.append(ProtocolOffset(tcp_start + tcp_header_len, "tcp_end", "tcp", 4))

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
                ProtocolOffset(base + 2, "udp_dst", "udp", 4),  # After source port
                ProtocolOffset(base + 4, "udp_len", "udp", 4),  # After destination port
                ProtocolOffset(base + 6, "udp_csum", "udp", 4),  # After length
                ProtocolOffset(base + 8, "udp_payload", "udp", 4)  # After checksum
            ])
        else:
            offsets.append(ProtocolOffset(udp_start + 8, "udp_end", "udp", 4))

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
                ProtocolOffset(base + 1, "icmp_code", "icmp", 4),  # After type
                ProtocolOffset(base + 2, "icmp_csum", "icmp", 4),  # After code
                ProtocolOffset(base + 4, "icmp_data", "icmp", 4),  # After checksum
                ProtocolOffset(base + 8, "icmp_payload", "icmp", 4)  # After header data
            ])
        else:
            offsets.append(ProtocolOffset(icmp_start + 8, "icmp_end", "icmp", 4))

        return offsets

    # ================================
    # APPLICATION LAYER PARSING (Layer 5+)
    # ================================

    def _calculate_application_offsets(self, packet_bytes: bytes, app_start: int,
                                       src_port: int, dst_port: int,
                                       transport_protocol: str) -> List[ProtocolOffset]:
        """Calculate byte offsets for application layer protocols."""
        if not self.separator_config.insert_application_fields:
            return []

        offsets = []

        # Get payload for analysis
        payload = packet_bytes[app_start:] if app_start < len(packet_bytes) else b''
        if len(payload) == 0:
            return offsets

        # Detect application protocol
        app_protocol = ApplicationProtocolDetector.detect_protocol(src_port, dst_port, payload)

        if app_protocol:
            # Route to specific protocol parser
            if app_protocol in ["http", "https"]:
                offsets.extend(self._calculate_http_offsets(packet_bytes, app_start, app_protocol))
            elif app_protocol == "dns":
                offsets.extend(self._calculate_dns_offsets(packet_bytes, app_start))
            elif app_protocol in ["dhcp-server", "dhcp-client", "dhcp"]:
                offsets.extend(self._calculate_dhcp_offsets(packet_bytes, app_start))
            elif app_protocol in ["tls", "ssl"]:
                offsets.extend(self._calculate_tls_offsets(packet_bytes, app_start))
            elif app_protocol == "ntp":
                offsets.extend(self._calculate_ntp_offsets(packet_bytes, app_start))
            elif app_protocol in ["sip", "sips"]:
                offsets.extend(self._calculate_sip_offsets(packet_bytes, app_start))
            elif app_protocol == "rtp":
                offsets.extend(self._calculate_rtp_offsets(packet_bytes, app_start))
            elif app_protocol == "tftp":
                offsets.extend(self._calculate_tftp_offsets(packet_bytes, app_start))
            elif app_protocol == "telnet":
                offsets.extend(self._calculate_telnet_offsets(packet_bytes, app_start))
            elif app_protocol == "smb":
                offsets.extend(self._calculate_smb_offsets(packet_bytes, app_start))
            elif app_protocol == "netbios":
                offsets.extend(self._calculate_netbios_offsets(packet_bytes, app_start))
            elif app_protocol == "radius":
                offsets.extend(self._calculate_radius_offsets(packet_bytes, app_start))
            elif app_protocol == "stun":
                offsets.extend(self._calculate_stun_offsets(packet_bytes, app_start))
            elif app_protocol in ["rpc"]:
                offsets.extend(self._calculate_rpc_offsets(packet_bytes, app_start))
            elif app_protocol == "h225":
                offsets.extend(self._calculate_h225_offsets(packet_bytes, app_start))
            elif app_protocol == "rfb":
                offsets.extend(self._calculate_rfb_offsets(packet_bytes, app_start))
            # Add more protocol parsers as needed

        return offsets

    def _calculate_http_offsets(self, packet_bytes: bytes, http_start: int, protocol: str) -> List[ProtocolOffset]:
        """Calculate byte offsets for HTTP fields."""
        offsets = []

        try:
            payload = packet_bytes[http_start:]
            if len(payload) < 4:
                return offsets

            # Find header/body boundary (double CRLF)
            header_end = payload.find(b'\r\n\r\n')
            if header_end == -1:
                header_end = payload.find(b'\n\n')

            if header_end != -1:
                if self.separator_config.parse_http_headers and self.separator_config.detailed_app_parsing:
                    # Parse individual headers
                    header_data = payload[:header_end]
                    lines = header_data.split(b'\r\n') if b'\r\n' in header_data else header_data.split(b'\n')

                    current_offset = http_start
                    for i, line in enumerate(lines):
                        if i == 0:  # Request/Response line
                            current_offset += len(line) + (2 if b'\r\n' in header_data else 1)
                            offsets.append(ProtocolOffset(current_offset, "http_request_line", protocol, 5))
                        elif line.strip():  # Header line
                            current_offset += len(line) + (2 if b'\r\n' in header_data else 1)
                            if b':' in line:
                                offsets.append(ProtocolOffset(current_offset, "http_header", protocol, 5))

                # Header/body boundary
                body_start = http_start + header_end + (4 if b'\r\n\r\n' in payload else 2)
                offsets.append(ProtocolOffset(body_start, "http_body", protocol, 5))
            else:
                # No body separator found, might be headers only
                offsets.append(ProtocolOffset(http_start + len(payload), "http_end", protocol, 5))

        except Exception as e:
            logger.debug(f"Error parsing HTTP: {e}")

        return offsets

    def _calculate_dns_offsets(self, packet_bytes: bytes, dns_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for DNS fields."""
        offsets = []

        try:
            payload = packet_bytes[dns_start:]
            if len(payload) < 12:  # Minimum DNS header
                return offsets

            # DNS header is fixed 12 bytes
            base = dns_start
            if self.separator_config.parse_dns_sections:
                offsets.extend([
                    ProtocolOffset(base + 2, "dns_flags", "dns", 5),     # After ID
                    ProtocolOffset(base + 4, "dns_qdcount", "dns", 5),  # After flags
                    ProtocolOffset(base + 6, "dns_ancount", "dns", 5),  # After QDCOUNT
                    ProtocolOffset(base + 8, "dns_nscount", "dns", 5),  # After ANCOUNT
                    ProtocolOffset(base + 10, "dns_arcount", "dns", 5), # After NSCOUNT
                    ProtocolOffset(base + 12, "dns_questions", "dns", 5) # After header
                ])

                # Try to parse questions and answers with dpkt
                try:
                    dns_packet = dpkt.dns.DNS(payload)
                    current_offset = dns_start + 12

                    # Questions section
                    if hasattr(dns_packet, 'qd') and dns_packet.qd:
                        for q in dns_packet.qd:
                            # Estimate question size (name + type + class)
                            q_size = len(q.name) + 4  # Simplified
                            current_offset += q_size
                            offsets.append(ProtocolOffset(current_offset, "dns_question", "dns", 5))

                    # Answers section
                    if hasattr(dns_packet, 'an') and dns_packet.an:
                        offsets.append(ProtocolOffset(current_offset, "dns_answers", "dns", 5))

                except Exception:
                    # Fallback to basic parsing
                    pass
            else:
                offsets.append(ProtocolOffset(base + 12, "dns_data", "dns", 5))

        except Exception as e:
            logger.debug(f"Error parsing DNS: {e}")

        return offsets

    def _calculate_dhcp_offsets(self, packet_bytes: bytes, dhcp_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for DHCP fields."""
        offsets = []

        try:
            payload = packet_bytes[dhcp_start:]
            if len(payload) < 240:  # Minimum DHCP packet size
                return offsets

            base = dhcp_start
            if self.separator_config.parse_dhcp_options:
                # DHCP header fields
                offsets.extend([
                    ProtocolOffset(base + 1, "dhcp_htype", "dhcp", 5),    # After op
                    ProtocolOffset(base + 2, "dhcp_hlen", "dhcp", 5),     # After htype
                    ProtocolOffset(base + 3, "dhcp_hops", "dhcp", 5),     # After hlen
                    ProtocolOffset(base + 4, "dhcp_xid", "dhcp", 5),      # After hops
                    ProtocolOffset(base + 8, "dhcp_secs", "dhcp", 5),     # After xid
                    ProtocolOffset(base + 10, "dhcp_flags", "dhcp", 5),   # After secs
                    ProtocolOffset(base + 12, "dhcp_ciaddr", "dhcp", 5),  # After flags
                    ProtocolOffset(base + 16, "dhcp_yiaddr", "dhcp", 5),  # After ciaddr
                    ProtocolOffset(base + 20, "dhcp_siaddr", "dhcp", 5),  # After yiaddr
                    ProtocolOffset(base + 24, "dhcp_giaddr", "dhcp", 5),  # After siaddr
                    ProtocolOffset(base + 28, "dhcp_chaddr", "dhcp", 5),  # After giaddr
                    ProtocolOffset(base + 44, "dhcp_sname", "dhcp", 5),   # After chaddr
                    ProtocolOffset(base + 108, "dhcp_file", "dhcp", 5),   # After sname
                    ProtocolOffset(base + 236, "dhcp_magic", "dhcp", 5),  # After file
                    ProtocolOffset(base + 240, "dhcp_options", "dhcp", 5) # After magic cookie
                ])
            else:
                offsets.append(ProtocolOffset(base + 240, "dhcp_options", "dhcp", 5))

        except Exception as e:
            logger.debug(f"Error parsing DHCP: {e}")

        return offsets

    def _calculate_tls_offsets(self, packet_bytes: bytes, tls_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for TLS/SSL fields."""
        offsets = []

        try:
            payload = packet_bytes[tls_start:]
            if len(payload) < 5:  # Minimum TLS record header
                return offsets

            if self.separator_config.parse_tls_records:
                # Parse TLS records
                current_offset = tls_start
                remaining = payload

                while len(remaining) >= 5:
                    # TLS record: type(1) + version(2) + length(2) + data
                    record_type = remaining[0]
                    record_length = struct.unpack('!H', remaining[3:5])[0]

                    if len(remaining) < 5 + record_length:
                        break

                    # Add separator after record header
                    offsets.append(ProtocolOffset(current_offset + 5, "tls_record_data", "tls", 5))

                    # Move to next record
                    record_size = 5 + record_length
                    current_offset += record_size
                    remaining = remaining[record_size:]

                    if current_offset >= len(packet_bytes):
                        break

                    offsets.append(ProtocolOffset(current_offset, "tls_record", "tls", 5))
            else:
                offsets.append(ProtocolOffset(tls_start + 5, "tls_data", "tls", 5))

        except Exception as e:
            logger.debug(f"Error parsing TLS: {e}")

        return offsets

    def _calculate_ntp_offsets(self, packet_bytes: bytes, ntp_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for NTP fields."""
        offsets = []

        try:
            payload = packet_bytes[ntp_start:]
            if len(payload) < 48:  # NTP packet is fixed 48 bytes
                return offsets

            base = ntp_start
            if self.separator_config.detailed_app_parsing:
                # NTP header fields
                offsets.extend([
                    ProtocolOffset(base + 1, "ntp_stratum", "ntp", 5),       # After LI+VN+Mode
                    ProtocolOffset(base + 2, "ntp_poll", "ntp", 5),          # After stratum
                    ProtocolOffset(base + 3, "ntp_precision", "ntp", 5),     # After poll
                    ProtocolOffset(base + 4, "ntp_root_delay", "ntp", 5),    # After precision
                    ProtocolOffset(base + 8, "ntp_root_disp", "ntp", 5),     # After root delay
                    ProtocolOffset(base + 12, "ntp_ref_id", "ntp", 5),       # After root dispersion
                    ProtocolOffset(base + 16, "ntp_ref_ts", "ntp", 5),       # After reference ID
                    ProtocolOffset(base + 24, "ntp_orig_ts", "ntp", 5),      # After reference timestamp
                    ProtocolOffset(base + 32, "ntp_recv_ts", "ntp", 5),      # After originate timestamp
                    ProtocolOffset(base + 40, "ntp_xmit_ts", "ntp", 5)       # After receive timestamp
                ])
            else:
                offsets.append(ProtocolOffset(base + 48, "ntp_end", "ntp", 5))

        except Exception as e:
            logger.debug(f"Error parsing NTP: {e}")

        return offsets

    def _calculate_sip_offsets(self, packet_bytes: bytes, sip_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for SIP fields."""
        offsets = []

        try:
            payload = packet_bytes[sip_start:]
            if len(payload) < 4:
                return offsets

            # SIP is text-based, find header/body boundary
            header_end = payload.find(b'\r\n\r\n')
            if header_end == -1:
                header_end = payload.find(b'\n\n')

            if header_end != -1:
                body_start = sip_start + header_end + (4 if b'\r\n\r\n' in payload else 2)
                offsets.append(ProtocolOffset(body_start, "sip_body", "sip", 5))
            else:
                offsets.append(ProtocolOffset(sip_start + len(payload), "sip_end", "sip", 5))

        except Exception as e:
            logger.debug(f"Error parsing SIP: {e}")

        return offsets

    def _calculate_rtp_offsets(self, packet_bytes: bytes, rtp_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for RTP fields."""
        offsets = []

        try:
            payload = packet_bytes[rtp_start:]
            if len(payload) < 12:  # Minimum RTP header
                return offsets

            base = rtp_start
            if self.separator_config.detailed_app_parsing:
                # RTP header: V+P+X+CC(1) + M+PT(1) + Seq(2) + TS(4) + SSRC(4) + CSRC...
                offsets.extend([
                    ProtocolOffset(base + 1, "rtp_pt", "rtp", 5),      # After V+P+X+CC
                    ProtocolOffset(base + 2, "rtp_seq", "rtp", 5),     # After M+PT
                    ProtocolOffset(base + 4, "rtp_ts", "rtp", 5),      # After sequence
                    ProtocolOffset(base + 8, "rtp_ssrc", "rtp", 5),    # After timestamp
                    ProtocolOffset(base + 12, "rtp_payload", "rtp", 5) # After SSRC
                ])
            else:
                offsets.append(ProtocolOffset(base + 12, "rtp_payload", "rtp", 5))

        except Exception as e:
            logger.debug(f"Error parsing RTP: {e}")

        return offsets

    # Additional protocol parsers (simplified implementations)
    def _calculate_tftp_offsets(self, packet_bytes: bytes, tftp_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for TFTP fields."""
        offsets = []
        try:
            payload = packet_bytes[tftp_start:]
            if len(payload) >= 4:
                offsets.append(ProtocolOffset(tftp_start + 2, "tftp_data", "tftp", 5))
        except Exception as e:
            logger.debug(f"Error parsing TFTP: {e}")
        return offsets

    def _calculate_telnet_offsets(self, packet_bytes: bytes, telnet_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for Telnet fields."""
        offsets = []
        try:
            payload = packet_bytes[telnet_start:]
            if len(payload) > 0:
                # Telnet is mostly plain text with occasional IAC commands
                offsets.append(ProtocolOffset(telnet_start + len(payload), "telnet_end", "telnet", 5))
        except Exception as e:
            logger.debug(f"Error parsing Telnet: {e}")
        return offsets

    def _calculate_smb_offsets(self, packet_bytes: bytes, smb_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for SMB fields."""
        offsets = []
        try:
            payload = packet_bytes[smb_start:]
            if len(payload) >= 32:  # Basic SMB header check
                offsets.extend([
                    ProtocolOffset(smb_start + 4, "smb_command", "smb", 5),
                    ProtocolOffset(smb_start + 32, "smb_data", "smb", 5)
                ])
        except Exception as e:
            logger.debug(f"Error parsing SMB: {e}")
        return offsets

    def _calculate_netbios_offsets(self, packet_bytes: bytes, netbios_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for NetBIOS fields."""
        offsets = []
        try:
            payload = packet_bytes[netbios_start:]
            if len(payload) >= 4:
                offsets.append(ProtocolOffset(netbios_start + 4, "netbios_data", "netbios", 5))
        except Exception as e:
            logger.debug(f"Error parsing NetBIOS: {e}")
        return offsets

    def _calculate_radius_offsets(self, packet_bytes: bytes, radius_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for RADIUS fields."""
        offsets = []
        try:
            payload = packet_bytes[radius_start:]
            if len(payload) >= 20:  # RADIUS header is 20 bytes
                base = radius_start
                offsets.extend([
                    ProtocolOffset(base + 1, "radius_id", "radius", 5),
                    ProtocolOffset(base + 2, "radius_length", "radius", 5),
                    ProtocolOffset(base + 4, "radius_auth", "radius", 5),
                    ProtocolOffset(base + 20, "radius_attrs", "radius", 5)
                ])
        except Exception as e:
            logger.debug(f"Error parsing RADIUS: {e}")
        return offsets

    def _calculate_stun_offsets(self, packet_bytes: bytes, stun_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for STUN fields."""
        offsets = []
        try:
            payload = packet_bytes[stun_start:]
            if len(payload) >= 20:  # STUN header is 20 bytes
                base = stun_start
                offsets.extend([
                    ProtocolOffset(base + 2, "stun_length", "stun", 5),
                    ProtocolOffset(base + 4, "stun_magic", "stun", 5),
                    ProtocolOffset(base + 8, "stun_txn_id", "stun", 5),
                    ProtocolOffset(base + 20, "stun_attrs", "stun", 5)
                ])
        except Exception as e:
            logger.debug(f"Error parsing STUN: {e}")
        return offsets

    def _calculate_rpc_offsets(self, packet_bytes: bytes, rpc_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for RPC fields."""
        offsets = []
        try:
            payload = packet_bytes[rpc_start:]
            if len(payload) >= 12:
                offsets.append(ProtocolOffset(rpc_start + 12, "rpc_data", "rpc", 5))
        except Exception as e:
            logger.debug(f"Error parsing RPC: {e}")
        return offsets

    def _calculate_h225_offsets(self, packet_bytes: bytes, h225_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for H.225 fields."""
        offsets = []
        try:
            payload = packet_bytes[h225_start:]
            if len(payload) >= 8:
                offsets.append(ProtocolOffset(h225_start + 8, "h225_data", "h225", 5))
        except Exception as e:
            logger.debug(f"Error parsing H.225: {e}")
        return offsets

    def _calculate_rfb_offsets(self, packet_bytes: bytes, rfb_start: int) -> List[ProtocolOffset]:
        """Calculate byte offsets for RFB (VNC) fields."""
        offsets = []
        try:
            payload = packet_bytes[rfb_start:]
            if len(payload) >= 12:
                offsets.append(ProtocolOffset(rfb_start + 12, "rfb_data", "rfb", 5))
        except Exception as e:
            logger.debug(f"Error parsing RFB: {e}")
        return offsets

    def _calculate_protocol_offsets(self, packet_bytes: bytes, pcap_filename: str,
                                    packet_index: int) -> List[ProtocolOffset]:
        """
        Calculate all protocol field offsets for a packet using hybrid parsing approach.
        Now supports up to Layer 5+ (Application Layer).

        Args:
            packet_bytes: Raw packet bytes
            pcap_filename: Name of PCAP file (for error logging)
            packet_index: Index of packet in file (for error logging)

        Returns:
            List of ProtocolOffset objects sorted by byte position
        """
        offsets = []
        src_port = dst_port = 0
        transport_protocol = None
        app_start = None

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

                        # Extract ports for application layer detection
                        if protocol in [6, 17] and len(packet_bytes) >= transport_start + 4:  # TCP or UDP
                            src_port = struct.unpack('!H', packet_bytes[transport_start:transport_start+2])[0]
                            dst_port = struct.unpack('!H', packet_bytes[transport_start+2:transport_start+4])[0]
                            transport_protocol = "tcp" if protocol == 6 else "udp"

                            if protocol == 6:  # TCP
                                data_offset = (packet_bytes[transport_start + 12] >> 4) & 0xF
                                app_start = transport_start + (data_offset * 4)
                            else:  # UDP
                                app_start = transport_start + 8

                    elif isinstance(eth.data, dpkt.ip6.IP6):
                        protocol = eth.data.nxt
                        transport_start = 14 + 40  # IPv6 header is fixed 40 bytes

                        # Extract ports for IPv6
                        if protocol in [6, 17] and len(packet_bytes) >= transport_start + 4:
                            src_port = struct.unpack('!H', packet_bytes[transport_start:transport_start+2])[0]
                            dst_port = struct.unpack('!H', packet_bytes[transport_start+2:transport_start+4])[0]
                            transport_protocol = "tcp" if protocol == 6 else "udp"

                            if protocol == 6:  # TCP
                                data_offset = (packet_bytes[transport_start + 12] >> 4) & 0xF
                                app_start = transport_start + (data_offset * 4)
                            else:  # UDP
                                app_start = transport_start + 8
                    else:
                        protocol = None
                        transport_start = None

                    if protocol is not None and transport_start is not None:
                        transport_offsets = self._calculate_transport_offsets(
                            packet_bytes, transport_start, protocol)
                        offsets.extend(transport_offsets)

                # Layer 5+ - Application
                if (self.separator_config.max_depth >= 5 and
                    app_start is not None and
                    app_start < len(packet_bytes) and
                    transport_protocol is not None):

                    app_offsets = self._calculate_application_offsets(
                        packet_bytes, app_start, src_port, dst_port, transport_protocol)
                    offsets.extend(app_offsets)

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
        Now supports up to Layer 5+ separators.

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
        Now supports up to Layer 5+ (Application Layer) field separation.

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