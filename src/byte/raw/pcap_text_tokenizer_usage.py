# Example usage
from pathlib import Path
from scapy.all import wrpcap
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP

from src.byte.raw.pcap_text_tokenizer import PCAPTextTokenizer

if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer = PCAPTextTokenizer()

    # --- Create a temporary PCAP file with a valid packet ---
    temp_pcap_path = Path("temp_test.pcap")
    scapy_packet = Ether() / IP(dst="8.8.8.8") / TCP(dport=443) / b"GET / HTTP/1.1\r\n"
    original_packet_bytes = bytes(scapy_packet)
    wrpcap(str(temp_pcap_path), scapy_packet)

    # Test with text
    text = "Hello, world! 🌍"
    text_tokens = tokenizer.tokenize(text)
    text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    print(f"Text: {text}")
    print(f"Tokens: {text_tokens[:10]}...")
    print(f"Token IDs: {text_ids[:10]}...")
    print("-" * 20)

    # Test with a PCAP file path
    pcap_tokens = tokenizer.tokenize_pcap(temp_pcap_path)
    pcap_ids = tokenizer.convert_tokens_to_ids(pcap_tokens)
    print(f"Original Packet bytes: {original_packet_bytes.hex()}")
    print(f"PCAP file used: {temp_pcap_path}")
    print(f"PCAP tokens: {pcap_tokens[:20]}...")
    print(f"PCAP token IDs: {pcap_ids[:20]}...")
    print("-" * 20)

    # Test different extraction modes
    payload_tokens = tokenizer.tokenize_pcap(temp_pcap_path, extract_payload=True)
    print(f"Payload-only tokens: {payload_tokens[:20]}...")

    # Test skipping link layer
    no_link_layer_tokens = tokenizer.tokenize_pcap(temp_pcap_path, skip_link_layer=True)
    print(f"No-link-layer tokens: {no_link_layer_tokens[:20]}...")
    print("-" * 20)

    # Test mixed tokenization
    mixed_tokens = tokenizer.tokenize_mixed(text="Web request: ", pcap_data=temp_pcap_path)
    print(f"Mixed tokens: {mixed_tokens[:30]}...")
    print("-" * 20)

    # Test decoding packets back from tokens
    decoded_packets = tokenizer.decode_packets_from_tokens(pcap_tokens)
    decoded_packet_bytes = decoded_packets[0]
    print(f"Decoded full packet: {decoded_packet_bytes.hex()}")
    print(f"Original matches decoded: {original_packet_bytes == decoded_packet_bytes}")

    # --- Clean up the temporary file ---
    temp_pcap_path.unlink()
    print("\nTemporary PCAP file removed.")