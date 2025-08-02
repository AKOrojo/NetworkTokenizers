import dpkt

with open('/data/test.pcap', 'rb') as f:
    pcap = dpkt.pcap.Reader(f)
    for timestamp, buf in pcap:
        # buf contains the raw packet bytes as captured
        # This is NOT parsed/reconstructed data
        print(f"Raw packet bytes: {len(buf)} bytes")
        print(buf)
        # You can use buf directly for tokenization
        raw_bytes = buf