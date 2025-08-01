#!/usr/bin/env python3
"""
Debug script to see exactly how problematic fields are being handled.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.byte.raw.field_aware_tokenizer import PCAPFieldTokenizer
from scapy.all import rdpcap

def debug_field_processing():
    """Debug field processing for a single packet."""
    print("🔍 Debugging field processing...")
    
    try:
        pcap_path = "/home/orojoa/NetworkTokenizers/data/test.pcap"
        
        if not Path(pcap_path).exists():
            print(f"   ❌ Test file not found: {pcap_path}")
            return False
            
        # Read packets
        packets = rdpcap(pcap_path)
        print(f"📦 Loaded {len(packets)} packets")
        
        # Process first packet with debug
        packet = packets[0]
        print(f"\n🔎 Analyzing first packet: {packet.summary()}")
        
        current_layer = packet
        layer_num = 1
        
        while current_layer:
            print(f"\n--- Layer {layer_num}: {current_layer.__class__.__name__} ---")
            
            for field in current_layer.fields_desc:
                if field.name in current_layer.fields:
                    print(f"  🔧 Field: {field.name}")
                    field_bytes = PCAPFieldTokenizer._get_field_bytes(current_layer, field, debug=True)
                    print(f"    📊 Result: {len(field_bytes)} bytes: {field_bytes[:20]}{'...' if len(field_bytes) > 20 else ''}")
                    
                    # Show what tokens this becomes
                    if field_bytes:
                        field_string = field_bytes.decode('latin-1')
                        tokens = [char for char in field_string]
                        print(f"    🔤 Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            
            # Move to next layer
            if hasattr(current_layer, 'payload'):
                if isinstance(current_layer.payload, type(current_layer.payload)) and hasattr(current_layer.payload, 'fields_desc'):
                    current_layer = current_layer.payload
                    layer_num += 1
                else:
                    print(f"\n--- Raw Payload ---")
                    try:
                        payload_bytes = bytes(current_layer.payload)
                        print(f"  📊 Payload: {len(payload_bytes)} bytes")
                        if payload_bytes:
                            payload_string = payload_bytes.decode('latin-1')
                            tokens = [char for char in payload_string]
                            print(f"    🔤 Sample tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
                    except Exception as e:
                        print(f"    ❌ Payload error: {e}")
                    break
            else:
                break
                
        print(f"\n✅ Field processing debug complete!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_field_processing()
    sys.exit(0 if success else 1)