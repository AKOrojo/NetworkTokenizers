import argparse
import tempfile
import os
import json # Import the json library
from pathlib import Path
from typing import Dict, Optional
from multiprocessing import Pool

import sentencepiece as spm
from scapy.all import rdpcap
from tqdm import tqdm

# This mapping is the same as before.
_BYTE_TO_CHAR: Dict[int, str] = {i: chr(i + 1) for i in range(256)}

def process_single_pcap(pcap_path: Path) -> Optional[str]:
    """Worker function to process a single PCAP file."""
    try:
        packets = rdpcap(str(pcap_path))
        all_bytes = b"".join(bytes(p) for p in packets)
        return "".join([_BYTE_TO_CHAR[b] for b in all_bytes])
    except Exception as e:
        print(f"Warning: Could not process file {pcap_path}. Reason: {e}")
        return None

def prepare_training_data_parallel(pcap_directory: str, output_file: str) -> None:
    """Finds all PCAP files and processes them in parallel."""
    pcap_dir = Path(pcap_directory)
    pcap_files = list(pcap_dir.glob("*.pcap")) + list(pcap_dir.glob("*.pcapng"))
    if not pcap_files:
        raise FileNotFoundError(f"No .pcap or .pcapng files found in '{pcap_directory}'")
    print(f"Found {len(pcap_files)} PCAP files. Processing in parallel...")
    with open(output_file, "w", encoding="utf-8") as f:
        with Pool() as pool:
            for mapped_string in tqdm(pool.imap(process_single_pcap, pcap_files), total=len(pcap_files)):
                if mapped_string:
                    f.write(mapped_string + "\n")

def save_vocab_as_json(model_prefix: str) -> None:
    """
    Reads the .vocab file from SentencePiece and saves it as a transformers-compatible vocab.json.
    """
    vocab_file = f"{model_prefix}.vocab"
    json_file = f"{model_prefix}-vocab.json" # Name the output file
    
    print(f"Converting '{vocab_file}' to JSON format...")
    
    vocab = {}
    with open(vocab_file, "r", encoding="utf-8") as f:
        # Enumerate each line to get the integer ID for each token
        for i, line in enumerate(f):
            # The token is the part of the line before the first tab
            token = line.split("\t")[0]
            vocab[token] = i
            
    with open(json_file, "w", encoding="utf-8") as f:
        # Dump the dictionary to a JSON file with nice formatting
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Vocabulary saved to '{json_file}'")


def train_pcap_sentencepiece_tokenizer(
    pcap_dir: str,
    model_prefix: str,
    vocab_size: int = 1024,
    model_type: str = "bpe",
    num_threads: int = 0
) -> None:
    """Trains a SentencePiece tokenizer and creates a JSON vocab file."""
    preprocessed_data_file = "/data/orojoa/preprocessed_pcap_data.txt"
    print(f"Preparing to write pre-processed data to: {preprocessed_data_file}")

    try:
        prepare_training_data_parallel(pcap_dir, preprocessed_data_file)
        print("\nStarting SentencePiece training...")

        spm.SentencePieceTrainer.train(
            input=preprocessed_data_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            byte_fallback=True,
            user_defined_symbols=",".join(_BYTE_TO_CHAR.values()),
            num_threads=num_threads
        )

        print(f"\n✅ Training complete! Model saved as '{model_prefix}.model' and '{model_prefix}.vocab'")

        # --- NEW STEP: Convert the .vocab file to .json ---
        save_vocab_as_json(model_prefix)

    finally:
        print("Pre-processed data file has been saved for future use.")


def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer on a directory of PCAP files.")
    parser.add_argument("--pcap-dir", type=str, required=True, help="Directory containing the PCAP files.")
    parser.add_argument("--model-prefix", type=str, required=True, help="Output prefix for the trained model and vocab files.")
    parser.add_argument("--vocab-size", type=int, default=4096, help="The desired size of the vocabulary.")
    parser.add_argument("--model-type", type=str, default="bpe", choices=["bpe", "unigram", "char", "word"], help="The model type for SentencePiece.")
    parser.add_argument("--num-threads", type=int, default=os.cpu_count(), help="Number of threads to use for both preparation and training.")
    args = parser.parse_args()

    train_pcap_sentencepiece_tokenizer(
        pcap_dir=args.pcap_dir,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        num_threads=args.num_threads
    )

if __name__ == '__main__':
    main()