import tempfile
from pathlib import Path
from typing import Dict

import sentencepiece as spm
from scapy.all import rdpcap
from tqdm import tqdm

_BYTE_TO_CHAR: Dict[int, str] = {i: chr(i + 1) for i in range(256)}

def prepare_training_data(pcap_directory: str, output_file: str) -> None:
    """
    Reads all PCAP files, converts their bytes to mapped characters, and saves to a text file.
    """
    pcap_dir = Path(pcap_directory)
    pcap_files = list(pcap_dir.glob("*.pcap")) + list(pcap_dir.glob("*.pcapng"))

    if not pcap_files:
        raise FileNotFoundError(f"No .pcap or .pcapng files found in '{pcap_directory}'")

    print(f"Found {len(pcap_files)} PCAP files. Preparing training data...")

    with open(output_file, "w", encoding="utf-8") as f:
        for pcap_path in tqdm(pcap_files, desc="Processing PCAPs"):
            try:
                # Read all packets and concatenate their raw bytes
                packets = rdpcap(str(pcap_path))
                all_bytes = b"".join(bytes(p) for p in packets)

                # Convert bytes to the mapped character representation
                mapped_chars = [_BYTE_TO_CHAR[b] for b in all_bytes]
                
                # Write the mapped string as a single line in the training file
                f.write("".join(mapped_chars) + "\n")

            except Exception as e:
                print(f"Warning: Could not process file {pcap_path}. Reason: {e}")

def train_pcap_sentencepiece_tokenizer(
    pcap_dir: str,
    model_prefix: str,
    vocab_size: int = 1024,
    model_type: str = "bpe",
) -> None:
    """
    Trains a SentencePiece tokenizer on a directory of PCAP files.

    Args:
        pcap_dir (str): Path to the directory containing .pcap or .pcapng files.
        model_prefix (str): Prefix for the output model and vocab files (e.g., 'pcap_spm').
        vocab_size (int): The desired size of the vocabulary.
        model_type (str): The model type for SentencePiece ('bpe', 'unigram', 'char', 'word').
    """
    # Use a temporary file to store the preprocessed training data
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as temp_f:
        training_file_path = temp_f.name

    try:
        # 1. Prepare the data by converting PCAP bytes to a mapped text file
        prepare_training_data(pcap_dir, training_file_path)

        # 2. Define the arguments for the SentencePiece trainer
        # The user_defined_symbols are our 256 byte characters. This ensures they are
        # all included in the vocabulary as basic units.
        spm_command = (
            f"--input={training_file_path} "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} "
            f"--model_type={model_type} "
            f"--character_coverage=1.0 "
            f"--byte_fallback=true " # Handles any unexpected characters by treating them as bytes
            f"--user_defined_symbols={','.join(_BYTE_TO_CHAR.values())}"
        )

        print("\nStarting SentencePiece training...")
        print(f"Command: sentencepiece.SentencePieceTrainer.train('{spm_command}')")

        # 3. Run the trainer
        spm.SentencePieceTrainer.train(spm_command)

        print(f"\n✅ Training complete! Model saved as '{model_prefix}.model' and '{model_prefix}.vocab'")

    finally:
        # 4. Clean up the temporary training file
        Path(training_file_path).unlink()
        print(f"Cleaned up temporary file: {training_file_path}")

if __name__ == '__main__':
    # --- How to use ---
    
    # 1. Place all your .pcap files into a single directory.
    PCAP_DATA_DIRECTORY = "./data"
    
    # 2. Define the output model name and desired vocabulary size.
    VOCABULARY_SIZE = 512
    MODEL_NAME_PREFIX = f"pcap_byte_bpe_{VOCABULARY_SIZE}"
    
    # 3. Run the training function.
    train_pcap_sentencepiece_tokenizer(
        pcap_dir=PCAP_DATA_DIRECTORY,
        model_prefix=MODEL_NAME_PREFIX,
        vocab_size=VOCABULARY_SIZE
    )