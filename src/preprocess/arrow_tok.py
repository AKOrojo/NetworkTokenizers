import argparse
import multiprocessing
import os
import time
from pathlib import Path
from typing import List

import pyarrow as pa
from tqdm import tqdm

from src.byte.raw.token_pcap_byte_tokenizer import TokenPCAPByteTokenizer

# --- Configuration ---
TOKENS_PER_ARROW_FILE = 2_000_000_000
FILES_PER_CHUNK = 500


def get_pcap_files(pcap_dir: Path) -> List[Path]:
    """Scans a directory for .pcap and .pcapng files."""
    print("Scanning for PCAP files...")
    files = list(pcap_dir.glob("**/*.pcap")) + list(pcap_dir.glob("**/*.pcapng"))
    print(f"Found {len(files)} PCAP files.")
    return files


def process_chunk(
        pcap_files: List[Path],
        tokenizer: TokenPCAPByteTokenizer,
        output_dir: Path,
        worker_id: int,
        delete_originals: bool = False,
):
    """
    Processes a chunk of PCAP files.
    - Tokenizes each file.
    - Buffers tokens in memory.
    - Writes buffered tokens to Arrow files once a threshold is met.
    - Optionally deletes original PCAP files after successful processing.
    """
    token_buffer = []
    tokens_in_buffer = 0
    arrow_file_count = 0
    pbar = tqdm(
        total=len(pcap_files),
        desc=f"Worker {worker_id}",
        position=worker_id,
        leave=False,
    )

    for pcap_file in pcap_files:
        try:
            token_ids = tokenizer.tokenize_pcap_to_ids(pcap_file)

            if token_ids:
                token_buffer.append(token_ids)
                tokens_in_buffer += len(token_ids)

            # If buffer is full, write it to an Arrow file
            if tokens_in_buffer >= TOKENS_PER_ARROW_FILE:
                write_arrow_file(token_buffer, output_dir, worker_id, arrow_file_count)
                arrow_file_count += 1
                # Reset buffer
                token_buffer = []
                tokens_in_buffer = 0

        except Exception as e:
            print(f"Worker {worker_id}: Error processing {pcap_file}: {e}")
        pbar.update(1)

    # Write any remaining tokens in the buffer to a final file for this chunk
    if token_buffer:
        write_arrow_file(token_buffer, output_dir, worker_id, arrow_file_count)

    pbar.close()

    # Only delete source files if explicitly requested
    if delete_originals:
        print(f"Worker {worker_id}: Deleting {len(pcap_files)} source PCAP files...")
        for pcap_file in pcap_files:
            try:
                os.remove(pcap_file)
            except OSError as e:
                print(f"Worker {worker_id}: Failed to delete {pcap_file}: {e}")
        print(f"Worker {worker_id}: Finished chunk processing and deletion.")
    else:
        print(f"Worker {worker_id}: Finished chunk processing (original files preserved).")


def write_arrow_file(token_buffer: List[List[int]], output_dir: Path, worker_id: int, file_index: int):
    """Writes a list of token sequences to a compressed Arrow file."""
    try:
        # --- START: ROBUSTNESS FIX ---

        # 1. Defensively filter out any empty lists that might have slipped through.
        #    This is a critical step to prevent pyarrow from processing empty structures.
        filtered_buffer = [tokens for tokens in token_buffer if tokens]
        if not filtered_buffer:
            tqdm.write(f"Worker {worker_id}: Skipping write for empty buffer.")
            return

        # 2. Define the schema explicitly for clarity and consistency.
        schema = pa.schema([
            pa.field("tokens", pa.list_(pa.uint16()))
        ])

        # 3. Create the Arrow Array from the filtered buffer.
        token_array = pa.array(filtered_buffer, type=schema.field('tokens').type)

        # 4. Create a Table directly from the array and schema.
        table = pa.table([token_array], schema=schema)

        # --- END: ROBUSTNESS FIX ---

        # Use a unique filename for each worker's output
        output_filename = output_dir / f"worker_{worker_id}_part_{file_index:04d}.arrow"

        # Write the file with ZSTD compression for a good balance of speed and size
        with pa.ipc.new_file(output_filename, table.schema, options=pa.ipc.IpcWriteOptions(compression="zstd")) as writer:
            writer.write_table(table)

        tqdm.write(f"Worker {worker_id}: Wrote {len(token_buffer)} sequences to {output_filename}")

    except Exception as e:
        tqdm.write(f"Worker {worker_id}: Failed to write Arrow file: {e}")


def main():
    """Main function to drive the preprocessing."""
    parser = argparse.ArgumentParser(
        description="Tokenize PCAP files and save to Arrow format, with option to delete originals."
    )
    parser.add_argument(
        "pcap_dir", type=str, help="Directory containing the source PCAP files."
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to save the output Arrow files."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel worker processes to use.",
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete original PCAP files after successful processing. DEFAULT: Keep original files.",
    )
    args = parser.parse_args()

    pcap_dir = Path(args.pcap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pcap_dir.is_dir():
        print(f"Error: PCAP directory not found at {pcap_dir}")
        return

    # Show deletion behavior to user
    if args.delete_originals:
        print("⚠️  WARNING: Original PCAP files will be DELETED after processing!")
    else:
        print("✅ Original PCAP files will be preserved.")

    # 1. Discover all files first
    all_files = get_pcap_files(pcap_dir)
    if not all_files:
        print("No PCAP files found. Exiting.")
        return

    # 2. Create file chunks
    chunks = [
        all_files[i: i + FILES_PER_CHUNK]
        for i in range(0, len(all_files), FILES_PER_CHUNK)
    ]
    print(f"Divided files into {len(chunks)} chunks of up to {FILES_PER_CHUNK} files each.")

    # 3. Initialize tokenizer
    tokenizer = TokenPCAPByteTokenizer()

    # 4. Process chunks in parallel
    start_time = time.time()

    # Prepare arguments for each worker (now includes delete_originals flag)
    worker_args = [
        (chunks[i], tokenizer, output_dir, i, args.delete_originals) for i in range(len(chunks))
    ]

    # Use a multiprocessing pool
    # starmap is useful for functions with multiple arguments
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        pool.starmap(process_chunk, worker_args)

    end_time = time.time()
    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Tokenized data is available in: {output_dir}")
    if args.delete_originals:
        print("Original PCAP files have been deleted.")
    else:
        print("Original PCAP files have been preserved.")
    print("=" * 50)


if __name__ == "__main__":
    main()