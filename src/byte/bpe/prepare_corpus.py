#!/usr/bin/env python3
"""
PCAP Corpus Preparation Script for SentencePiece Training

This script processes multiple PCAP files using multiprocessing and creates a corpus
file where each packet is treated as a sentence. Each packet's bytes are converted
to space-separated decimal values for SentencePiece training.

Usage:
    python prepare_corpus.py --input-dir /path/to/pcap/files --output corpus.txt --workers 8
"""

import argparse
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Any

from src.byte.raw.token_pcap_byte_tokenizer import TokenPCAPByteTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PCAPCorpusBuilder:
    """
    Builds a SentencePiece training corpus from PCAP files.
    Each packet is treated as a sentence with space-separated byte values.
    """

    def __init__(self, output_file: str, use_hex: bool = False):
        """
        Initialize the corpus builder.

        Args:
            output_file: Path to output corpus file
            use_hex: If True, use hex representation; if False, use decimal
        """
        self.output_file = output_file
        self.use_hex = use_hex
        self.tokenizer = TokenPCAPByteTokenizer()

        # Statistics
        self.total_packets = 0
        self.total_files_processed = 0
        self.skipped_packets = 0
        self.total_bytes_processed = 0

    def packet_to_sentence(self, packet_bytes: bytes) -> str:
        """
        Convert packet bytes to a sentence string for SentencePiece.

        Args:
            packet_bytes: Raw packet bytes

        Returns:
            String representation of packet for corpus
        """
        if self.use_hex:
            # Hex representation: "00 01 02 ff"
            return " ".join(f"{b:02x}" for b in packet_bytes)
        else:
            # Decimal representation: "0 1 2 255"
            return " ".join(str(b) for b in packet_bytes)

    def process_single_pcap(self, pcap_path: str) -> tuple[int | Any, int | Any, str] | tuple[int, int, int, str]:
        """
        Process a single PCAP file and return packet sentences.

        Args:
            pcap_path: Path to PCAP file

        Returns:
            Tuple of (num_packets, num_skipped, num_bytes, temp_file_path)
        """
        try:
            # Read packets from PCAP
            packets = self.tokenizer.read_pcap_packets(pcap_path)

            # Create temporary file for this worker's output
            temp_fd, temp_path = tempfile.mkstemp(suffix='.txt', prefix='pcap_corpus_')

            num_packets = 0
            num_bytes = 0

            with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
                for packet in packets:
                    packet_size = len(packet)

                    # Convert packet to sentence
                    sentence = self.packet_to_sentence(packet)
                    temp_file.write(sentence + '\n')

                    num_packets += 1
                    num_bytes += packet_size

            logger.debug(f"Processed {pcap_path}: {num_packets} packets")
            return num_packets, num_bytes, temp_path

        except Exception as e:
            logger.error(f"Error processing {pcap_path}: {e}")
            return 0, 0, 0, ""

    def merge_temp_files(self, temp_files: List[str]) -> None:
        """
        Merge temporary files into the final corpus file.

        Args:
            temp_files: List of temporary file paths to merge
        """
        logger.info(f"Merging {len(temp_files)} temporary files into {self.output_file}")

        with open(self.output_file, 'w', encoding='utf-8') as output:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    try:
                        with open(temp_file, 'r', encoding='utf-8') as temp:
                            shutil.copyfileobj(temp, output)

                        # Clean up temporary file
                        os.unlink(temp_file)

                    except Exception as e:
                        logger.error(f"Error merging {temp_file}: {e}")
    @staticmethod
    def find_pcap_files(input_path: str, recursive: bool = True) -> List[str]:
        """
        Find all PCAP files in the input path.

        Args:
            input_path: Input directory or file path
            recursive: Whether to search recursively

        Returns:
            List of PCAP file paths
        """
        pcap_files = []
        input_path = Path(input_path)

        if input_path.is_file():
            if input_path.suffix.lower() in ['.pcap', '.pcapng', '.cap']:
                pcap_files.append(str(input_path))
        elif input_path.is_dir():
            pattern = "**/*" if recursive else "*"
            for ext in ['.pcap', '.pcapng', '.cap']:
                pcap_files.extend([
                    str(f) for f in input_path.glob(f"{pattern}{ext}")
                ])
                pcap_files.extend([
                    str(f) for f in input_path.glob(f"{pattern}{ext.upper()}")
                ])

        return sorted(pcap_files)

    def build_corpus(self, input_path: str, num_workers: int = None,
                     recursive: bool = True, progress_interval: int = 100) -> None:
        """
        Build corpus from PCAP files using multiprocessing.

        Args:
            input_path: Input directory or file path
            num_workers: Number of worker processes (None for auto)
            recursive: Whether to search recursively for PCAP files
            progress_interval: How often to log progress
        """
        # Find all PCAP files
        pcap_files = self.find_pcap_files(input_path, recursive)

        if not pcap_files:
            logger.error(f"No PCAP files found in {input_path}")
            return

        logger.info(f"Found {len(pcap_files)} PCAP files to process")

        # Determine number of workers
        if num_workers is None:
            num_workers = min(mp.cpu_count(), len(pcap_files))

        logger.info(f"Using {num_workers} worker processes")

        # Process files in parallel
        start_time = time.time()
        temp_files = []

        with mp.Pool(processes=num_workers) as pool:
            # Submit all jobs
            results = pool.map_async(self.process_single_pcap, pcap_files)

            # Process results
            for i, (num_packets, num_bytes, temp_file) in enumerate(results.get()):
                if temp_file:
                    temp_files.append(temp_file)

                self.total_packets += num_packets
                self.total_bytes_processed += num_bytes
                self.total_files_processed += 1

                # Log progress
                if (i + 1) % progress_interval == 0 or i == len(pcap_files) - 1:
                    elapsed = time.time() - start_time
                    files_per_sec = (i + 1) / elapsed
                    logger.info(f"Processed {i + 1}/{len(pcap_files)} files "
                                f"({files_per_sec:.1f} files/sec) - "
                                f"{self.total_packets:,} packets total")

        # Merge temporary files into final corpus
        self.merge_temp_files(temp_files)

        # Final statistics
        total_time = time.time() - start_time
        self.print_statistics(total_time)

    def print_statistics(self, total_time: float) -> None:
        """Print final processing statistics."""
        print("\n" + "=" * 60)
        print("CORPUS BUILDING COMPLETE")
        print("=" * 60)
        print(f"📁 Files processed: {self.total_files_processed:,}")
        print(f"📦 Total packets: {self.total_packets:,}")
        print(f"🚫 Skipped packets: {self.skipped_packets:,}")
        print(
            f"💾 Total bytes processed: {self.total_bytes_processed:,} ({self.total_bytes_processed / 1024 / 1024:.1f} MB)")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"🚀 Processing rate: {self.total_files_processed / total_time:.1f} files/sec")
        print(f"📊 Average packets per file: {self.total_packets / max(1, self.total_files_processed):.1f}")

        if os.path.exists(self.output_file):
            output_size = os.path.getsize(self.output_file)
            print(f"📄 Output corpus size: {output_size:,} bytes ({output_size / 1024 / 1024:.1f} MB)")

            # Count lines in output file
            with open(self.output_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"📝 Corpus lines (sentences): {line_count:,}")

        print(f"✅ Corpus saved to: {self.output_file}")
        print("=" * 60)


def main():
    """Main function to parse arguments and run corpus building."""
    parser = argparse.ArgumentParser(
        description="Build SentencePiece corpus from PCAP files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Input directory containing PCAP files, or single PCAP file"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output corpus file path"
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of worker processes (default: auto)"
    )

    parser.add_argument(
        "--use-hex",
        action="store_true",
        help="Use hex representation instead of decimal"
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories recursively"
    )

    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="How often to log progress (number of files)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not os.path.exists(args.input_dir):
        logger.error(f"Input path does not exist: {args.input_dir}")
        return 1

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build corpus
    logger.info("Starting PCAP corpus building...")

    corpus_builder = PCAPCorpusBuilder(
        output_file=args.output,
        use_hex=args.use_hex
    )

    try:
        corpus_builder.build_corpus(
            input_path=args.input_dir,
            num_workers=args.workers,
            recursive=not args.no_recursive,
            progress_interval=args.progress_interval
        )

        logger.info("Corpus building completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("Corpus building interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during corpus building: {e}")
        return 1


if __name__ == "__main__":
    exit(main())