#!/usr/bin/env python3
"""
Field-Aware PCAP Corpus Preparation Script for WordPiece Training

This script processes multiple PCAP files using the enhanced field-aware tokenizer
and creates a corpus suitable for WordPiece training. Each packet is treated as a
sentence with protocol field boundaries marked by special tokens.

Usage:
    python prepare_field_aware_corpus.py --input-dir /path/to/pcap/files --output corpus.txt --workers 8
"""

import argparse
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Any, Tuple

from src.byte.raw.field_aware_pcap_byte_tokenizer import FieldAwarePCAPByteTokenizer, SeparatorConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FieldAwareCorpusBuilder:
    """
    Builds a WordPiece training corpus from PCAP files using field-aware tokenization.
    Each packet is treated as a sentence with protocol field boundaries preserved.
    """

    def __init__(self,
                 output_file: str,
                 separator_config: SeparatorConfig,
                 corpus_format: str = "tokens",
                 include_special_tokens: bool = True,
                 malformed_log_dir: str = None):
        """
        Initialize the field-aware corpus builder.

        Args:
            output_file: Path to output corpus file
            separator_config: Configuration for field separation
            corpus_format: Output format ("tokens", "hex", "decimal", "mixed")
            include_special_tokens: Whether to include <sep> tokens in output
            malformed_log_dir: Directory for malformed packet logs
        """
        self.output_file = output_file
        self.separator_config = separator_config
        self.corpus_format = corpus_format
        self.include_special_tokens = include_special_tokens

        # Setup malformed logging
        if malformed_log_dir:
            os.makedirs(malformed_log_dir, exist_ok=True)
            self.malformed_log_path = os.path.join(malformed_log_dir, "malformed_packets.log")
        else:
            self.malformed_log_path = "malformed_packets.log"

        # Initialize tokenizer
        self.tokenizer = FieldAwarePCAPByteTokenizer(
            separator_config=self.separator_config,
            malformed_log_path=self.malformed_log_path
        )

        # Statistics
        self.total_packets = 0
        self.total_files_processed = 0
        self.total_bytes_processed = 0
        self.total_separators_inserted = 0
        self.malformed_packets = 0

    def packet_to_sentence(self, packet_bytes: bytes, add_field_separators: bool = True) -> str:
        """
        Convert packet bytes to a sentence string for WordPiece training.

        Args:
            packet_bytes: Raw packet bytes
            add_field_separators: Whether to add field separators

        Returns:
            String representation of packet for corpus
        """
        if self.corpus_format == "tokens":
            # Use tokenizer to get field-aware token sequence
            tokens = self.tokenizer._tokenize_packet_with_separators(
                packet_bytes,
                "corpus_packet",
                0
            ) if add_field_separators else [chr(b) for b in packet_bytes]

            if self.include_special_tokens:
                # Keep <sep> tokens as special markers
                sentence_parts = []
                for token in tokens:
                    if token == "<sep>":
                        sentence_parts.append("[SEP]")  # WordPiece-friendly separator
                    else:
                        # Convert byte to token representation
                        byte_val = ord(token)
                        if 32 <= byte_val <= 126 and byte_val not in [ord('<'), ord('>')]:
                            sentence_parts.append(f"▁{token}")  # Use ▁ prefix for visible chars
                        else:
                            sentence_parts.append(f"▁{byte_val:02x}")  # Hex for non-printable
                return " ".join(sentence_parts)
            else:
                # Filter out separators and just use byte tokens
                byte_tokens = [token for token in tokens if token != "<sep>"]
                sentence_parts = []
                for token in byte_tokens:
                    byte_val = ord(token)
                    if 32 <= byte_val <= 126 and byte_val not in [ord('<'), ord('>')]:
                        sentence_parts.append(f"▁{token}")
                    else:
                        sentence_parts.append(f"▁{byte_val:02x}")
                return " ".join(sentence_parts)

        elif self.corpus_format == "hex":
            # Hex representation with field separators
            if add_field_separators:
                tokens = self.tokenizer._tokenize_packet_with_separators(
                    packet_bytes, "corpus_packet", 0
                )
                hex_parts = []
                for token in tokens:
                    if token == "<sep>":
                        hex_parts.append("[SEP]" if self.include_special_tokens else "")
                    else:
                        hex_parts.append(f"{ord(token):02x}")
                return " ".join(part for part in hex_parts if part)
            else:
                return " ".join(f"{b:02x}" for b in packet_bytes)

        elif self.corpus_format == "decimal":
            # Decimal representation with field separators
            if add_field_separators:
                tokens = self.tokenizer._tokenize_packet_with_separators(
                    packet_bytes, "corpus_packet", 0
                )
                dec_parts = []
                for token in tokens:
                    if token == "<sep>":
                        dec_parts.append("[SEP]" if self.include_special_tokens else "")
                    else:
                        dec_parts.append(str(ord(token)))
                return " ".join(part for part in dec_parts if part)
            else:
                return " ".join(str(b) for b in packet_bytes)

        elif self.corpus_format == "mixed":
            # Mixed format: printable chars as tokens, others as hex, with separators
            if add_field_separators:
                tokens = self.tokenizer._tokenize_packet_with_separators(
                    packet_bytes, "corpus_packet", 0
                )
                mixed_parts = []
                for token in tokens:
                    if token == "<sep>":
                        mixed_parts.append("[SEP]" if self.include_special_tokens else "")
                    else:
                        byte_val = ord(token)
                        if 32 <= byte_val <= 126 and byte_val not in [ord('<'), ord('>')]:
                            mixed_parts.append(f"▁{token}")
                        else:
                            mixed_parts.append(f"▁0x{byte_val:02x}")
                return " ".join(part for part in mixed_parts if part)
            else:
                mixed_parts = []
                for b in packet_bytes:
                    if 32 <= b <= 126 and b not in [ord('<'), ord('>')]:
                        mixed_parts.append(f"▁{chr(b)}")
                    else:
                        mixed_parts.append(f"▁0x{b:02x}")
                return " ".join(mixed_parts)

        # Fallback to hex
        return " ".join(f"{b:02x}" for b in packet_bytes)

    def process_single_pcap(self, pcap_path: str) -> Tuple[int, int, int, str]:
        """
        Process a single PCAP file and return packet sentences with field awareness.

        Args:
            pcap_path: Path to PCAP file

        Returns:
            Tuple of (num_packets, num_bytes, num_separators, temp_file_path)
        """
        try:
            # Read packets from PCAP
            packets = self.tokenizer.read_pcap_packets(pcap_path)

            # Create temporary file for this worker's output
            temp_fd, temp_path = tempfile.mkstemp(suffix='.txt', prefix='field_corpus_')

            num_packets = 0
            num_bytes = 0
            num_separators = 0

            with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
                for packet in packets:
                    packet_size = len(packet)

                    # Convert packet to sentence with field awareness
                    sentence = self.packet_to_sentence(packet, add_field_separators=True)
                    temp_file.write(sentence + '\n')

                    # Count separators in this packet
                    if self.include_special_tokens:
                        num_separators += sentence.count("[SEP]")

                    num_packets += 1
                    num_bytes += packet_size

            logger.debug(f"Processed {pcap_path}: {num_packets} packets, {num_separators} separators")
            return num_packets, num_bytes, num_separators, temp_path

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
        Build field-aware corpus from PCAP files using multiprocessing.

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
        logger.info(f"Field separation policy: {self.separator_config.policy}")
        logger.info(f"Max parsing depth: {self.separator_config.max_depth}")
        logger.info(f"Corpus format: {self.corpus_format}")
        logger.info(f"Include separators: {self.include_special_tokens}")

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
            for i, (num_packets, num_bytes, num_separators, temp_file) in enumerate(results.get()):
                if temp_file:
                    temp_files.append(temp_file)

                self.total_packets += num_packets
                self.total_bytes_processed += num_bytes
                self.total_separators_inserted += num_separators
                self.total_files_processed += 1

                # Log progress
                if (i + 1) % progress_interval == 0 or i == len(pcap_files) - 1:
                    elapsed = time.time() - start_time
                    files_per_sec = (i + 1) / elapsed
                    logger.info(f"Processed {i + 1}/{len(pcap_files)} files "
                                f"({files_per_sec:.1f} files/sec) - "
                                f"{self.total_packets:,} packets, "
                                f"{self.total_separators_inserted:,} separators")

        # Merge temporary files into final corpus
        self.merge_temp_files(temp_files)

        # Check for malformed packets
        self.check_malformed_packets()

        # Final statistics
        total_time = time.time() - start_time
        self.print_statistics(total_time)

    def check_malformed_packets(self) -> None:
        """Check and report malformed packet statistics."""
        if os.path.exists(self.malformed_log_path):
            try:
                with open(self.malformed_log_path, 'r') as f:
                    malformed_count = sum(1 for line in f if line.strip())
                self.malformed_packets = malformed_count

                if malformed_count > 0:
                    logger.warning(f"Found {malformed_count} malformed packets - see {self.malformed_log_path}")
            except Exception as e:
                logger.error(f"Error reading malformed packet log: {e}")

    def print_statistics(self, total_time: float) -> None:
        """Print final processing statistics."""
        print("\n" + "=" * 70)
        print("FIELD-AWARE CORPUS BUILDING COMPLETE")
        print("=" * 70)
        print(f"📁 Files processed: {self.total_files_processed:,}")
        print(f"📦 Total packets: {self.total_packets:,}")
        print(f"🔗 Field separators inserted: {self.total_separators_inserted:,}")
        print(f"⚠️  Malformed packets: {self.malformed_packets:,}")
        print(
            f"💾 Total bytes processed: {self.total_bytes_processed:,} ({self.total_bytes_processed / 1024 / 1024:.1f} MB)")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"🚀 Processing rate: {self.total_files_processed / total_time:.1f} files/sec")

        if self.total_packets > 0:
            avg_seps_per_packet = self.total_separators_inserted / self.total_packets
            print(f"📊 Average separators per packet: {avg_seps_per_packet:.1f}")
            print(f"📊 Average packets per file: {self.total_packets / max(1, self.total_files_processed):.1f}")

        if os.path.exists(self.output_file):
            output_size = os.path.getsize(self.output_file)
            print(f"📄 Output corpus size: {output_size:,} bytes ({output_size / 1024 / 1024:.1f} MB)")

            # Count lines in output file
            with open(self.output_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"📝 Corpus lines (sentences): {line_count:,}")

        print(f"\n🔧 Configuration Used:")
        print(f"   Policy: {self.separator_config.policy}")
        print(f"   Max depth: {self.separator_config.max_depth}")
        print(f"   Ethernet fields: {self.separator_config.insert_ethernet_fields}")
        print(f"   IP fields: {self.separator_config.insert_ip_fields}")
        print(f"   Transport fields: {self.separator_config.insert_transport_fields}")
        print(f"   Format: {self.corpus_format}")
        print(f"   Include separators: {self.include_special_tokens}")

        print(f"\n✅ Field-aware corpus saved to: {self.output_file}")
        print("=" * 70)


def main():
    """Main function to parse arguments and run field-aware corpus building."""
    parser = argparse.ArgumentParser(
        description="Build WordPiece corpus from PCAP files with protocol field awareness",
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
        "--format",
        choices=["tokens", "hex", "decimal", "mixed"],
        default="tokens",
        help="Output format for corpus"
    )

    parser.add_argument(
        "--policy",
        choices=["conservative", "hybrid", "aggressive"],
        default="hybrid",
        help="Field separation policy for malformed packets"
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum protocol parsing depth (2=L2, 3=L3, 4=L4)"
    )

    parser.add_argument(
        "--no-ethernet-fields",
        action="store_true",
        help="Don't separate Ethernet fields"
    )

    parser.add_argument(
        "--no-ip-fields",
        action="store_true",
        help="Don't separate IP fields"
    )

    parser.add_argument(
        "--no-transport-fields",
        action="store_true",
        help="Don't separate transport fields"
    )

    parser.add_argument(
        "--no-separators",
        action="store_true",
        help="Exclude [SEP] tokens from corpus output"
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
        "--malformed-log-dir",
        help="Directory for malformed packet logs (default: current dir)"
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

    # Create separator configuration
    separator_config = SeparatorConfig(
        policy=args.policy,
        max_depth=args.max_depth,
        insert_ethernet_fields=not args.no_ethernet_fields,
        insert_ip_fields=not args.no_ip_fields,
        insert_transport_fields=not args.no_transport_fields
    )

    # Build corpus
    logger.info("Starting field-aware PCAP corpus building...")

    corpus_builder = FieldAwareCorpusBuilder(
        output_file=args.output,
        separator_config=separator_config,
        corpus_format=args.format,
        include_special_tokens=not args.no_separators,
        malformed_log_dir=args.malformed_log_dir
    )

    try:
        corpus_builder.build_corpus(
            input_path=args.input_dir,
            num_workers=args.workers,
            recursive=not args.no_recursive,
            progress_interval=args.progress_interval
        )

        logger.info("Field-aware corpus building completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("Corpus building interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during corpus building: {e}")
        return 1


if __name__ == "__main__":
    exit(main())