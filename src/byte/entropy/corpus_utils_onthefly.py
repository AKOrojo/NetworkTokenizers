import os
import sys
from itertools import cycle
from pathlib import Path
import random
from typing import List

from datasets import load_dataset
import torch.distributed as dist
from torch.utils.data import IterableDataset
import torch

# Assuming TokenPCAPByteTokenizer is in this path
from src.byte.raw.token_pcap_byte_tokenizer import TokenPCAPByteTokenizer


class StreamingCorpusDataset(IterableDataset):
    """
    An IterableDataset that streams raw data from PCAP files and/or the DCLM dataset.
    It's designed for multi-process data loading in a distributed setting.
    """

    def __init__(self, pcap_dir: str, pcap_ratio: float = 0.5):
        super().__init__()
        self.pcap_dir = pcap_dir
        self.pcap_ratio = pcap_ratio

        if self.pcap_dir:
            print("Scanning for pcap files...")
            all_pcap_files = sorted(
                list(Path(self.pcap_dir).glob("**/*.pcap")) + list(Path(self.pcap_dir).glob("**/*.pcapng")))

            if not all_pcap_files:
                raise FileNotFoundError(f"No .pcap or .pcapng files found in {self.pcap_dir}")

            # Filter out very large files to prevent OOM issues during tokenization
            size_limit_bytes = 100 * 1024 * 1024  # 100 MB
            self.pcap_files = [
                f for f in all_pcap_files
                if os.path.getsize(f) < size_limit_bytes
            ]

            # --- START: DISTRIBUTED SAMPLING FIX ---
            # Shard the dataset for each distributed process
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()

                # Each rank gets a unique slice of the file list
                files_for_rank = self.pcap_files[rank::world_size]

                print(
                    f"[Rank {rank}] Original files: {len(self.pcap_files)}. "
                    f"Files for this rank: {len(files_for_rank)}"
                )
                self.pcap_files = files_for_rank
            # --- END: DISTRIBUTED SAMPLING FIX ---

        else:
            self.pcap_files = []

    def _init_pcap_iterator(self):
        """Initializes the iterator for this worker's shard of PCAP files."""
        # The file list is now worker-specific, so we just shuffle and cycle it.
        worker_files = self.pcap_files.copy()
        random.shuffle(worker_files)
        return cycle(worker_files)

    @staticmethod
    def _init_text_iterator():
        """Initializes the iterator for the text dataset."""
        # `datasets` handles DDP sharding automatically when streaming
        text_stream = load_dataset(
            "mlfoundations/dclm-baseline-1.0",
            streaming=True,
            split="train"
        ).shuffle(seed=42, buffer_size=10_000)
        return iter(text_stream)

    def __iter__(self):
        """The core logic that yields one raw data sample at a time."""
        worker_info = torch.utils.data.get_worker_info()

        # Seed each worker process for reproducible shuffling within its shard
        if worker_info is not None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            seed = worker_info.seed + rank * 1000
        else:
            seed = 42

        random.seed(seed)

        pcap_iterator = None
        if self.pcap_ratio > 0.0 and self.pcap_files:
            pcap_iterator = self._init_pcap_iterator()

        text_iterator = None
        if self.pcap_ratio < 1.0:
            text_iterator = self._init_text_iterator()

        while True:
            use_pcap_source = pcap_iterator is not None and random.random() < self.pcap_ratio

            try:
                if use_pcap_source:
                    yield {"type": "pcap", "data": next(pcap_iterator)}
                elif text_iterator is not None:
                    # The huggingface datasets library handles DDP for the text stream
                    sample = next(text_iterator)
                    yield {"type": "text", "data": sample["text"]}
                else:
                    break
            except StopIteration:
                break


class TokenizerCollate:
    def __init__(self, tokenizer: TokenPCAPByteTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch: List[dict]) -> torch.Tensor:
        padded_batch = []
        for sample in batch:
            token_ids = []
            try:
                tokens = []
                if sample["type"] == "pcap":
                    pcap_file_path = sample["data"]
                    tokens = self.tokenizer.tokenize_pcap(pcap_file_path)
                elif sample["type"] == "text":
                    tokens = self.tokenizer.tokenize(sample["data"])

                if tokens:
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                    # Validate token IDs are in valid range
                    if any(tid < 0 or tid >= self.tokenizer.vocab_size for tid in token_ids):
                        print(
                            f"Warning: Invalid token IDs found: {[tid for tid in token_ids if tid < 0 or tid >= self.tokenizer.vocab_size]}")
                        token_ids = [tid for tid in token_ids if 0 <= tid < self.tokenizer.vocab_size]

            except Exception as e:
                problematic_source = sample.get("data", "Unknown source")
                print(
                    f"\n!!! Warning: Error processing sample from source: {problematic_source}. "
                    f"Replacing with padding. Error: {e}\n", file=sys.stderr
                )

            truncated_seq = token_ids[:self.max_len]
            padded_seq = truncated_seq + [self.pad_id] * (self.max_len - len(truncated_seq))
            padded_batch.append(padded_seq)

        return torch.tensor(padded_batch, dtype=torch.long)