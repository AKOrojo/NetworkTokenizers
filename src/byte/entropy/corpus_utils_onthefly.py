import os
import random
import sys
from itertools import cycle
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from src.byte.raw.token_pcap_byte_tokenizer import TokenPCAPByteTokenizer


class StreamingCorpusDataset(IterableDataset):
    """
    An IterableDataset that streams raw data from PCAP files.
    """

    def __init__(self, pcap_dir: str):
        super().__init__()
        self.pcap_dir = pcap_dir

        if self.pcap_dir:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                if rank == 0:
                    print("Scanning for pcap files...")
            else:
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

        else:
            self.pcap_files = []

    def _init_pcap_iterator(self):
        """Initializes the iterator for this worker's shard of PCAP files."""
        # The file list is now worker-specific, so we just shuffle and cycle it.
        worker_files = self.pcap_files.copy()
        random.shuffle(worker_files)
        return cycle(worker_files)


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

        pcap_iterator = self._init_pcap_iterator()

        while True:
            use_pcap_source = pcap_iterator is not None

            try:
                if use_pcap_source:
                    yield {"type": "pcap", "data": next(pcap_iterator)}
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