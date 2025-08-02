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

from others.pcap_text_tokenizer import PCAPTextTokenizer


class StreamingCorpusDataset(IterableDataset):
    """
    An IterableDataset that streams raw data from PCAP files and/or the DCLM dataset.
    It's designed to be used with a DataLoader to enable multiprocess data loading.
    """
    def __init__(self, pcap_dir: str, pcap_ratio: float = 0.5):
        super().__init__()
        self.pcap_dir = pcap_dir
        self.pcap_ratio = pcap_ratio
        # Pre-fetch and sort the file list once to ensure order is always the same
        if self.pcap_dir:
            print("Scanning for pcap files...")
            all_pcap_files = sorted(
                list(Path(self.pcap_dir).glob("**/*.pcap")) + list(Path(self.pcap_dir).glob("**/*.pcapng")))

            if not all_pcap_files:
                raise FileNotFoundError(f"No .pcap or .pcapng files found in {self.pcap_dir}")

            # --- THIS IS THE FIX ---
            size_limit_bytes = 100 * 1024 * 1024  # 100 MB

            self.pcap_files = [
                f for f in all_pcap_files
                if os.path.getsize(f) < size_limit_bytes
            ]

            num_excluded = len(all_pcap_files) - len(self.pcap_files)
            print(f"Found {len(all_pcap_files)} total files.")
            print(f"Excluding {num_excluded} files larger than {size_limit_bytes / (1024 * 1024)} MB.")
            print(f"Using {len(self.pcap_files)} files for training.")
            # --- END OF FIX ---

        else:
            self.pcap_files = []

    def _init_pcap_iterator(self):
        """Initializes the iterator for PCAP files."""
        # Use a copy of the pre-shuffled list for each iterator
        shuffled_files = self.pcap_files.copy()
        random.shuffle(shuffled_files) # The random state is now seeded per worker
        return cycle(shuffled_files)

    @staticmethod
    def _init_text_iterator():
        """Initializes the iterator for the text dataset."""
        text_stream = load_dataset(
            "mlfoundations/dclm-baseline-1.0",
            streaming=True,
            split="train"
        ).shuffle(seed=42, buffer_size=10_000) # datasets.shuffle is DDP-safe
        return iter(text_stream)

    def __iter__(self):
        """The core logic that yields one raw data sample at a time."""
        # --- CRITICAL FIX: Seed each worker process for deterministic behavior ---
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Ensure each worker uses a different seed
            seed = worker_info.seed + dist.get_rank() * 1000  # Add rank offset
        else:
            seed = 42 + dist.get_rank() * 1000
        random.seed(seed)

        pcap_iterator = None
        if self.pcap_ratio > 0.0 and self.pcap_files:
            pcap_iterator = self._init_pcap_iterator()

        text_iterator = None
        if self.pcap_ratio < 1.0:
            text_iterator = self._init_text_iterator()

        while True:
            # Determine source based on the seeded random state
            use_pcap_source = pcap_iterator is not None and random.random() < self.pcap_ratio

            try:
                if use_pcap_source:
                    yield {"type": "pcap", "data": next(pcap_iterator)}
                elif text_iterator is not None:
                    sample = next(text_iterator)
                    yield {"type": "text", "data": sample["text"]}
                else:
                    # If only one source is configured and it's exhausted
                    break
            except StopIteration:
                break # Stop if any iterator is exhausted


class TokenizerCollate:
    def __init__(self, tokenizer: PCAPTextTokenizer, max_len: int):
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
                    # The raw data is the file path
                    pcap_file_path = sample["data"]
                    tokens = self.tokenizer.tokenize_pcap(pcap_file_path)
                elif sample["type"] == "text":
                    tokens = self.tokenizer.tokenize(sample["data"])

                if tokens:
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            except Exception as e:
                # --- THIS IS THE KEY CHANGE ---
                # Log the source of the error to identify the problematic file.
                problematic_source = sample.get("data", "Unknown source")
                print(
                    f"\n!!! Warning: Error processing sample from source: {problematic_source}. "
                    f"Replacing with padding. Error: {e}\n", file=sys.stderr
                )
                # The rest of the logic correctly handles this by creating a padded sequence

            # Truncate and pad the sequence
            truncated_seq = token_ids[:self.max_len]
            padded_seq = truncated_seq + [self.pad_id] * (self.max_len - len(truncated_seq))
            padded_batch.append(padded_seq)

        return torch.tensor(padded_batch, dtype=torch.long)