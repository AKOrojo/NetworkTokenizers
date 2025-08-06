"""
Dataset classes for loading raw PCAP data and tokenizing on-the-fly.
"""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue, Empty
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, IterableDataset


class RawPCAPDataset(Dataset):
    """
    A Dataset that loads raw PCAP chunks and tokenizes them on-the-fly.
    This is a map-style dataset for better performance with DataLoader.
    """

    def __init__(self,
                 raw_data_dir: str,
                 tokenizer,
                 chunk_size: int = 8192,
                 max_length: Optional[int] = None,
                 file_extensions: List[str] = ['.pcap', '.pcapng', '.bin'],
                 cache_tokenized: bool = False,
                 cache_size: int = 1000):
        """
        Initialize the dataset.

        Args:
            raw_data_dir: Directory containing raw PCAP files
            tokenizer: Tokenizer instance for on-the-fly tokenization
            chunk_size: Size of chunks to create from each file
            max_length: Maximum sequence length (for truncation/padding)
            file_extensions: List of file extensions to include
            cache_tokenized: Whether to cache tokenized chunks in memory
            cache_size: Maximum number of chunks to cache
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.file_extensions = file_extensions
        self.cache_tokenized = cache_tokenized
        self.cache_size = cache_size

        # Initialize cache if enabled
        if self.cache_tokenized:
            self.cache = {}
            self.cache_access_order = []
            self.cache_lock = threading.Lock()

        # Discover all PCAP files
        self.files = []
        self._discover_files()

        # Create chunks metadata
        self.chunks = []
        self._create_chunk_metadata()

        print(f"Loaded dataset with {len(self.files)} files")
        print(f"Total chunks: {len(self.chunks)}")
        print(f"Chunk size: {self.chunk_size} bytes")
        if self.cache_tokenized:
            print(f"Tokenization cache enabled (max {self.cache_size} chunks)")

    def _discover_files(self):
        """Discover all PCAP files in the directory."""
        for ext in self.file_extensions:
            pattern = f"**/*{ext}"
            files = list(self.raw_data_dir.glob(pattern))
            self.files.extend(files)

        # Sort for reproducibility
        self.files.sort()

        if not self.files:
            raise ValueError(f"No files found with extensions {self.file_extensions} in {self.raw_data_dir}")

    def _create_chunk_metadata(self):
        """Create metadata for all chunks."""
        for file_idx, file_path in enumerate(self.files):
            try:
                file_size = file_path.stat().st_size
                num_chunks = max(1, (file_size + self.chunk_size - 1) // self.chunk_size)

                for chunk_idx in range(num_chunks):
                    start_offset = chunk_idx * self.chunk_size
                    end_offset = min(start_offset + self.chunk_size, file_size)
                    actual_size = end_offset - start_offset

                    self.chunks.append({
                        'file_idx': file_idx,
                        'file_path': file_path,
                        'chunk_idx': chunk_idx,
                        'start_offset': start_offset,
                        'end_offset': end_offset,
                        'size': actual_size
                    })
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

    def _load_raw_chunk(self, chunk_info: Dict[str, Any]) -> bytes:
        """Load raw bytes from a chunk."""
        file_path = chunk_info['file_path']
        start_offset = chunk_info['start_offset']
        size = chunk_info['size']

        try:
            with open(file_path, 'rb') as f:
                f.seek(start_offset)
                data = f.read(size)
            return data
        except Exception as e:
            print(f"Error loading chunk from {file_path}: {e}")
            return b''

    def _tokenize_chunk(self, raw_data: bytes) -> torch.Tensor:
        """Tokenize raw data using the tokenizer."""
        try:
            # Convert bytes to list for tokenizer
            byte_list = list(raw_data)

            # Tokenize
            tokens = self.tokenizer.encode(byte_list)

            # Convert to tensor
            tokens_tensor = torch.tensor(tokens, dtype=torch.long)

            return tokens_tensor
        except Exception as e:
            print(f"Error tokenizing chunk: {e}")
            # Return empty tensor on error
            return torch.tensor([], dtype=torch.long)

    def _get_from_cache(self, chunk_idx: int) -> Optional[torch.Tensor]:
        """Get tokenized chunk from cache if available."""
        if not self.cache_tokenized:
            return None

        with self.cache_lock:
            if chunk_idx in self.cache:
                # Update access order
                self.cache_access_order.remove(chunk_idx)
                self.cache_access_order.append(chunk_idx)
                return self.cache[chunk_idx].clone()
        return None

    def _add_to_cache(self, chunk_idx: int, tokens: torch.Tensor):
        """Add tokenized chunk to cache."""
        if not self.cache_tokenized:
            return

        with self.cache_lock:
            # Remove oldest items if cache is full
            while len(self.cache) >= self.cache_size:
                oldest_idx = self.cache_access_order.pop(0)
                del self.cache[oldest_idx]

            # Add new item
            self.cache[chunk_idx] = tokens.clone()
            self.cache_access_order.append(chunk_idx)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and tokenize a chunk on-the-fly."""
        # Try cache first
        cached_tokens = self._get_from_cache(idx)
        if cached_tokens is not None:
            tokens = cached_tokens
        else:
            # Load and tokenize
            chunk_info = self.chunks[idx]
            raw_data = self._load_raw_chunk(chunk_info)
            tokens = self._tokenize_chunk(raw_data)

            # Add to cache
            self._add_to_cache(idx, tokens)

        # Apply max_length if specified
        if self.max_length is not None:
            if len(tokens) > self.max_length:
                # Truncate
                tokens = tokens[:self.max_length]
            elif len(tokens) < self.max_length:
                # Pad
                padding_length = self.max_length - len(tokens)
                padding = torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=tokens.dtype)
                tokens = torch.cat([tokens, padding])

        return tokens

    def get_chunk_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata about a specific chunk."""
        return self.chunks[idx]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache_tokenized:
            return {"cache_enabled": False}

        with self.cache_lock:
            return {
                "cache_enabled": True,
                "cache_size": len(self.cache),
                "cache_max_size": self.cache_size,
                "cache_hit_ratio": getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_attempts', 1), 1)
            }


class RawPCAPIterableDataset(IterableDataset):
    """
    An IterableDataset that streams raw PCAP data and tokenizes on-the-fly.
    Useful for very large datasets.
    """

    def __init__(self,
                 raw_data_dir: str,
                 tokenizer,
                 chunk_size: int = 8192,
                 max_length: Optional[int] = None,
                 file_extensions: List[str] = ['.pcap', '.pcapng', '.bin'],
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 buffer_size: int = 100):
        """
        Initialize the iterable dataset.

        Args:
            raw_data_dir: Directory containing raw PCAP files
            tokenizer: Tokenizer instance for on-the-fly tokenization
            chunk_size: Size of chunks to create from each file
            max_length: Maximum sequence length (for truncation/padding)
            file_extensions: List of file extensions to include
            shuffle: Whether to shuffle chunks
            seed: Random seed for shuffling
            buffer_size: Number of chunks to buffer for shuffling
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.file_extensions = file_extensions
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = buffer_size

        # Discover all PCAP files
        self.files = []
        self._discover_files()

        print(f"Loaded iterable dataset with {len(self.files)} files")

    def _discover_files(self):
        """Discover all PCAP files in the directory."""
        for ext in self.file_extensions:
            pattern = f"**/*{ext}"
            files = list(self.raw_data_dir.glob(pattern))
            self.files.extend(files)

        # Sort for reproducibility
        self.files.sort()

        if not self.files:
            raise ValueError(f"No files found with extensions {self.file_extensions} in {self.raw_data_dir}")

    def _generate_chunks(self):
        """Generate chunks from all files."""
        # Create a list of all chunks
        all_chunks = []

        for file_idx, file_path in enumerate(self.files):
            try:
                file_size = file_path.stat().st_size
                num_chunks = max(1, (file_size + self.chunk_size - 1) // self.chunk_size)

                for chunk_idx in range(num_chunks):
                    start_offset = chunk_idx * self.chunk_size
                    end_offset = min(start_offset + self.chunk_size, file_size)
                    actual_size = end_offset - start_offset

                    all_chunks.append({
                        'file_path': file_path,
                        'start_offset': start_offset,
                        'size': actual_size
                    })
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        # Shuffle if requested
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(all_chunks)

        return all_chunks

    def _load_and_tokenize_chunk(self, chunk_info: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Load and tokenize a chunk."""
        try:
            # Load raw data
            file_path = chunk_info['file_path']
            start_offset = chunk_info['start_offset']
            size = chunk_info['size']

            with open(file_path, 'rb') as f:
                f.seek(start_offset)
                raw_data = f.read(size)

            # Tokenize
            byte_list = list(raw_data)
            tokens = self.tokenizer.encode(byte_list)
            tokens_tensor = torch.tensor(tokens, dtype=torch.long)

            # Apply max_length if specified
            if self.max_length is not None:
                if len(tokens_tensor) > self.max_length:
                    tokens_tensor = tokens_tensor[:self.max_length]
                elif len(tokens_tensor) < self.max_length:
                    padding_length = self.max_length - len(tokens_tensor)
                    padding = torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=tokens_tensor.dtype)
                    tokens_tensor = torch.cat([tokens_tensor, padding])

            return tokens_tensor

        except Exception as e:
            print(f"Error loading/tokenizing chunk: {e}")
            return None

    def __iter__(self):
        """Iterate through chunks."""
        chunks = self._generate_chunks()

        for chunk_info in chunks:
            tokens = self._load_and_tokenize_chunk(chunk_info)
            if tokens is not None:
                yield tokens


class OnTheFlyCollate:
    """
    Collate function for on-the-fly tokenized data.
    Handles variable-length sequences and batching.
    """

    def __init__(self, pad_token_id: int, max_length: Optional[int] = None):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Collate a batch of tokenized sequences.

        Args:
            batch: List of tokenized sequences

        Returns:
            Batched tensor
        """
        # Filter out empty sequences
        batch = [seq for seq in batch if len(seq) > 0]

        if not batch:
            # Return empty batch if all sequences were empty
            empty_seq = torch.full((1, 1), self.pad_token_id, dtype=torch.long)
            return empty_seq

        # Determine target length
        if self.max_length is not None:
            target_length = self.max_length
        else:
            target_length = max(len(seq) for seq in batch)

        # Pad/truncate all sequences to target length
        processed_batch = []
        for seq in batch:
            if len(seq) > target_length:
                seq = seq[:target_length]
            elif len(seq) < target_length:
                padding_length = target_length - len(seq)
                padding = torch.full((padding_length,), self.pad_token_id, dtype=seq.dtype)
                seq = torch.cat([seq, padding])
            processed_batch.append(seq)

        # Stack sequences into a batch
        return torch.stack(processed_batch)


class AsyncTokenizer:
    """
    Asynchronous tokenizer that pre-processes chunks in background threads.
    Useful for improving training throughput with on-the-fly tokenization.
    """

    def __init__(self, dataset, tokenizer, buffer_size: int = 50, num_workers: int = 4):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.buffer = Queue(maxsize=buffer_size)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.current_idx = 0
        self.shutdown = False

        # Start background workers
        self._start_workers()

    def _worker(self):
        """Background worker that tokenizes chunks."""
        while not self.shutdown:
            try:
                if self.current_idx >= len(self.dataset):
                    time.sleep(0.1)
                    continue

                idx = self.current_idx
                self.current_idx += 1

                # Load and tokenize chunk
                chunk_info = self.dataset.chunks[idx]
                raw_data = self.dataset._load_raw_chunk(chunk_info)
                tokens = self.dataset._tokenize_chunk(raw_data)

                # Add to buffer (this will block if buffer is full)
                self.buffer.put((idx, tokens), timeout=1.0)

            except Exception as e:
                print(f"Error in async tokenizer worker: {e}")
                time.sleep(0.1)

    def _start_workers(self):
        """Start background workers."""
        for _ in range(self.num_workers):
            self.executor.submit(self._worker)

    def get_next(self, timeout: float = 1.0) -> Optional[torch.Tensor]:
        """Get next tokenized chunk."""
        try:
            idx, tokens = self.buffer.get(timeout=timeout)
            return tokens
        except Empty:
            return None

    def shutdown_workers(self):
        """Shutdown background workers."""
        self.shutdown = True
        self.executor.shutdown(wait=True)


# Example usage function
def create_dataloader_from_raw(raw_data_dir: str,
                               tokenizer,
                               batch_size: int = 32,
                               chunk_size: int = 8192,
                               max_length: Optional[int] = None,
                               shuffle: bool = True,
                               num_workers: int = 0,
                               use_iterable: bool = False,
                               cache_tokenized: bool = False,
                               cache_size: int = 1000) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from raw PCAP data with on-the-fly tokenization.

    Args:
        raw_data_dir: Directory containing raw PCAP files
        tokenizer: Tokenizer instance
        batch_size: Batch size
        chunk_size: Size of chunks to create from files
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        use_iterable: Whether to use IterableDataset
        cache_tokenized: Whether to cache tokenized chunks
        cache_size: Maximum number of chunks to cache

    Returns:
        DataLoader instance
    """
    if use_iterable:
        dataset = RawPCAPIterableDataset(
            raw_data_dir=raw_data_dir,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            max_length=max_length,
            shuffle=shuffle
        )
        # For IterableDataset, shuffle should be False in DataLoader
        dataloader_shuffle = False
    else:
        dataset = RawPCAPDataset(
            raw_data_dir=raw_data_dir,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            max_length=max_length,
            cache_tokenized=cache_tokenized,
            cache_size=cache_size
        )
        dataloader_shuffle = shuffle

    # Create collate function
    collate_fn = OnTheFlyCollate(
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )