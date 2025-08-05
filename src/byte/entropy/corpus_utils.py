"""
Dataset class for loading pre-tokenized PCAP chunks.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, IterableDataset


class PreTokenizedPCAPDataset(Dataset):
    """
    A Dataset that loads pre-tokenized PCAP chunks from disk.
    This is a map-style dataset for better performance with DataLoader.
    """
    
    def __init__(self, 
                 tokenized_dir: str,
                 max_length: Optional[int] = None,
                 pad_token_id: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            tokenized_dir: Directory containing pre-tokenized chunks
            max_length: Maximum sequence length (for truncation/padding)
            pad_token_id: Padding token ID (if None, read from metadata)
        """
        self.tokenized_dir = Path(tokenized_dir)
        self.max_length = max_length
        
        # Load metadata
        metadata_path = self.tokenized_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Extract configuration
        self.tokenizer_config = self.metadata["tokenizer_config"]
        self.dataset_config = self.metadata["dataset_config"]
        self.chunks = self.metadata["chunks"]
        
        # Set pad token ID
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        else:
            self.pad_token_id = self.tokenizer_config["pad_token_id"]
        
        # Determine output format
        self.output_format = self.dataset_config["output_format"]
        
        print(f"Loaded dataset with {len(self.chunks)} chunks")
        print(f"Total tokens: {self.metadata['total_tokens']:,}")
        print(f"Chunk size: {self.dataset_config['chunk_size']}")
        print(f"Output format: {self.output_format}")
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and return a tokenized chunk."""
        chunk_info = self.chunks[idx]
        chunk_path = self.tokenized_dir / chunk_info["filename"]
        
        # Load the chunk based on format
        if self.output_format == "torch":
            tokens = torch.load(chunk_path, map_location='cpu')
        elif self.output_format == "numpy":
            tokens = torch.from_numpy(np.load(chunk_path))
        elif self.output_format == "pickle":
            with open(chunk_path, 'rb') as f:
                tokens = torch.tensor(pickle.load(f), dtype=torch.long)
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")
        
        # Apply max_length if specified
        if self.max_length is not None:
            if len(tokens) > self.max_length:
                # Truncate
                tokens = tokens[:self.max_length]
            elif len(tokens) < self.max_length:
                # Pad
                padding_length = self.max_length - len(tokens)
                padding = torch.full((padding_length,), self.pad_token_id, dtype=tokens.dtype)
                tokens = torch.cat([tokens, padding])
        
        return tokens
    
    def get_chunk_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata about a specific chunk."""
        return self.chunks[idx]
    
    def get_source_files(self) -> List[Dict[str, Any]]:
        """Get information about source PCAP files."""
        return self.metadata["source_files"]


class PreTokenizedPCAPIterableDataset(IterableDataset):
    """
    An IterableDataset that streams pre-tokenized PCAP chunks.
    Useful for very large datasets that don't fit in memory.
    """
    
    def __init__(self, 
                 tokenized_dir: str,
                 max_length: Optional[int] = None,
                 pad_token_id: Optional[int] = None,
                 shuffle: bool = True,
                 seed: Optional[int] = None):
        """
        Initialize the iterable dataset.
        
        Args:
            tokenized_dir: Directory containing pre-tokenized chunks
            max_length: Maximum sequence length (for truncation/padding)
            pad_token_id: Padding token ID (if None, read from metadata)
            shuffle: Whether to shuffle chunks
            seed: Random seed for shuffling
        """
        self.tokenized_dir = Path(tokenized_dir)
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed
        
        # Load metadata
        metadata_path = self.tokenized_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Extract configuration
        self.tokenizer_config = self.metadata["tokenizer_config"]
        self.dataset_config = self.metadata["dataset_config"]
        self.chunks = self.metadata["chunks"]
        
        # Set pad token ID
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        else:
            self.pad_token_id = self.tokenizer_config["pad_token_id"]
        
        # Determine output format
        self.output_format = self.dataset_config["output_format"]
        
        print(f"Loaded iterable dataset with {len(self.chunks)} chunks")
    
    def __iter__(self):
        """Iterate through chunks."""
        # Create a copy of chunk indices
        chunk_indices = list(range(len(self.chunks)))
        
        # Shuffle if requested
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(chunk_indices)
        
        for idx in chunk_indices:
            chunk_info = self.chunks[idx]
            chunk_path = self.tokenized_dir / chunk_info["filename"]
            
            try:
                # Load the chunk based on format
                if self.output_format == "torch":
                    tokens = torch.load(chunk_path, map_location='cpu')
                elif self.output_format == "numpy":
                    tokens = torch.from_numpy(np.load(chunk_path))
                elif self.output_format == "pickle":
                    with open(chunk_path, 'rb') as f:
                        tokens = torch.tensor(pickle.load(f), dtype=torch.long)
                else:
                    raise ValueError(f"Unknown output format: {self.output_format}")
                
                # Apply max_length if specified
                if self.max_length is not None:
                    if len(tokens) > self.max_length:
                        # Truncate
                        tokens = tokens[:self.max_length]
                    elif len(tokens) < self.max_length:
                        # Pad
                        padding_length = self.max_length - len(tokens)
                        padding = torch.full((padding_length,), self.pad_token_id, dtype=tokens.dtype)
                        tokens = torch.cat([tokens, padding])
                
                yield tokens
                
            except Exception as e:
                print(f"Error loading chunk {chunk_path}: {e}")
                continue


class PreTokenizedCollate:
    """
    Simple collate function for pre-tokenized data.
    Since chunks are already tokenized, this just stacks them.
    """
    
    def __init__(self, pad_token_id: int, max_length: Optional[int] = None):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def __call__(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Collate a batch of pre-tokenized chunks.
        
        Args:
            batch: List of tokenized sequences
            
        Returns:
            Batched tensor
        """
        if self.max_length is not None:
            # Ensure all sequences have the same length
            processed_batch = []
            for seq in batch:
                if len(seq) > self.max_length:
                    seq = seq[:self.max_length]
                elif len(seq) < self.max_length:
                    padding_length = self.max_length - len(seq)
                    padding = torch.full((padding_length,), self.pad_token_id, dtype=seq.dtype)
                    seq = torch.cat([seq, padding])
                processed_batch.append(seq)
            batch = processed_batch
        
        # Stack sequences into a batch
        return torch.stack(batch)


# Example usage function
def create_dataloader_from_pretokenized(tokenized_dir: str,
                                      batch_size: int = 32,
                                      max_length: Optional[int] = None,
                                      shuffle: bool = True,
                                      num_workers: int = 0,
                                      use_iterable: bool = False) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from pre-tokenized PCAP data.
    
    Args:
        tokenized_dir: Directory containing pre-tokenized chunks
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        use_iterable: Whether to use IterableDataset
        
    Returns:
        DataLoader instance
    """
    if use_iterable:
        dataset = PreTokenizedPCAPIterableDataset(
            tokenized_dir=tokenized_dir,
            max_length=max_length,
            shuffle=shuffle
        )
        # For IterableDataset, shuffle should be False in DataLoader
        dataloader_shuffle = False
    else:
        dataset = PreTokenizedPCAPDataset(
            tokenized_dir=tokenized_dir,
            max_length=max_length
        )
        dataloader_shuffle = shuffle
    
    # Create collate function
    collate_fn = PreTokenizedCollate(
        pad_token_id=dataset.pad_token_id,
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