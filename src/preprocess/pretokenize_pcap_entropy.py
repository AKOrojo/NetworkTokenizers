#!/usr/bin/env python3
"""
Pre-tokenize PCAP files into chunks for efficient training.
"""

import argparse
import json
import multiprocessing as mp
import pickle
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.byte.raw.token_pcap_byte_tokenizer import TokenPCAPByteTokenizer


def process_pcap_file_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function to process a single PCAP file.
    Returns metadata about the processed file and its chunks.
    """
    (file_path, output_dir, chunk_size, output_format, 
     add_packet_separators, max_file_size_bytes, start_chunk_idx) = args
    
    try:
        # Initialize tokenizer in worker process
        tokenizer = TokenPCAPByteTokenizer()
        
        # Check file size
        if file_path.stat().st_size > max_file_size_bytes:
            return {
                "success": False,
                "file_path": str(file_path),
                "error": "File too large",
                "chunks": [],
                "file_info": None
            }
        
        # Tokenize the file
        try:
            token_ids = tokenizer.tokenize_pcap_to_ids(
                file_path,
                add_packet_separators=add_packet_separators,
                add_bos=False,
                add_eos=False
            )
        except Exception as e:
            return {
                "success": False,
                "file_path": str(file_path),
                "error": f"Tokenization failed: {str(e)}",
                "chunks": [],
                "file_info": None
            }
        
        if not token_ids:
            return {
                "success": False,
                "file_path": str(file_path),
                "error": "No tokens generated",
                "chunks": [],
                "file_info": None
            }
        
        # Create chunks
        chunks = []
        chunk_idx = start_chunk_idx
        
        for i in range(0, len(token_ids), chunk_size):
            chunk = token_ids[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append({
                    "chunk_idx": chunk_idx,
                    "data": chunk,
                    "num_tokens": len(chunk)
                })
                chunk_idx += 1
        
        # Save chunks to disk
        saved_chunks = []
        for chunk_data in chunks:
            chunk_filename = get_chunk_filename(chunk_data["chunk_idx"], output_format)
            chunk_path = output_dir / chunk_filename
            
            # Save chunk
            if output_format == "torch":
                torch.save(torch.tensor(chunk_data["data"], dtype=torch.long), chunk_path)
            elif output_format == "numpy":
                np.save(chunk_path, np.array(chunk_data["data"], dtype=np.int64))
            elif output_format == "pickle":
                with open(chunk_path, 'wb') as f:
                    pickle.dump(chunk_data["data"], f)
            
            saved_chunks.append({
                "chunk_idx": chunk_data["chunk_idx"],
                "filename": chunk_filename,
                "source_file": str(file_path),
                "num_tokens": chunk_data["num_tokens"]
            })
        
        # File info
        file_info = {
            "file_path": str(file_path),
            "file_size_bytes": file_path.stat().st_size,
            "num_tokens": len(token_ids),
            "chunk_start_idx": start_chunk_idx,
            "num_chunks": len(saved_chunks),
        }
        
        return {
            "success": True,
            "file_path": str(file_path),
            "error": None,
            "chunks": saved_chunks,
            "file_info": file_info,
            "total_tokens": len(token_ids)
        }
        
    except Exception as e:
        return {
            "success": False,
            "file_path": str(file_path),
            "error": f"Unexpected error: {str(e)}\n{traceback.format_exc()}",
            "chunks": [],
            "file_info": None
        }


def get_chunk_filename(chunk_idx: int, output_format: str) -> str:
    """Get filename for a chunk."""
    if output_format == "torch":
        return f"chunk_{chunk_idx:08d}.pt"
    elif output_format == "numpy":
        return f"chunk_{chunk_idx:08d}.npy"
    elif output_format == "pickle":
        return f"chunk_{chunk_idx:08d}.pkl"


class PCAPPreTokenizer:
    """Pre-tokenizes PCAP files and saves them in chunks."""
    
    def __init__(self, 
                 tokenizer: TokenPCAPByteTokenizer,
                 chunk_size: int = 512,
                 max_file_size_gb: float = 100.0,
                 output_format: str = "torch",
                 num_processes: Optional[int] = None):
        """
        Initialize the pre-tokenizer.
        
        Args:
            tokenizer: The PCAP tokenizer to use
            chunk_size: Maximum tokens per chunk
            max_file_size_gb: Maximum file size to process (in GB)
            output_format: Output format ('torch', 'numpy', or 'pickle')
            num_processes: Number of processes to use (None = auto-detect)
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_file_size_bytes = int(max_file_size_gb * 1024 * 1024 * 1024)
        self.output_format = output_format
        self.num_processes = num_processes or max(1, mp.cpu_count() - 1)
        
        if output_format not in ['torch', 'numpy', 'pickle']:
            raise ValueError("output_format must be 'torch', 'numpy', or 'pickle'")
        
        print(f"Using {self.num_processes} processes for parallel processing")
    
    def find_pcap_files(self, pcap_dir: str) -> List[Path]:
        """Find all PCAP files in directory."""
        pcap_dir = Path(pcap_dir)
        
        print(f"Scanning for PCAP files in {pcap_dir}...")
        all_files = []
        all_files.extend(pcap_dir.glob("**/*.pcap"))
        all_files.extend(pcap_dir.glob("**/*.pcapng"))
        
        if not all_files:
            raise FileNotFoundError(f"No .pcap or .pcapng files found in {pcap_dir}")
        
        # Filter by file size
        valid_files = []
        excluded_count = 0
        
        for file_path in all_files:
            try:
                file_size = file_path.stat().st_size
                if file_size <= self.max_file_size_bytes:
                    valid_files.append(file_path)
                else:
                    excluded_count += 1
            except OSError:
                print(f"Warning: Could not get size for {file_path}")
                excluded_count += 1
        
        print(f"Found {len(all_files)} total files")
        print(f"Excluded {excluded_count} files (too large or inaccessible)")
        print(f"Processing {len(valid_files)} files")
        
        return sorted(valid_files)
    
    def estimate_chunks_per_file(self, pcap_files: List[Path], sample_size: int = 100) -> float:
        """Estimate average chunks per file by sampling."""
        if len(pcap_files) <= sample_size:
            return 1.0  # Conservative estimate
        
        print(f"Estimating chunks per file using {sample_size} samples...")
        sample_files = pcap_files[:sample_size]
        total_chunks = 0
        valid_samples = 0
        
        for file_path in tqdm(sample_files, desc="Sampling files"):
            try:
                if file_path.stat().st_size > self.max_file_size_bytes:
                    continue
                    
                token_ids = self.tokenizer.tokenize_pcap_to_ids(
                    file_path, add_packet_separators=True, add_bos=False, add_eos=False
                )
                if token_ids:
                    num_chunks = (len(token_ids) + self.chunk_size - 1) // self.chunk_size
                    total_chunks += num_chunks
                    valid_samples += 1
            except Exception:
                continue
        
        if valid_samples == 0:
            return 1.0
        
        avg_chunks = total_chunks / valid_samples
        print(f"Estimated average chunks per file: {avg_chunks:.2f}")
        return avg_chunks

    def prepare_work_items(self, pcap_files: List[Path], output_dir: Path, 
                          add_packet_separators: bool) -> List[Tuple]:
        """Prepare work items for multiprocessing with proper chunk index allocation."""
        
        # Estimate total chunks needed
        avg_chunks_per_file = self.estimate_chunks_per_file(pcap_files)
        
        # Allocate chunk indices
        work_items = []
        current_chunk_idx = 0
        
        for file_path in pcap_files:
            # Estimate chunks for this file (conservative allocation)
            estimated_chunks = max(1, int(avg_chunks_per_file * 2))  # 2x buffer
            
            work_item = (
                file_path,
                output_dir,
                self.chunk_size,
                self.output_format,
                add_packet_separators,
                self.max_file_size_bytes,
                current_chunk_idx
            )
            work_items.append(work_item)
            current_chunk_idx += estimated_chunks
        
        return work_items

    def pretokenize_dataset(self, 
                          pcap_dir: str, 
                          output_dir: str,
                          add_packet_separators: bool = True,
                          save_progress_every: int = 1000) -> Dict[str, Any]:
        """
        Pre-tokenize all PCAP files in directory using multiprocessing.
        
        Args:
            pcap_dir: Directory containing PCAP files
            output_dir: Directory to save tokenized chunks
            add_packet_separators: Whether to add packet separator tokens
            save_progress_every: Save progress metadata every N files
            
        Returns:
            Metadata about the tokenized dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all PCAP files
        pcap_files = self.find_pcap_files(pcap_dir)
        
        print(f"Preparing work for {len(pcap_files)} files across {self.num_processes} processes...")
        
        # Prepare work items
        work_items = self.prepare_work_items(pcap_files, output_dir, add_packet_separators)
        
        # Initialize metadata
        metadata = {
            "tokenizer_config": {
                "vocab_size": self.tokenizer.vocab_size,
                "pad_token_id": self.tokenizer.pad_token_id,
                "unk_token_id": self.tokenizer.unk_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "bos_token_id": self.tokenizer.bos_token_id,
                "pkt_token_id": self.tokenizer.pkt_token_id,
            },
            "dataset_config": {
                "chunk_size": self.chunk_size,
                "add_packet_separators": add_packet_separators,
                "output_format": self.output_format,
                "max_file_size_gb": self.max_file_size_bytes / (1024**3),
                "num_processes": self.num_processes,
            },
            "source_files": [],
            "chunks": [],
            "total_chunks": 0,
            "total_tokens": 0,
            "processing_stats": {
                "successful_files": 0,
                "failed_files": 0,
                "start_time": time.time(),
            }
        }
        
        # Process files in parallel
        print(f"Starting parallel processing with {self.num_processes} processes...")
        
        with Pool(processes=self.num_processes) as pool:
            # Use imap for progress tracking
            results = []
            with tqdm(total=len(work_items), desc="Processing files") as pbar:
                for result in pool.imap(process_pcap_file_worker, work_items):
                    results.append(result)
                    pbar.update(1)
                    
                    # Save progress periodically
                    if len(results) % save_progress_every == 0:
                        self._save_progress_metadata(results, metadata, output_dir)
        
        print("\nCollecting results and finalizing metadata...")
        
        # Collect all results
        all_chunks = []
        successful_files = 0
        failed_files = 0
        total_tokens = 0
        
        for result in tqdm(results, desc="Collecting results"):
            if result["success"]:
                successful_files += 1
                metadata["source_files"].append(result["file_info"])
                all_chunks.extend(result["chunks"])
                total_tokens += result["total_tokens"]
            else:
                failed_files += 1
                print(f"Failed: {result['file_path']} - {result['error']}")
        
        # Sort chunks by chunk_idx to ensure proper ordering
        all_chunks.sort(key=lambda x: x["chunk_idx"])
        
        # Renumber chunks to be contiguous (in case of gaps from estimation)
        final_chunks = []
        for new_idx, chunk in enumerate(all_chunks):
            chunk_copy = chunk.copy()
            
            # Rename the file if necessary
            old_filename = chunk["filename"]
            new_filename = get_chunk_filename(new_idx, self.output_format)
            
            if old_filename != new_filename:
                old_path = output_dir / old_filename
                new_path = output_dir / new_filename
                if old_path.exists():
                    old_path.rename(new_path)
                chunk_copy["filename"] = new_filename
            
            chunk_copy["chunk_idx"] = new_idx
            final_chunks.append(chunk_copy)
        
        metadata["chunks"] = final_chunks
        metadata["total_chunks"] = len(final_chunks)
        metadata["total_tokens"] = total_tokens
        metadata["processing_stats"].update({
            "successful_files": successful_files,
            "failed_files": failed_files,
            "end_time": time.time(),
            "total_time": time.time() - metadata["processing_stats"]["start_time"]
        })
        
        # Save final metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nPre-tokenization complete!")
        print(f"Successful files: {successful_files}")
        print(f"Failed files: {failed_files}")
        print(f"Total chunks created: {metadata['total_chunks']}")
        print(f"Total tokens: {metadata['total_tokens']:,}")
        print(f"Average tokens per chunk: {metadata['total_tokens'] / max(1, metadata['total_chunks']):.1f}")
        print(f"Processing time: {metadata['processing_stats']['total_time']:.1f} seconds")
        print(f"Output directory: {output_dir}")
        print(f"Metadata saved to: {metadata_path}")
        
        return metadata
    
    def _save_progress_metadata(self, results: List[Dict], metadata: Dict, output_dir: Path):
        """Save progress metadata during processing."""
        try:
            progress_metadata = metadata.copy()
            progress_metadata["partial_results"] = len(results)
            progress_metadata["timestamp"] = time.time()
            
            progress_path = output_dir / "progress.json"
            with open(progress_path, 'w') as f:
                json.dump(progress_metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress metadata: {e}")


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize PCAP files for training with multiprocessing")
    parser.add_argument("pcap_dir", help="Directory containing PCAP files")
    parser.add_argument("output_dir", help="Directory to save tokenized chunks")
    parser.add_argument("--chunk-size", type=int, default=2048, 
                       help="Maximum tokens per chunk (default: 2048)")
    parser.add_argument("--max-file-size-gb", type=float, default=10.0,
                       help="Maximum file size to process in GB (default: 10)")
    parser.add_argument("--output-format", choices=["torch", "numpy", "pickle"], 
                       default="numpy", help="Output format (default: numpy)")
    parser.add_argument("--no-packet-separators", action="store_true",
                       help="Don't add packet separator tokens")
    parser.add_argument("--num-processes", type=int, default=None,
                       help="Number of processes to use (default: auto-detect)")
    parser.add_argument("--save-progress-every", type=int, default=1000,
                       help="Save progress metadata every N files (default: 1000)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_processes is not None and args.num_processes < 1:
        print("Error: num_processes must be >= 1")
        return
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = TokenPCAPByteTokenizer()
    
    # Initialize pre-tokenizer
    pre_tokenizer = PCAPPreTokenizer(
        tokenizer=tokenizer,
        chunk_size=args.chunk_size,
        max_file_size_gb=args.max_file_size_gb,
        output_format=args.output_format,
        num_processes=args.num_processes
    )
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Input directory: {args.pcap_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Max file size: {args.max_file_size_gb} GB")
    print(f"  Output format: {args.output_format}")
    print(f"  Processes: {pre_tokenizer.num_processes}")
    print(f"  Packet separators: {not args.no_packet_separators}")
    
    # Pre-tokenize dataset
    start_time = time.time()
    metadata = pre_tokenizer.pretokenize_dataset(
        pcap_dir=args.pcap_dir,
        output_dir=args.output_dir,
        add_packet_separators=not args.no_packet_separators,
        save_progress_every=args.save_progress_every
    )
    
    total_time = time.time() - start_time
    
    # Print final statistics
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Total processing time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)")
    print(f"Files processed: {metadata['processing_stats']['successful_files']:,}")
    print(f"Files failed: {metadata['processing_stats']['failed_files']:,}")
    print(f"Total chunks: {metadata['total_chunks']:,}")
    print(f"Total tokens: {metadata['total_tokens']:,}")
    print(f"Processing speed: {metadata['processing_stats']['successful_files']/total_time:.1f} files/second")
    
    if metadata['total_tokens'] > 0:
        tokens_per_second = metadata['total_tokens'] / total_time
        print(f"Tokenization speed: {tokens_per_second:,.0f} tokens/second")
    
    return metadata


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Method already set
    
    main()