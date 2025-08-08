#!/usr/bin/env python3
"""
PCAP Duplicate Detector using dpkt with Multiprocessing
Compares packet content to find duplicate pcap files regardless of filename
"""

import os
import sys
import hashlib
import dpkt
import socket
from collections import defaultdict, Counter
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count, Manager
import time
from functools import partial

def extract_packet_content_worker(pcap_file):
    """Worker function for multiprocessing - extract packet data from pcap file"""
    try:
        with open(pcap_file, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            packet_data = []
            
            for timestamp, buf in pcap:
                # Extract the raw packet data (excluding timestamp)
                packet_data.append(buf)
            
            return str(pcap_file), packet_data
            
    except Exception as e:
        return str(pcap_file), None

def calculate_content_hash_worker(file_and_packets):
    """Worker function to calculate hash from file and packet data"""
    file_path, packet_data = file_and_packets
    
    if not packet_data:
        return file_path, None
        
    # Create a hash based on all packet contents
    hasher = hashlib.sha256()
    for packet in packet_data:
        hasher.update(packet)
    
    return file_path, hasher.hexdigest()

def calculate_summary_hash_worker(file_and_packets):
    """Worker function to calculate summary hash"""
    file_path, packet_data = file_and_packets
    
    if not packet_data:
        return file_path, None
        
    packet_sizes = [len(packet) for packet in packet_data]
    packet_count = len(packet_data)
    
    # Create summary including count and size distribution
    summary = f"{packet_count}:{':'.join(map(str, sorted(packet_sizes)))}"
    summary_hash = hashlib.md5(summary.encode()).hexdigest()
    
    return file_path, summary_hash, packet_data

class PcapAnalyzer:
    def __init__(self, progress=True, num_processes=None):
        self.progress = progress
        self.num_processes = num_processes or cpu_count()
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pcap_analysis.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def extract_packet_content(self, pcap_file):
        """Extract packet data from pcap file for comparison"""
        try:
            with open(pcap_file, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                packet_data = []
                
                for timestamp, buf in pcap:
                    # Extract the raw packet data (excluding timestamp)
                    packet_data.append(buf)
                
                return packet_data
                
        except Exception as e:
            self.logger.error(f"Error reading {pcap_file}: {e}")
            return None

    def calculate_content_hash(self, packet_data):
        """Calculate hash of packet content for comparison"""
        if not packet_data:
            return None
            
        # Create a hash based on all packet contents
        hasher = hashlib.sha256()
        for packet in packet_data:
            hasher.update(packet)
        
        return hasher.hexdigest()

    def process_files_multiprocessing(self, pcap_files, chunk_size=None):
        """Process files using multiprocessing"""
        if chunk_size is None:
            chunk_size = max(1, len(pcap_files) // (self.num_processes * 4))
        
        self.logger.info(f"Processing {len(pcap_files)} files using {self.num_processes} processes")
        
        with Pool(self.num_processes) as pool:
            if self.progress:
                # Use imap for progress tracking
                results = []
                with tqdm(total=len(pcap_files), desc="Extracting packet data") as pbar:
                    for result in pool.imap(extract_packet_content_worker, pcap_files, chunksize=chunk_size):
                        results.append(result)
                        pbar.update(1)
            else:
                results = pool.map(extract_packet_content_worker, pcap_files, chunksize=chunk_size)
        
        return results

    def find_duplicates_multiprocessing(self, directory, output_file=None, deletion_file=None):
        """Find duplicates using multiprocessing for faster processing"""
        pcap_files = list(Path(directory).rglob("*.pcap")) + list(Path(directory).rglob("*.pcapng"))
        
        if not pcap_files:
            self.logger.warning(f"No pcap files found in {directory}")
            return {}
        
        self.logger.info(f"Found {len(pcap_files)} pcap files to analyze")
        
        # Step 1: Extract packet data using multiprocessing
        file_packet_results = self.process_files_multiprocessing(pcap_files)
        
        # Filter out failed reads
        valid_results = [(f, p) for f, p in file_packet_results if p is not None]
        self.logger.info(f"Successfully read {len(valid_results)} files")
        
        # Step 2: Calculate content hashes using multiprocessing
        chunk_size = max(1, len(valid_results) // (self.num_processes * 4))
        
        with Pool(self.num_processes) as pool:
            if self.progress:
                hash_results = []
                with tqdm(total=len(valid_results), desc="Calculating content hashes") as pbar:
                    for result in pool.imap(calculate_content_hash_worker, valid_results, chunksize=chunk_size):
                        hash_results.append(result)
                        pbar.update(1)
            else:
                hash_results = pool.map(calculate_content_hash_worker, valid_results, chunksize=chunk_size)
        
        # Step 3: Group by hash to find duplicates
        hash_to_files = defaultdict(list)
        for file_path, content_hash in hash_results:
            if content_hash:
                hash_to_files[content_hash].append(file_path)
        
        # Find duplicates
        duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
        
        self.logger.info(f"Found {len(duplicates)} groups of duplicate files")
        
        if output_file:
            self._save_results(duplicates, output_file)
        
        if deletion_file:
            self._create_deletion_file(duplicates, deletion_file)
        
        return duplicates

    def find_duplicates_smart_multiprocessing(self, directory, output_file=None, deletion_file=None):
        """Find duplicates using smart two-pass comparison with multiprocessing"""
        pcap_files = list(Path(directory).rglob("*.pcap")) + list(Path(directory).rglob("*.pcapng"))
        
        if not pcap_files:
            self.logger.warning(f"No pcap files found in {directory}")
            return {}
        
        self.logger.info(f"Found {len(pcap_files)} pcap files to analyze")
        
        # Step 1: Extract packet data and calculate summary hashes
        file_packet_results = self.process_files_multiprocessing(pcap_files)
        valid_results = [(f, p) for f, p in file_packet_results if p is not None]
        
        # Step 2: Calculate summary hashes using multiprocessing
        chunk_size = max(1, len(valid_results) // (self.num_processes * 4))
        
        with Pool(self.num_processes) as pool:
            if self.progress:
                summary_results = []
                with tqdm(total=len(valid_results), desc="Calculating summary hashes") as pbar:
                    for result in pool.imap(calculate_summary_hash_worker, valid_results, chunksize=chunk_size):
                        summary_results.append(result)
                        pbar.update(1)
            else:
                summary_results = pool.map(calculate_summary_hash_worker, valid_results, chunksize=chunk_size)
        
        # Group by summary hash
        summary_groups = defaultdict(list)
        for file_path, summary_hash, packet_data in summary_results:
            if summary_hash:
                summary_groups[summary_hash].append((file_path, packet_data))
        
        # Step 3: Detailed comparison within groups that have potential duplicates
        duplicates = {}
        candidates_for_detailed = []
        
        for summary_hash, candidates in summary_groups.items():
            if len(candidates) > 1:
                candidates_for_detailed.extend(candidates)
        
        if candidates_for_detailed:
            self.logger.info(f"Performing detailed comparison on {len(candidates_for_detailed)} candidate files")
            
            with Pool(self.num_processes) as pool:
                if self.progress:
                    detailed_results = []
                    with tqdm(total=len(candidates_for_detailed), desc="Detailed content analysis") as pbar:
                        for result in pool.imap(calculate_content_hash_worker, candidates_for_detailed, chunksize=chunk_size):
                            detailed_results.append(result)
                            pbar.update(1)
                else:
                    detailed_results = pool.map(calculate_content_hash_worker, candidates_for_detailed, chunksize=chunk_size)
            
            # Group by content hash
            content_groups = defaultdict(list)
            for file_path, content_hash in detailed_results:
                if content_hash:
                    content_groups[content_hash].append(file_path)
            
            # Find actual duplicates
            duplicates = {h: files for h, files in content_groups.items() if len(files) > 1}
        
        self.logger.info(f"Found {len(duplicates)} groups of duplicate files")
        
        if output_file:
            self._save_results(duplicates, output_file)
        
        if deletion_file:
            self._create_deletion_file(duplicates, deletion_file)
        
        return duplicates

    def batch_process_multiprocessing(self, directory, batch_size=5000, output_file=None, deletion_file=None):
        """Process large directories in batches using multiprocessing"""
        pcap_files = list(Path(directory).rglob("*.pcap")) + list(Path(directory).rglob("*.pcapng"))
        
        self.logger.info(f"Found {len(pcap_files)} pcap files. Processing in batches of {batch_size}")
        
        all_hashes = {}
        duplicates = defaultdict(list)
        
        for i in range(0, len(pcap_files), batch_size):
            batch = pcap_files[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(pcap_files) + batch_size - 1)//batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Process batch with multiprocessing
            batch_results = self.process_files_multiprocessing(batch)
            valid_batch = [(f, p) for f, p in batch_results if p is not None]
            
            # Calculate hashes for this batch
            chunk_size = max(1, len(valid_batch) // (self.num_processes * 2))
            
            with Pool(self.num_processes) as pool:
                if self.progress:
                    hash_results = []
                    with tqdm(total=len(valid_batch), desc=f"Batch {batch_num} hashing") as pbar:
                        for result in pool.imap(calculate_content_hash_worker, valid_batch, chunksize=chunk_size):
                            hash_results.append(result)
                            pbar.update(1)
                else:
                    hash_results = pool.map(calculate_content_hash_worker, valid_batch, chunksize=chunk_size)
            
            # Check for duplicates
            for file_path, content_hash in hash_results:
                if content_hash:
                    if content_hash in all_hashes:
                        # Found a duplicate
                        if content_hash not in duplicates:
                            duplicates[content_hash] = [all_hashes[content_hash]]
                        duplicates[content_hash].append(file_path)
                    else:
                        all_hashes[content_hash] = file_path
        
        duplicates = dict(duplicates)
        self.logger.info(f"Found {len(duplicates)} groups of duplicate files")
        
        if output_file:
            self._save_results(duplicates, output_file)
        
        if deletion_file:
            self._create_deletion_file(duplicates, deletion_file)
        
        return duplicates

    def _save_results(self, duplicates, output_file):
        """Save results to JSON file"""
        result = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_duplicate_groups': len(duplicates),
            'total_duplicate_files': sum(len(files) for files in duplicates.values()),
            'duplicates': duplicates
        }
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")

    def _create_deletion_file(self, duplicates, deletion_file):
        """Create file with list of files to delete (keeping first file in each group)"""
        files_to_delete = []
        
        for hash_val, files in duplicates.items():
            # Keep the first file, mark others for deletion
            if len(files) > 1:
                files_to_delete.extend(files[1:])  # Skip first file, delete the rest
        
        # Write deletion list
        with open(deletion_file, 'w') as f:
            for file_path in sorted(files_to_delete):
                f.write(f"{file_path}\n")
        
        # Also create a shell script for easy deletion
        script_file = deletion_file.replace('.txt', '_delete.sh')
        with open(script_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Script to delete duplicate PCAP files\n")
            f.write(f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total files to delete: {len(files_to_delete)}\n\n")
            f.write(f"echo 'Deleting {len(files_to_delete)} duplicate files...'\n")
            f.write(f"while IFS= read -r file; do\n")
            f.write(f"    if [ -f \"$file\" ]; then\n")
            f.write(f"        echo \"Deleting: $file\"\n")
            f.write(f"        rm \"$file\"\n")
            f.write(f"    else\n")
            f.write(f"        echo \"File not found: $file\"\n")
            f.write(f"    fi\n")
            f.write(f"done < \"{deletion_file}\"\n")
            f.write(f"echo 'Deletion complete!'\n")
        
        # Make script executable
        os.chmod(script_file, 0o755)
        
        total_saved = len(files_to_delete)
        total_groups = len(duplicates)
        
        self.logger.info(f"Deletion file created: {deletion_file}")
        self.logger.info(f"Shell script created: {script_file}")
        self.logger.info(f"Files marked for deletion: {total_saved}")
        self.logger.info(f"Duplicate groups: {total_groups}")
        
        print(f"\nDeletion Summary:")
        print(f"- Files to delete: {total_saved}")
        print(f"- Files to keep: {total_groups}")
        print(f"- Deletion list: {deletion_file}")
        print(f"- Shell script: {script_file}")
        
        print(f"\nTo delete the files, you can use:")
        print(f"1. Execute the shell script: ./{script_file}")
        print(f"2. Use xargs: cat {deletion_file} | xargs rm")
        print(f"3. Use while loop: while read file; do rm \"$file\"; done < {deletion_file}")

    def print_summary(self, duplicates):
        """Print summary of duplicates found"""
        if not duplicates:
            print("No duplicate files found.")
            return
        
        total_files = sum(len(files) for files in duplicates.values())
        total_groups = len(duplicates)
        
        print(f"\nDuplicate Analysis Summary:")
        print(f"- Total duplicate groups: {total_groups}")
        print(f"- Total duplicate files: {total_files}")
        print(f"- Files that could be removed: {total_files - total_groups}")
        
        print(f"\nTop 10 largest duplicate groups:")
        sorted_groups = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
        
        for i, (hash_val, files) in enumerate(sorted_groups[:10]):
            print(f"{i+1}. Group with {len(files)} files:")
            for file_path in files[:3]:  # Show first 3 files
                print(f"   - {file_path}")
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more files")


def main():
    parser = argparse.ArgumentParser(description="Find duplicate PCAP files based on packet content using multiprocessing")
    parser.add_argument("directory", help="Directory containing PCAP files")
    parser.add_argument("-o", "--output", help="Output file for results (JSON format)")
    parser.add_argument("-d", "--delete-list", help="Output txt file with files to delete")
    parser.add_argument("-m", "--method", choices=["fast", "smart", "batch"], default="smart",
                      help="Comparison method: fast (content hash), smart (two-pass), batch (for huge datasets)")
    parser.add_argument("-b", "--batch-size", type=int, default=5000,
                      help="Batch size for batch processing mode")
    parser.add_argument("-j", "--jobs", type=int, default=cpu_count(),
                      help=f"Number of parallel processes (default: {cpu_count()})")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} does not exist")
        sys.exit(1)
    
    analyzer = PcapAnalyzer(progress=not args.no_progress, num_processes=args.jobs)
    
    print(f"Analyzing PCAP files in: {args.directory}")
    print(f"Method: {args.method}")
    print(f"Parallel processes: {args.jobs}")
    
    start_time = time.time()
    
    if args.method == "fast":
        duplicates = analyzer.find_duplicates_multiprocessing(args.directory, args.output, args.delete_list)
    elif args.method == "smart":
        duplicates = analyzer.find_duplicates_smart_multiprocessing(args.directory, args.output, args.delete_list)
    elif args.method == "batch":
        duplicates = analyzer.batch_process_multiprocessing(args.directory, args.batch_size, args.output, args.delete_list)
    
    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    
    analyzer.print_summary(duplicates)


if __name__ == "__main__":
    main()
