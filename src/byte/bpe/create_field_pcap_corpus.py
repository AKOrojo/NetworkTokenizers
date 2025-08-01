# create_corpus_from_tokenizer_parallel.py

import argparse
from pathlib import Path
import sys
import os
import time
import threading
from tqdm import tqdm
from multiprocessing import Pool

# 3rd-party libraries for monitoring
import wandb
import psutil

# Directly import your custom tokenizer class
from src.byte.raw.field_aware_tokenizer import PCAPFieldTokenizer

# A global variable to hold the tokenizer instance within each worker process.
tokenizer = None

def init_worker():
    """Initializer function for each worker process in the Pool."""
    global tokenizer
    # Each worker process creates its own instance of the tokenizer
    # to avoid issues with sending complex objects between processes.
    tokenizer = PCAPFieldTokenizer()

def process_single_pcap(pcap_path: str) -> str:
    """The main worker function that processes one PCAP file."""
    global tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer has not been initialized in this worker process.")
    
    try:
        all_tokens_in_file = tokenizer._tokenize(pcap_path=pcap_path)
        if not all_tokens_in_file:
            return ""
        return " ".join(all_tokens_in_file)
    except Exception as e:
        print(f"Error processing file {pcap_path}: {e}", file=sys.stderr)
        return ""

class SystemMonitor:
    """A class to monitor system and job stats in a background thread."""
    def __init__(self, output_filepath: str, interval: int = 30):
        self.output_filepath = Path(output_filepath)
        self.interval = interval
        self.processed_count = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)

    def _monitor_loop(self):
        """The main monitoring loop that runs in the background."""
        while not self._stop_event.is_set():
            # --- Get File Stats ---
            output_file_size_gb = 0
            if self.output_filepath.exists():
                output_file_size_gb = self.output_filepath.stat().st_size / (1024**3)

            # --- Get System Stats ---
            cpu_usage = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()

            # --- Log to W&B ---
            wandb.log({
                "system/cpu_usage_percent": cpu_usage,
                "system/ram_usage_percent": memory_info.percent,
                "job/output_file_size_gb": output_file_size_gb,
                "job/files_processed": self.processed_count,
            })
            
            time.sleep(self.interval)

    def start(self):
        print("Starting system monitor...")
        self._thread.start()

    def stop(self):
        print("Stopping system monitor...")
        self._stop_event.set()
        self._thread.join()

def main():
    parser = argparse.ArgumentParser(description="Create a corpus in parallel with W&B monitoring.")
    parser.add_argument("--pcap-dir", type=str, required=True, help="Directory containing .pcap or .pcapng files.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the output corpus text file.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes. Defaults to all available cores.")
    parser.add_argument("--wandb-project", type=str, default="corpus-creation", help="W&B project name.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="A name for this specific W&B run.")
    args = parser.parse_args()

    # --- Initialize W&B Run ---
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"process_{Path(args.pcap_dir).name}",
        job_type="preprocess",
        config=vars(args) # Log all command-line arguments
    )

    pcap_directory = Path(args.pcap_dir)
    pcap_files_paths = [str(p) for p in pcap_directory.glob("*.pcap")] + [str(p) for p in pcap_directory.glob("*.pcapng")]
    total_files = len(pcap_files_paths)

    if not pcap_files_paths:
        print(f"Error: No .pcap or .pcapng files found in '{args.pcap_dir}'", file=sys.stderr)
        wandb.log({"error": "No PCAP files found"})
        run.finish(exit_code=1)
        sys.exit(1)

    wandb.config.update({"total_files_found": total_files})
    print(f"Found {total_files} PCAP files. Processing in parallel on {args.num_workers or 'all available'} cores...")

    # --- Start Background Monitoring ---
    monitor = SystemMonitor(args.output_file)
    monitor.start()

    try:
        # Use a multiprocessing Pool to process files in parallel.
        with Pool(processes=args.num_workers, initializer=init_worker) as pool:
            with open(args.output_file, "w", encoding="utf-8") as f_out:
                # Use imap_unordered for efficiency
                for line in tqdm(pool.imap_unordered(process_single_pcap, pcap_files_paths), total=total_files):
                    monitor.processed_count += 1
                    if line:
                        f_out.write(line + "\n")

        # Log final metrics after the job is done
        final_file_size = Path(args.output_file).stat().st_size / (1024**3)
        wandb.summary["final_output_file_size_gb"] = final_file_size
        wandb.summary["status"] = "Completed"
        print(f"✅ Corpus creation complete. Output saved to '{args.output_file}'")

    except Exception as e:
        print(f"An error occurred during processing: {e}", file=sys.stderr)
        wandb.summary["status"] = "Failed"
        wandb.log({"error_message": str(e)})
        run.finish(exit_code=1) # Finish W&B run with a failure code
    
    finally:
        # --- Stop Monitoring and Finalize ---
        monitor.stop()
        if run and run.socket.is_open:
            if wandb.summary.get("status") != "Failed":
                 wandb.summary["status"] = "Completed"
            run.finish() # Ensure W&B run is always finished
        
if __name__ == "__main__":
    main()