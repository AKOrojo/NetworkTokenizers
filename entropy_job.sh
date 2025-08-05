#!/bin/bash

CPUS=30

#=======================================================================
# PBS/TORQUE CONFIGURATION
#=======================================================================

# --- PBS Directives ---
#PBS -N pcap_pretokenize
#PBS -o pcap_pretokenize.out
#PBS -e pcap_pretokenize.err
#PBS -l nodes=1:ppn=46 
#PBS -l mem=512gb
#PBS -l walltime=216:00:00
#PBS -k oed
#PBS -m be -M abanisenioluwa_oroj1@baylor.edu

#=======================================================================
# JOB EXECUTION
#=======================================================================

# --- Force immediate output flushing ---
export PYTHONUNBUFFERED=1

# --- 1. Set up Environment ---
echo "========================================================"
echo "PCAP Pre-tokenization Job started at: $(date)"
echo "Job ID: ${PBS_JOBID}"
echo "Node: $(hostname)"
echo "========================================================"

# --- Load required system modules ---
echo "Loading system modules..."
module load cmake-gcc9/3.21.3
module load cuda12.6/toolkit/12.6.2

# Define project directories
PROJECT_ROOT="/home/orojoa/NetworkTokenizers"
INPUT_DIR="/data/orojoa/flows"
OUTPUT_DIR="/data/orojoa/chunks_new"
LOG_DIR="/data/orojoa/logs"

# Create output and log directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Activate the virtual environment
source "${PROJECT_ROOT}/.venv/bin/activate"
cd "${PROJECT_ROOT}"

# --- Set number of threads ---
THREADS=${PBS_NP:-42}

# Check if running under PBS and print info
if [ -z "$PBS_JOBID" ]; then
    echo "Warning: Not running in a PBS job. Log files will not be redirected."
    echo "Using ${THREADS} threads for this interactive run."
else
    echo "Running in PBS Job ID: $PBS_JOBID with ${THREADS} threads."
fi

# --- 2. Pre-flight Checks ---
echo "========================================================"
echo "Pre-flight checks..."
echo "========================================================"

# Check input directory
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory does not exist: ${INPUT_DIR}" >&2
    exit 1
fi

# Count PCAP files
PCAP_COUNT=$(find "${INPUT_DIR}" -name "*.pcap" -o -name "*.pcapng" | wc -l)
echo "Found ${PCAP_COUNT} PCAP files in input directory"

if [ ${PCAP_COUNT} -eq 0 ]; then
    echo "Error: No PCAP files found in ${INPUT_DIR}" >&2
    exit 1
fi

# Check available disk space
INPUT_SIZE=$(du -sb "${INPUT_DIR}" | cut -f1)
AVAILABLE_SPACE=$(df "${OUTPUT_DIR}" | tail -1 | awk '{print $4 * 1024}')
ESTIMATED_OUTPUT_SIZE=$((INPUT_SIZE * 8))  # Conservative estimate: 8x expansion

echo "Input directory size: $(du -sh "${INPUT_DIR}" | cut -f1)"
echo "Available space: $(df -h "${OUTPUT_DIR}" | tail -1 | awk '{print $4}')"
echo "Estimated output size: $(echo "${ESTIMATED_OUTPUT_SIZE}" | numfmt --to=iec)"

if [ ${ESTIMATED_OUTPUT_SIZE} -gt ${AVAILABLE_SPACE} ]; then
    echo "Warning: Estimated output size may exceed available space!" >&2
    echo "Consider reducing chunk size or clearing space." >&2
fi

# Check inode availability
AVAILABLE_INODES=$(df -i "${OUTPUT_DIR}" | tail -1 | awk '{print $4}')
echo "Available inodes: ${AVAILABLE_INODES}"

if [ ${AVAILABLE_INODES} -lt 750000 ]; then
    echo "Warning: Available inodes (${AVAILABLE_INODES}) may be insufficient!" >&2
fi

# --- 3. Run Pre-tokenization ---
echo "========================================================"
echo "Starting PCAP pre-tokenization at: $(date)"
echo "========================================================"

echo "Configuration:"
echo "  Input directory: ${INPUT_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Chunk size: 32768 tokens"
echo "  Max file size: 10GB"
echo "  Output format: numpy"
echo "  Processes: ${THREADS}"
echo "  PCAP files to process: ${PCAP_COUNT}"

# Run the pre-tokenization with progress monitoring
python -m src.preprocess.pretokenize_pcap_entropy \
    "${INPUT_DIR}" \
    "${OUTPUT_DIR}" \
    --chunk-size 2097152 \
    --max-file-size-gb 10 \
    --output-format numpy \
    --num-processes "${THREADS}" \
    --save-progress-every 1000 \
    2>&1 | tee "${LOG_DIR}/pretokenize_$(date +%Y%m%d_%H%M%S).log"

# Capture exit status
EXIT_STATUS=$?

# --- 4. Post-processing Analysis ---
echo "========================================================"
echo "Post-processing analysis..."
echo "========================================================"

if [ ${EXIT_STATUS} -eq 0 ]; then
    echo "Pre-tokenization completed successfully at: $(date)"
    
    # Analyze output
    if [ -f "${OUTPUT_DIR}/metadata.json" ]; then
        echo "Metadata file created successfully"
        
        # Extract key statistics from metadata
        python3 -c "
import json
try:
    with open('${OUTPUT_DIR}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f\"Final Statistics:\")
    print(f\"  Total chunks: {metadata.get('total_chunks', 'N/A'):,}\")
    print(f\"  Total tokens: {metadata.get('total_tokens', 'N/A'):,}\")
    print(f\"  Successful files: {metadata.get('processing_stats', {}).get('successful_files', 'N/A'):,}\")
    print(f\"  Failed files: {metadata.get('processing_stats', {}).get('failed_files', 'N/A'):,}\")
    
    if 'processing_stats' in metadata and 'total_time' in metadata['processing_stats']:
        total_time = metadata['processing_stats']['total_time']
        print(f\"  Processing time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)\")
    
    if metadata.get('total_tokens', 0) > 0 and 'processing_stats' in metadata:
        if 'total_time' in metadata['processing_stats']:
            tokens_per_sec = metadata['total_tokens'] / metadata['processing_stats']['total_time']
            print(f\"  Tokenization speed: {tokens_per_sec:,.0f} tokens/second\")
except Exception as e:
    print(f\"Error reading metadata: {e}\")
"
    else
        echo "Warning: metadata.json not found"
    fi
    
    # Check output directory size and file count
    OUTPUT_SIZE=$(du -sh "${OUTPUT_DIR}" | cut -f1)
    OUTPUT_FILES=$(find "${OUTPUT_DIR}" -name "chunk_*.npy" | wc -l)
    
    echo "Output directory size: ${OUTPUT_SIZE}"
    echo "Number of chunk files created: ${OUTPUT_FILES}"
    
    # Check inode usage
    USED_INODES=$(df -i "${OUTPUT_DIR}" | tail -1 | awk '{print $3}')
    echo "Inodes used: ${USED_INODES}"
    
    # Verify random chunks
    echo "Verifying random chunks..."
    python3 -c "
import numpy as np
import os
from pathlib import Path
import random

output_dir = Path('${OUTPUT_DIR}')
chunk_files = list(output_dir.glob('chunk_*.npy'))

if chunk_files:
    # Test 5 random chunks
    sample_files = random.sample(chunk_files, min(5, len(chunk_files)))
    for chunk_file in sample_files:
        try:
            chunk = np.load(chunk_file)
            print(f\"  {chunk_file.name}: {len(chunk)} tokens, dtype={chunk.dtype}\")
        except Exception as e:
            print(f\"  Error loading {chunk_file.name}: {e}\")
else:
    print('  No chunk files found!')
"
    
else
    echo "========================================================"
    echo "Error: Pre-tokenization failed with exit code: ${EXIT_STATUS}" >&2
    echo "========================================================"
    
    # Check for common issues
    if [ -f "${OUTPUT_DIR}/progress.json" ]; then
        echo "Progress file exists - job may have been partially completed"
        echo "Check progress.json for details"
    fi
    
    # Check disk space issues
    CURRENT_SPACE=$(df "${OUTPUT_DIR}" | tail -1 | awk '{print $4 * 1024}')
    if [ ${CURRENT_SPACE} -lt 1073741824 ]; then  # Less than 1GB
        echo "Warning: Low disk space may have caused failure" >&2
    fi
    
    # Check for core dumps or memory issues
    if [ -f "core" ]; then
        echo "Core dump detected - possible memory issue" >&2
    fi
    
    exit ${EXIT_STATUS}
fi

echo "========================================================"
echo "Job completed successfully at: $(date)"
echo "========================================================"

# --- 5. Cleanup (optional) ---
echo "Cleaning up temporary files..."
# Remove any temporary files if needed
# find /tmp -name "*pcap*" -user $(whoami) -delete 2>/dev/null || true

echo "Pre-tokenization job finished."
echo "========================================================"