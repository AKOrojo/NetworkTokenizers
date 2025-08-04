#!/bin/bash

#=======================================================================
# PBS/TORQUE CONFIGURATION
#=======================================================================

# --- PBS Directives ---
#PBS -N sentencepiece_training
#PBS -o sentencepiece_training.out
#PBS -e sentencepiece_training.err
#PBS -l nodes=1:ppn=48
#PBS -l mem=800gb
#PBS -l walltime=72:00:00
#PBS -k oed
#PBS -m be -M abanisenioluwa_oroj1@baylor.edu

#=======================================================================
# CONFIGURATION
#=======================================================================

# Vocabulary sizes to train (adjust based on your needs) - VOCAB_SIZES=(1024 2048 4096 8192 16384 32768 65536)
VOCAB_SIZES=(8192)

# Input corpus path
CORPUS_PATH="/data/orojoa/network_corpus.txt"

# Number of CPU cores to use for training
CPUS=46  # Leave 2 cores for system

# Model types to try
MODEL_TYPES=("bpe" "unigram")

# Maximum sentence length (packets can be long)
MAX_SENTENCE_LENGTH=10000

# Number of input sentences for training (use subset for faster training)
INPUT_SENTENCE_SIZE=10000000  # 10M sentences instead of full 184M

#=======================================================================
# JOB EXECUTION
#=======================================================================

# --- Force immediate output flushing ---
export PYTHONUNBUFFERED=1

# --- 1. Set up Environment ---
echo "========================================================"
echo "SentencePiece Training Job started at: $(date)"
echo "Job ID: ${PBS_JOBID}"
echo "Node: $(hostname)"
echo "Available Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "CPU Cores: $(nproc)"
echo "========================================================"

# --- Load required system modules ---
echo "Loading system modules..."
module load cmake-gcc9/3.21.3
module load cuda12.6/toolkit/12.6.2

# Define project directories
PROJECT_ROOT="/home/orojoa/NetworkTokenizers"

# Activate the virtual environment
source "${PROJECT_ROOT}/.venv/bin/activate"
cd "${PROJECT_ROOT}"

# Check if running under PBS
THREADS=${PBS_NP:-8}
if [ -z "$PBS_JOBID" ]; then
    echo "Warning: Not running in a PBS job."
    THREADS=8
else
    echo "Running in PBS Job ID: $PBS_JOBID with ${THREADS} threads available."
fi

# --- 2. Verify Input Corpus ---
echo "Verifying input corpus..."
if [ ! -f "$CORPUS_PATH" ]; then
    echo "ERROR: Corpus file not found: $CORPUS_PATH"
    exit 1
fi

CORPUS_SIZE=$(du -h "$CORPUS_PATH" | cut -f1)
CORPUS_LINES=$(wc -l < "$CORPUS_PATH")
echo "Corpus size: $CORPUS_SIZE"
echo "Corpus lines: $CORPUS_LINES"

# --- 3. Create Results Directory ---
RESULTS_DIR="/data/orojoa/sentencepiece_models"
mkdir -p "$RESULTS_DIR"
echo "Results will be saved to: $RESULTS_DIR"

# --- 4. Training Function ---
train_model() {
    local vocab_size=$1
    local model_type=$2
    
    echo ""
    echo "========================================================"
    echo "Training ${model_type} model with vocab size: ${vocab_size}"
    echo "Started at: $(date)"
    echo "========================================================"
    
    # Create model prefix
    local model_prefix="network_${model_type}_${vocab_size}k"
    
    # Run training with evaluation
    python -m src.byte.bpe.train_sentencepiece \
        --input "$CORPUS_PATH" \
        --model-prefix "$model_prefix" \
        --vocab-size "$vocab_size" \
        --model-type "$model_type" \
        --max-sentence-length "$MAX_SENTENCE_LENGTH" \
        --input-sentence-size "$INPUT_SENTENCE_SIZE" \
        --num-threads "$CPUS" \
        --character-coverage 1.0 \
        --evaluate \
        --eval-samples 5000 \
        --verbose
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Successfully trained ${model_type} model with vocab size ${vocab_size}"
        
        # Move results to central location
        if [ -d "vocab" ]; then
            mv vocab/* "$RESULTS_DIR/" 2>/dev/null || true
            rmdir vocab 2>/dev/null || true
        fi
        
        # Print model info
        local model_files=$(find "$RESULTS_DIR" -name "*${model_prefix}*" -type f)
        if [ -n "$model_files" ]; then
            echo "Model files created:"
            echo "$model_files" | while read file; do
                echo "  $(basename "$file"): $(du -h "$file" | cut -f1)"
            done
        fi
    else
        echo "❌ Failed to train ${model_type} model with vocab size ${vocab_size} (exit code: $exit_code)"
        return $exit_code
    fi
    
    echo "Completed at: $(date)"
    echo "========================================================"
}

# --- 5. Main Training Loop ---
echo ""
echo "========================================================"
echo "Starting training with the following configuration:"
echo "Vocabulary sizes: ${VOCAB_SIZES[*]}"
echo "Model types: ${MODEL_TYPES[*]}"
echo "Input sentence size: $INPUT_SENTENCE_SIZE"
echo "Max sentence length: $MAX_SENTENCE_LENGTH"
echo "CPU threads: $CPUS"
echo "========================================================"

TOTAL_MODELS=$((${#VOCAB_SIZES[@]} * ${#MODEL_TYPES[@]}))
CURRENT_MODEL=0
FAILED_MODELS=0

# Train models for each combination
for model_type in "${MODEL_TYPES[@]}"; do
    for vocab_size in "${VOCAB_SIZES[@]}"; do
        CURRENT_MODEL=$((CURRENT_MODEL + 1))
        echo ""
        echo "🚀 Training model $CURRENT_MODEL of $TOTAL_MODELS"
        
        if ! train_model "$vocab_size" "$model_type"; then
            FAILED_MODELS=$((FAILED_MODELS + 1))
            echo "⚠️  Continuing with next model despite failure..."
        fi
        
        # Show progress
        REMAINING=$((TOTAL_MODELS - CURRENT_MODEL))
        echo "Progress: $CURRENT_MODEL/$TOTAL_MODELS completed, $REMAINING remaining"
        
        # Memory check
        echo "Memory usage: $(free -h | grep '^Mem:' | awk '{print "Used: " $3 ", Available: " $7}')"
    done
done

# --- 6. Final Summary ---
echo ""
echo "========================================================"
echo "TRAINING SUMMARY"
echo "========================================================"
echo "Total models attempted: $TOTAL_MODELS"
echo "Successfully trained: $((TOTAL_MODELS - FAILED_MODELS))"
echo "Failed: $FAILED_MODELS"
echo ""

# List all created models
if [ -d "$RESULTS_DIR" ]; then
    echo "Created model files:"
    find "$RESULTS_DIR" -name "*.model" -type f | sort | while read model; do
        model_size=$(du -h "$model" | cut -f1)
        model_name=$(basename "$model")
        echo "  $model_name ($model_size)"
    done
    
    echo ""
    echo "Total storage used: $(du -sh "$RESULTS_DIR" | cut -f1)"
    
    # Show evaluation summaries if available
    echo ""
    echo "Evaluation summaries:"
    find "$RESULTS_DIR" -name "*_info.json" -type f | sort | while read info_file; do
        if command -v jq >/dev/null 2>&1; then
            vocab_size=$(jq -r '.training_stats.vocab_size' "$info_file" 2>/dev/null || echo "unknown")
            model_type=$(jq -r '.training_stats.model_type' "$info_file" 2>/dev/null || echo "unknown")
            compression=$(jq -r '.evaluation_stats.compression_ratio' "$info_file" 2>/dev/null || echo "unknown")
            echo "  $(basename "$info_file"): ${model_type} vocab=${vocab_size} compression=${compression}x"
        else
            echo "  $(basename "$info_file")"
        fi
    done
fi

echo ""
echo "Job completed at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "========================================================"

# Exit with error code if any models failed
if [ $FAILED_MODELS -gt 0 ]; then
    echo "Warning: $FAILED_MODELS models failed to train"
    exit 1
else
    echo "All models trained successfully!"
    exit 0
fi