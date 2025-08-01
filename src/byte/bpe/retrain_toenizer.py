import argparse
import os
import sentencepiece as spm
import wandb
import json

def save_vocab_as_json(model_prefix: str) -> None:
    """
    Reads the .vocab file from SentencePiece and saves it as a transformers-compatible vocab.json.
    """
    vocab_file = f"{model_prefix}.vocab"
    json_file = f"{model_prefix}-vocab.json" # Name the output file
    
    print(f"Converting '{vocab_file}' to JSON format...")
    
    vocab = {}
    with open(vocab_file, "r", encoding="utf-8") as f:
        # Enumerate each line to get the integer ID for each token
        for i, line in enumerate(f):
            # The token is the part of the line before the first tab
            token = line.split("\t")[0]
            vocab[token] = i
            
    with open(json_file, "w", encoding="utf-8") as f:
        # Dump the dictionary to a JSON file with nice formatting
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Vocabulary saved to '{json_file}'")

def main():
    parser = argparse.ArgumentParser(
        description="Re-train a SentencePiece tokenizer from a pre-processed text file with W&B reporting."
    )
    # --- Existing Arguments ---
    parser.add_argument(
        "--input-file",
        type=str,
        default="/data/orojoa/preprocessed_pcap_data.txt",
        help="Path to the large, pre-processed training text file."
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        required=True,
        help="Output prefix for the new trained model (e.g., 'unigram_16k')."
    )
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--model-type", type=str, default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--num-threads", type=int, default=os.cpu_count())
    
    # --- NEW ARGUMENT FOR SAMPLING ---
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=100_000_000, # Default to 100M sentences to prevent OOM errors
        help="Number of sentences to sample for training. Set to 0 to use the full dataset (requires huge memory)."
    )
    args = parser.parse_args()

    # --- 1. INITIALIZE WANDB ---
    # The project name can be customized.
    run = wandb.init(
        project="network-tokenizer-training",
        config=args # Log all hyperparameters from argparse
    )
    
    try:
        print(f"Starting new training run from: {args.input_file}")
        print(f"Model Type: {args.model_type}, Vocab Size: {args.vocab_size}")

        # --- 2. CONFIGURE TRAINING PARAMETERS ---
        # Build a dictionary of parameters to pass to the trainer
        trainer_params = {
            'input': args.input_file,
            'model_prefix': args.model_prefix,
            'vocab_size': args.vocab_size,
            'model_type': args.model_type,
            'character_coverage': 1.0,
            'byte_fallback': True,
            'num_threads': args.num_threads
        }
        
        # Add sampling parameters only if a positive size is provided
        if args.input_sentence_size > 0:
            print(f"Using a random sample of {args.input_sentence_size:,} sentences for training.")
            trainer_params['input_sentence_size'] = args.input_sentence_size
            trainer_params['shuffle_input_sentence'] = True
        else:
            print("WARNING: Using the full dataset for training. This may be very slow and memory-intensive.")

        # --- 3. RUN THE TRAINING ---
        spm.SentencePieceTrainer.train(**trainer_params)
        
        print("\n✅ Re-training complete!")
        save_vocab_as_json(args.model_prefix) # This is your existing helper function

        # --- 4. LOG RESULTS TO WANDB ---
        model_file = f"{args.model_prefix}.model"
        vocab_file = f"{args.model_prefix}-vocab.json"
        
        wandb.log({
            "status": "success",
            "output_model_file": model_file,
            "output_vocab_json": vocab_file
        })
        print(f"✅ Results successfully logged to W&B run: {run.name}")

    except Exception as e:
        print(f"❌ An error occurred during training: {e}")
        # Log the failure to W&B
        wandb.log({"status": "failed", "error_message": str(e)})
        raise # Re-raise the exception after logging
    finally:
        # --- 5. FINISH THE WANDB RUN ---
        # This ensures that the run is closed properly, even if an error occurs.
        wandb.finish()


if __name__ == '__main__':
    main()