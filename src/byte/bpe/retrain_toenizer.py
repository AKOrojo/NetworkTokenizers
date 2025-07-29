import argparse
import os
import sentencepiece as spm
from src.byte.bpe.train_pcap_sentencepiece_tokenizer import save_vocab_as_json

def main():
    parser = argparse.ArgumentParser(
        description="Re-train a SentencePiece tokenizer from a pre-processed text file."
    )
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
    args = parser.parse_args()

    print(f"Starting new training run from: {args.input_file}")
    print(f"Model Type: {args.model_type}, Vocab Size: {args.vocab_size}")

    # This just runs the fast training step
    spm.SentencePieceTrainer.train(
        input=args.input_file,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=1.0,
        byte_fallback=True,
        num_threads=args.num_threads
    )
    
    print("\n✅ Re-training complete!")
    save_vocab_as_json(args.model_prefix)

if __name__ == '__main__':
    main()