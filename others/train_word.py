# train_wordpiece.py

import argparse
from tokenizers import BertWordPieceTokenizer

def main():
    parser = argparse.ArgumentParser(description="Train a WordPiece tokenizer from a pre-tokenized corpus.")
    parser.add_argument("--corpus-file", type=str, required=True, help="Path to the training corpus (e.g., 'field_corpus.txt').")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the trained tokenizer vocab.")
    parser.add_argument("--vocab-size", type=int, default=30522, help="The desired size of the vocabulary.")
    args = parser.parse_args()

    # Initialize an empty tokenizer
    # We use BertWordPieceTokenizer as it's the standard WordPiece implementation.
    # We disable text cleaning/normalization because our "words" are byte representations.
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )

    print(f"Training WordPiece tokenizer on {args.corpus_file}...")

    # Train the tokenizer
    tokenizer.train(
        files=[args.corpus_file],
        vocab_size=args.vocab_size,
        min_frequency=2,
        # Standard special tokens for BERT-like models
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    # Save the tokenizer vocabulary file
    tokenizer.save_model(args.output_dir)
    print(f"✅ WordPiece tokenizer trained and saved in '{args.output_dir}' directory.")
    print(f"The vocabulary file is located at: {args.output_dir}/vocab.txt")

if __name__ == "__main__":
    main()