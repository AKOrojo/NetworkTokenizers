#!/usr/bin/env python3
"""
SentencePiece Model Training Script for PCAP Data

This script trains a SentencePiece model on the corpus prepared from PCAP files,
where each packet is treated as a sentence with space-separated byte values.

Usage:
    python train_sentencepiece.py --input corpus.txt --model-prefix pcap_model --vocab-size 8000
"""

import argparse
import json
import logging
import os
import time
from typing import List, Optional
import sentencepiece as spm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentencePieceTrainer:
    """
    Trains SentencePiece models on PCAP corpus data.
    """

    def __init__(self):
        self.model = None
        self.training_stats = {}

    def validate_corpus(self, corpus_path: str) -> dict:
        """
        Validate and analyze the input corpus.

        Args:
            corpus_path: Path to the corpus file

        Returns:
            Dictionary with corpus statistics
        """
        logger.info(f"Validating corpus: {corpus_path}")

        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        stats = {
            'total_lines': 0,
            'total_tokens': 0,
            'avg_tokens_per_line': 0,
            'max_tokens_per_line': 0,
            'min_tokens_per_line': float('inf'),
            'unique_tokens': set(),
            'file_size_mb': 0
        }

        file_size = os.path.getsize(corpus_path)
        stats['file_size_mb'] = file_size / 1024 / 1024

        logger.info(f"Corpus file size: {stats['file_size_mb']:.1f} MB")

        # Sample and analyze the corpus
        sample_lines = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                tokens = line.split()
                num_tokens = len(tokens)

                stats['total_lines'] += 1
                stats['total_tokens'] += num_tokens
                stats['max_tokens_per_line'] = max(stats['max_tokens_per_line'], num_tokens)
                stats['min_tokens_per_line'] = min(stats['min_tokens_per_line'], num_tokens)

                # Sample first 1000 lines for unique token analysis
                if line_num <= 1000:
                    stats['unique_tokens'].update(tokens)
                    sample_lines.append(line)

                # Progress logging
                if line_num % 100000 == 0:
                    logger.info(f"Analyzed {line_num:,} lines...")

        if stats['total_lines'] > 0:
            stats['avg_tokens_per_line'] = stats['total_tokens'] / stats['total_lines']

        # Convert set to count for JSON serialization
        stats['unique_tokens_sample'] = len(stats['unique_tokens'])
        del stats['unique_tokens']

        logger.info(f"Corpus analysis complete:")
        logger.info(f"  Total sentences (packets): {stats['total_lines']:,}")
        logger.info(f"  Total tokens (bytes): {stats['total_tokens']:,}")
        logger.info(f"  Average tokens per sentence: {stats['avg_tokens_per_line']:.1f}")
        logger.info(f"  Token range per sentence: {stats['min_tokens_per_line']} - {stats['max_tokens_per_line']}")
        logger.info(f"  Unique tokens in sample: {stats['unique_tokens_sample']}")

        return stats

    def train_model(self,
                    corpus_path: str,
                    model_prefix: str,
                    vocab_size: int = 8000,
                    model_type: str = 'bpe',
                    character_coverage: float = 1.0,
                    input_sentence_size: int = 1000000,
                    shuffle_input_sentence: bool = True,
                    normalization_rule_name: str = 'identity',
                    add_dummy_prefix: bool = False,
                    remove_extra_whitespaces: bool = False,
                    max_sentence_length: int = 4096,
                    num_threads: int = None,
                    split_digits: bool = False,
                    user_defined_symbols: Optional[List[str]] = None) -> str:
        """
        Train a SentencePiece model.

        Args:
            corpus_path: Path to training corpus
            model_prefix: Output model prefix
            vocab_size: Target vocabulary size
            model_type: Model type ('bpe', 'unigram', 'char', 'word')
            character_coverage: Character coverage ratio
            input_sentence_size: Number of sentences to use for training
            shuffle_input_sentence: Whether to shuffle input sentences
            normalization_rule_name: Text normalization rule
            add_dummy_prefix: Whether to add dummy prefix
            remove_extra_whitespaces: Whether to remove extra whitespaces
            max_sentence_length: Maximum sentence length
            num_threads: Number of threads for training
            split_digits: Whether to split digits
            user_defined_symbols: Additional symbols to include in vocabulary

        Returns:
            Path to the trained model file
        """
        logger.info("Starting SentencePiece model training...")

        # Prepare training arguments
        training_args = [
            f'--input={corpus_path}',
            f'--model_prefix={model_prefix}',
            f'--vocab_size={vocab_size}',
            f'--model_type={model_type}',
            f'--character_coverage={character_coverage}',
            f'--input_sentence_size={input_sentence_size}',
            f'--shuffle_input_sentence={shuffle_input_sentence}',
            f'--normalization_rule_name={normalization_rule_name}',
            f'--add_dummy_prefix={add_dummy_prefix}',
            f'--remove_extra_whitespaces={remove_extra_whitespaces}',
            f'--max_sentence_length={max_sentence_length}',
            f'--split_digits={split_digits}',
        ]

        # Add threading if specified
        if num_threads:
            training_args.append(f'--num_threads={num_threads}')

        # Add user-defined symbols
        if user_defined_symbols:
            symbols_str = ','.join(user_defined_symbols)
            training_args.append(f'--user_defined_symbols={symbols_str}')

        # For network data, we might want special tokens for common byte patterns
        if model_type in ['bpe', 'unigram']:
            # Add common network byte patterns as user symbols
            network_symbols = [
                '0', '1', '255',  # Common byte values
                '0 0', '255 255',  # Common patterns
                '8 0', '8 6',  # Common Ethernet types
            ]
            if user_defined_symbols:
                network_symbols.extend(user_defined_symbols)
            symbols_str = ','.join(network_symbols)
            training_args.append(f'--user_defined_symbols={symbols_str}')

        # Log training configuration
        logger.info("Training configuration:")
        for arg in training_args:
            logger.info(f"  {arg}")

        # Start training
        start_time = time.time()

        try:
            spm.SentencePieceTrainer.train(' '.join(training_args))
            training_time = time.time() - start_time

            logger.info(f"Training completed in {training_time:.1f} seconds")

            # Store training stats
            self.training_stats = {
                'training_time': training_time,
                'vocab_size': vocab_size,
                'model_type': model_type,
                'corpus_path': corpus_path,
                'model_prefix': model_prefix
            }

            model_path = f"{model_prefix}.model"
            if os.path.exists(model_path):
                logger.info(f"Model saved to: {model_path}")
                return model_path
            else:
                raise FileNotFoundError(f"Expected model file not found: {model_path}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def evaluate_model(self, model_path: str, corpus_path: str,
                       num_samples: int = 1000) -> dict:
        """
        Evaluate the trained SentencePiece model.

        Args:
            model_path: Path to the trained model
            corpus_path: Path to the corpus for evaluation
            num_samples: Number of samples to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model: {model_path}")

        # Load the model
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)

        eval_stats = {
            'vocab_size': sp.vocab_size(),
            'avg_pieces_per_sentence': 0,
            'compression_ratio': 0,
            'sample_encodings': [],
            'vocabulary_usage': {},
        }

        logger.info(f"Model vocabulary size: {eval_stats['vocab_size']}")

        # Sample evaluation
        total_original_tokens = 0
        total_pieces = 0
        piece_counts = {}

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                # Original tokens
                original_tokens = line.split()
                original_count = len(original_tokens)

                # Encode with SentencePiece
                pieces = sp.encode_as_pieces(line)
                piece_count = len(pieces)

                total_original_tokens += original_count
                total_pieces += piece_count

                # Count piece usage
                for piece in pieces:
                    piece_counts[piece] = piece_counts.get(piece, 0) + 1

                # Store some samples
                if i < 10:
                    eval_stats['sample_encodings'].append({
                        'original': line[:100] + ('...' if len(line) > 100 else ''),
                        'original_tokens': original_count,
                        'pieces': pieces[:20] + (['...'] if len(pieces) > 20 else []),
                        'piece_count': piece_count,
                        'compression': f"{original_count}/{piece_count} = {original_count / piece_count:.2f}"
                    })

        if total_original_tokens > 0:
            eval_stats['avg_pieces_per_sentence'] = total_pieces / min(num_samples, total_original_tokens)
            eval_stats['compression_ratio'] = total_original_tokens / total_pieces

        # Vocabulary usage analysis
        eval_stats['vocabulary_usage'] = {
            'unique_pieces_used': len(piece_counts),
            'vocab_utilization': len(piece_counts) / eval_stats['vocab_size'],
            'most_common_pieces': sorted(piece_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        }

        logger.info(f"Evaluation results:")
        logger.info(f"  Average pieces per sentence: {eval_stats['avg_pieces_per_sentence']:.1f}")
        logger.info(f"  Compression ratio: {eval_stats['compression_ratio']:.2f}x")
        logger.info(f"  Vocabulary utilization: {eval_stats['vocabulary_usage']['vocab_utilization']:.1%}")

        return eval_stats

    def save_training_info(self, output_dir: str, corpus_stats: dict,
                           eval_stats: dict) -> None:
        """
        Save training information and statistics.

        Args:
            output_dir: Directory to save information
            corpus_stats: Corpus analysis statistics
            eval_stats: Model evaluation statistics
        """
        info = {
            'training_stats': self.training_stats,
            'corpus_stats': corpus_stats,
            'evaluation_stats': eval_stats,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        info_path = os.path.join(output_dir, f"{self.training_stats.get('model_prefix', 'model')}_info.json")

        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, default=str)

        logger.info(f"Training information saved to: {info_path}")

    def test_model_interactively(self, model_path: str) -> None:
        """
        Interactive testing of the trained model.

        Args:
            model_path: Path to the trained model
        """
        logger.info("Loading model for interactive testing...")

        sp = spm.SentencePieceProcessor()
        sp.load(model_path)

        print(f"\n🤖 SentencePiece Model Interactive Tester")
        print(f"Model: {model_path}")
        print(f"Vocabulary size: {sp.vocab_size()}")
        print("Enter packet byte sequences (space-separated decimal or hex), or 'quit' to exit")
        print("Examples:")
        print("  Decimal: 0 1 2 3 255")
        print("  Hex: 00 01 02 03 ff")
        print()

        while True:
            try:
                user_input = input("Packet> ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if not user_input:
                    continue

                # Encode the input
                pieces = sp.encode_as_pieces(user_input)
                ids = sp.encode_as_ids(user_input)

                print(f"  Pieces: {pieces}")
                print(f"  IDs: {ids}")
                print(
                    f"  Compression: {len(user_input.split())}/{len(pieces)} = {len(user_input.split()) / len(pieces):.2f}x")

                # Decode back
                decoded = sp.decode_pieces(pieces)
                print(f"  Decoded: {decoded}")
                print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train SentencePiece model on PCAP corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input corpus file path"
    )

    parser.add_argument(
        "--model-prefix", "-m",
        required=True,
        help="Output model prefix (will create .model and .vocab files)"
    )

    parser.add_argument(
        "--vocab-size", "-v",
        type=int,
        default=8000,
        help="Target vocabulary size"
    )

    parser.add_argument(
        "--model-type", "-t",
        choices=['bpe', 'unigram', 'char', 'word'],
        default='bpe',
        help="SentencePiece model type"
    )

    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=1000000,
        help="Number of sentences to use for training"
    )

    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=4096,
        help="Maximum sentence length in characters"
    )

    parser.add_argument(
        "--character-coverage",
        type=float,
        default=1.0,
        help="Character coverage ratio"
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of threads for training"
    )

    parser.add_argument(
        "--user-symbols",
        nargs='*',
        help="Additional user-defined symbols to include"
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle input sentences"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the trained model"
    )

    parser.add_argument(
        "--eval-samples",
        type=int,
        default=1000,
        help="Number of samples for evaluation"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive testing after training"
    )

    parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not os.path.exists(args.input):
        logger.error(f"Input corpus file does not exist: {args.input}")
        return 1

    if args.vocab_size <= 0:
        logger.error("Vocabulary size must be positive")
        return 1

    # Create output directory if needed
    output_dir = os.path.dirname(args.model_prefix) or '.'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize trainer
    trainer = SentencePieceTrainer()

    try:
        # Validate and analyze corpus
        corpus_stats = trainer.validate_corpus(args.input)

        # Train the model
        logger.info("Starting SentencePiece training...")
        model_path = trainer.train_model(
            corpus_path=args.input,
            model_prefix=args.model_prefix,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            input_sentence_size=args.input_sentence_size,
            shuffle_input_sentence=not args.no_shuffle,
            max_sentence_length=args.max_sentence_length,
            character_coverage=args.character_coverage,
            num_threads=args.num_threads,
            user_defined_symbols=args.user_symbols
        )

        eval_stats = {}

        # Evaluate the model if requested
        if args.evaluate:
            eval_stats = trainer.evaluate_model(
                model_path, args.input, args.eval_samples
            )

        # Save training information
        trainer.save_training_info(output_dir, corpus_stats, eval_stats)

        logger.info("Training completed successfully!")
        logger.info(f"Model files saved:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Vocab: {args.model_prefix}.vocab")

        # Interactive testing if requested
        if args.interactive:
            trainer.test_model_interactively(model_path)

        return 0

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return 1


if __name__ == "__main__":
    exit(main())