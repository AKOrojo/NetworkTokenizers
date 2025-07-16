# main_train_entropy.py

import torch
from torch.optim import AdamW
from pathlib import Path
from itertools import cycle
import random
from datasets import load_dataset
from typing import cast

from src.byte.raw.pcap_text_tokenizer import PCAPTextTokenizer
from src.byte.entropy.transformer import LMTransformer, LMTransformerArgs


def stream_mixed_corpus(pcap_dir: str, tokenizer: PCAPTextTokenizer, pcap_ratio: float = 0.5):
    """
    A generator that streams and tokenizes data from C4 and a PCAP directory.
    """
    # 1. Load C4 dataset in streaming mode
    c4_stream = load_dataset("allenai/c4", "en", streaming=True, split="train")
    c4_iterator = iter(c4_stream)

    # 2. Find all PCAP files
    pcap_files = list(Path(pcap_dir).glob("**/*.pcap")) + list(Path(pcap_dir).glob("**/*.pcapng"))
    if not pcap_files:
        raise FileNotFoundError(f"No .pcap files found in {pcap_dir}")
    pcap_iterator = cycle(pcap_files)

    print(f"Found {len(pcap_files)} PCAP files.")
    print("Starting mixed data stream...")

    # 3. Main streaming loop
    while True:
        if random.random() < pcap_ratio:
            pcap_file = next(pcap_iterator)
            tokens = tokenizer.tokenize_pcap(pcap_file)
        else:
            try:
                # Cast the item to a dict to help the type checker
                sample_dict = cast(dict, next(c4_iterator))
                text_sample = sample_dict["text"]
                tokens = tokenizer.tokenize(text_sample)
            except StopIteration:
                print("C4 stream finished. Resetting iterator.")
                c4_iterator = iter(c4_stream)
                continue

        if tokens:
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            yield token_ids


def create_entropy_model(tokenizer: PCAPTextTokenizer) -> LMTransformer:
    """Creates an instance of the LMTransformer to be used as an entropy model."""
    effective_vocab_size = tokenizer.vocab_size + tokenizer.offset

    model_args = LMTransformerArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=effective_vocab_size,
        max_seqlen=1024,
        sliding_window=512
    )

    print("Initializing entropy model with args:")
    print(model_args)

    model = LMTransformer(model_args)
    # The init_weights method is defined in the provided transformer.py
    model.init_weights()  #
    return model


# FIX 1: Removed the unused 'tokenizer' parameter from the function definition.
def train_entropy_model(model: LMTransformer, data_streamer, num_steps: int = 10000, lr: float = 1e-4):
    """
    A simple training loop for the entropy model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)

    for step in range(num_steps):
        try:
            token_ids = next(data_streamer)

            if len(token_ids) > model.args.max_seqlen:
                token_ids = token_ids[:model.args.max_seqlen]

            # Ensure sequence is long enough to create input and target
            if len(token_ids) < 2:
                continue

            input_tensor = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
            target_tensor = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

            # FIX 2: Changed model(...) to model.forward(...)
            # The LMTransformer's forward pass returns cross-entropy loss when a target is provided
            loss = model.forward(token_values=input_tensor, target=target_tensor)

            if step % 100 == 0:
                print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        except Exception as e:
            print(f"Error at step {step}: {e}. Skipping sample.")
            continue

    print("Training finished.")


if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Update this path to your actual PCAP directory
    PCAP_DIRECTORY = "/path/to/your/pcap_dir"
    PCAP_DATA_RATIO = 0.5
    TRAINING_STEPS = 50000
    LEARNING_RATE = 4e-4

    pcap_tokenizer = PCAPTextTokenizer()

    data_generator = stream_mixed_corpus(
        pcap_dir=PCAP_DIRECTORY,
        tokenizer=pcap_tokenizer,
        pcap_ratio=PCAP_DATA_RATIO
    )

    entropy_model = create_entropy_model(pcap_tokenizer)

    # FIX 3: Removed 'tokenizer' from the function call as it's no longer a parameter.
    train_entropy_model(
        model=entropy_model,
        data_streamer=data_generator,
        num_steps=TRAINING_STEPS,
        lr=LEARNING_RATE
    )

    torch.save(entropy_model.state_dict(), "entropy_model_trained.pt")
    print("Trained entropy model saved to entropy_model_trained.pt")