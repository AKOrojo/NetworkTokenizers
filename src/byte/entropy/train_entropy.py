import math
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader

# --- Add project root to the Python path ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from itertools import islice
from typing import List
from datetime import datetime
import wandb
from tqdm import tqdm

# --- Imports for DDP ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Import for Mixed Precision ---
from torch.amp import GradScaler, autocast

from src.byte.entropy.corpus_utils import TokenizerCollate, StreamingCorpusDataset
from src.byte.raw.pcap_text_tokenizer import PCAPTextTokenizer
from src.byte.entropy.transformer import LMTransformer, LMTransformerArgs


def setup_distributed():
    """Initializes the distributed environment."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

def create_batched_stream(data_streamer, batch_size: int):
    """Wraps a data streamer to yield batches of a specified size."""
    while True:
        batch = list(islice(data_streamer, batch_size))
        if not batch:
            break
        yield batch


def collate_and_pad_batch(batch: List[List[int]], pad_id: int, max_len: int) -> torch.Tensor:
    """Pads sequences in a batch to the max sequence length."""
    padded_batch = []
    for seq in batch:
        truncated_seq = seq[:max_len]
        padded_seq = truncated_seq + [pad_id] * (max_len - len(truncated_seq))
        padded_batch.append(padded_seq)
    return torch.tensor(padded_batch, dtype=torch.long)


def create_entropy_model(tokenizer: PCAPTextTokenizer) -> (LMTransformer, LMTransformerArgs):
    """Creates an instance of the LMTransformer with specs for a ~10M parameter model."""
    effective_vocab_size = len(tokenizer.get_vocab())

    model_args = LMTransformerArgs(
        dim=512,
        n_layers=3,
        n_heads=8,
        vocab_size=effective_vocab_size,
        max_seqlen=2048,
        sliding_window=1024,
        attn_impl="xformers"
    )

    if dist.get_rank() == 0:
        print("Initializing entropy model with args:")
        print(model_args)

    model = LMTransformer(model_args)
    model.init_weights()
    return model, model_args


def get_lr_scheduler(optimizer, warmup_steps, total_steps, min_ratio):
    """Creates a learning rate scheduler with warmup and cosine decay."""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def train_entropy_model(
        model: DDP,
        data_streamer: DataLoader,
        config: dict,
        rank: int,
        local_rank: int,
        start_step: int,
        optimizer: AdamW,
        scheduler: LambdaLR
):
    """A simple training loop with advanced optimizer, scheduler, and checkpointing."""
    device = torch.device(f"cuda:{local_rank}")
    model.train()
    scaler = GradScaler()

    # Create an iterator from the DataLoader to manually control steps
    data_iterator = iter(data_streamer)

    progress_bar = tqdm(
        range(start_step, config['training_steps']),
        desc="Training",
        disable=(rank != 0),
        unit="step",
        initial=start_step
    )

    for step in progress_bar:
        try:
            # if step % 100 == 0:
            #     dist.barrier()

            accumulated_loss = 0.0
            # Gradient accumulation loop
            for _ in range(config['gradient_accumulation_steps']):
                try:
                    # Get the already-padded tensor batch from the DataLoader
                    padded_batch = next(data_iterator)
                except StopIteration:
                    # Dataloader is exhausted, reset it
                    if rank == 0: print("\nData streamer is exhausted. Resetting.")
                    data_iterator = iter(data_streamer)
                    padded_batch = next(data_iterator)

                # Move batch to device and check if it's empty (from a failed collate)
                padded_batch = padded_batch.to(device)
                # if padded_batch.numel() == 0:
                #     if rank == 0: print("Skipping empty batch.")
                #     continue

                # The batch is already padded, directly create input and target
                input_tensor = padded_batch[:, :-1]
                target_tensor = padded_batch[:, 1:]

                with autocast(device_type="cuda"):
                    loss = model(token_values=input_tensor, target=target_tensor)
                    loss = loss / config['gradient_accumulation_steps']

                accumulated_loss += loss.item()
                scaler.scale(loss).backward()

            # Unscale, step, and update after accumulating gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({"loss": accumulated_loss, "learning_rate": current_lr, "step": step})
                progress_bar.set_postfix(loss=f"{accumulated_loss:.4f}", lr=f"{current_lr:.2e}")

                if (step + 1) % config['checkpoint_freq'] == 0:
                    checkpoint_path = Path(config['checkpoint_dir']) / f"checkpoint_step_{step + 1}.pt"
                    torch.save({
                        'step': step + 1,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict()
                    }, checkpoint_path)
                    print(f"\nSaved checkpoint to {checkpoint_path}")

        except StopIteration:
            if rank == 0: print("Data streamer finished early.")
            break
        except Exception as e:
            if rank == 0: print(f"Error at step {step}: {e}. Skipping.")
            continue

    if rank == 0:
        print("Training finished.")


if __name__ == "__main__":
    rank, world_size, local_rank = setup_distributed()

    train_config = {
        "pcap_directory": "/home/abanisenioluwa_oroj1/Downloads/flows",
        "pcap_data_ratio": 1.0,
        "training_steps": 100000,
        "learning_rate": 4e-4,
        "warmup_steps": 500,
        "batch_size_per_gpu": 64,
        "gradient_accumulation_steps": 12,
        "checkpoint_freq": 500,
        "checkpoint_dir": "checkpoints"
    }

    if rank == 0:
        Path(train_config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    pcap_tokenizer = PCAPTextTokenizer()

    entropy_model, model_args = create_entropy_model(pcap_tokenizer)
    entropy_model = torch.compile(entropy_model)
    entropy_model.to(local_rank)
    ddp_model = DDP(entropy_model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = AdamW(
        ddp_model.parameters(),
        lr=train_config['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8
    )

    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=train_config['warmup_steps'],
        total_steps=train_config['training_steps'],
        min_ratio=0.1
    )

    # --- NEW DATA PIPELINE ---
    NUM_WORKERS = 2

    # 1. Instantiate your custom dataset
    streaming_dataset = StreamingCorpusDataset(
        pcap_dir=train_config['pcap_directory'],
        pcap_ratio=train_config['pcap_data_ratio']
    )

    # 2. Instantiate your custom collate function
    tokenizer_collate_fn = TokenizerCollate(
        tokenizer=pcap_tokenizer,
        max_len=model_args.max_seqlen
    )

    # 3. Create the DataLoader
    train_loader = DataLoader(
        streaming_dataset,
        batch_size=train_config['batch_size_per_gpu'],
        num_workers=NUM_WORKERS,
        collate_fn=tokenizer_collate_fn,
        pin_memory=True,
        prefetch_factor=2
    )

    start_step = 0
    checkpoint_dir = Path(train_config['checkpoint_dir'])
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"), key=os.path.getmtime)
        if checkpoints:
            latest_checkpoint_path = checkpoints[-1]
            if rank == 0:
                print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=f"cuda:{local_rank}")
            ddp_model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_step = checkpoint['step']
            if 'scaler_state_dict' in checkpoint:
                scaler = GradScaler()
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

    if rank == 0:
        project_name = "blt_entropy_model_pcap_text"
        run_name = f"entropy-run-full-config-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        global_batch_size_bytes = (
                train_config['batch_size_per_gpu'] * model_args.max_seqlen * world_size * train_config[
            'gradient_accumulation_steps']
        )
        wandb_config = train_config.copy()
        wandb_config["global_batch_size_bytes"] = global_batch_size_bytes
        wandb_config["world_size"] = world_size
        wandb_config["model_args"] = model_args.model_dump()
        wandb.init(project=project_name, name=run_name, config=wandb_config, resume="allow", id=run_name)


    train_entropy_model(
        model=ddp_model,
        data_streamer=train_loader,
        config=train_config,
        rank=rank,
        local_rank=local_rank,
        start_step=start_step,
        optimizer=optimizer,
        scheduler=scheduler
    )

    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "entropy_model_trained.pt")
        print("Trained entropy model saved to entropy_model_trained.pt")
        wandb.finish()

    cleanup_distributed()