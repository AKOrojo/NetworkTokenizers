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
import time
import wandb
from tqdm import tqdm

# --- Imports for DDP ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Import for Mixed Precision ---
from torch.amp import GradScaler, autocast

# --- UPDATED IMPORTS FOR ON-THE-FLY TOKENIZATION ---
from corpus_utils_onthefly import RawPCAPDataset, OnTheFlyCollate, create_dataloader_from_raw
from src.byte.raw.token_pcap_byte_tokenizer import TokenPCAPByteTokenizer
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


def dist_mean(tensor):
    """Calculate mean across all processes."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


def dist_sum(tensor, reduce_dtype=None):
    """Calculate sum across all processes."""
    if dist.is_initialized():
        if reduce_dtype is not None:
            original_dtype = tensor.dtype
            tensor = tensor.to(reduce_dtype)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if reduce_dtype is not None:
            tensor = tensor.to(original_dtype)
    return tensor


def create_entropy_model(tokenizer: TokenPCAPByteTokenizer, rank: int) -> (LMTransformer, LMTransformerArgs):
    """Creates an instance of the LMTransformer following BLT's architecture."""
    effective_vocab_size = tokenizer.vocab_size

    # BLT-matched architecture
    model_args = LMTransformerArgs(
        dim=768,
        n_layers=14,
        n_heads=12,
        vocab_size=effective_vocab_size,
        max_seqlen=8192,
        sliding_window=512,
        ffn_dim_multiplier=1.0,
        attn_bias_type="local_block_causal",
        attn_impl="xformers"
    )

    if rank == 0:
        print("Initializing transformer model with args:")
        print(model_args)

    model = LMTransformer(model_args)
    model.init_weights()

    # Calculate actual parameter count
    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model has {total_params:,} parameters ({total_params / 1e6:.1f}M)")

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


def evaluate_model_blt_style(model, eval_loader, device, rank, world_size):
    """Evaluate entropy model with BLT-style metrics"""
    model.eval()
    total_loss = 0.0
    total_bytes = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device, non_blocking=True)

            # Handle chunks longer than max_seqlen
            if batch.size(1) > 8192:
                batch = batch[:, :8192]

            input_tensor = batch[:, :-1]
            target_tensor = batch[:, 1:]

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(token_values=input_tensor, target=target_tensor)

            total_loss += loss.item() * target_tensor.numel()
            total_bytes += target_tensor.numel()  # For byte-level: tokens = bytes

    # Local metrics
    local_avg_loss = total_loss / total_bytes
    local_bpb = local_avg_loss / math.log(2)
    local_perplexity = math.exp(local_avg_loss)

    # Global metrics (aggregated across processes)
    if world_size > 1:
        global_total_loss = dist_sum(torch.tensor(total_loss, device=device))
        global_total_bytes = dist_sum(torch.tensor(total_bytes, device=device))
        global_avg_loss = global_total_loss / global_total_bytes
        global_bpb = global_avg_loss / math.log(2)
        global_perplexity = torch.exp(global_avg_loss)

        # Convert to Python numbers
        global_avg_loss = global_avg_loss.item()
        global_bpb = global_bpb.item()
        global_perplexity = global_perplexity.item()
    else:
        global_avg_loss = local_avg_loss
        global_bpb = local_bpb
        global_perplexity = local_perplexity

    if rank == 0:
        print(f"Evaluation Results:")
        print(f"  Local - Loss: {local_avg_loss:.4f}, BPB: {local_bpb:.4f}, PPL: {local_perplexity:.2f}")
        print(f"  Global - Loss: {global_avg_loss:.4f}, BPB: {global_bpb:.4f}, PPL: {global_perplexity:.2f}")

    return {
        'eval_loss_local': local_avg_loss,
        'eval_loss_global': global_avg_loss,
        'bpb_local': local_bpb,
        'bpb_global': global_bpb,
        'perplexity_local': local_perplexity,
        'perplexity_global': global_perplexity,
    }


def train_entropy_model(
        model: DDP,
        data_loader: DataLoader,
        config: dict,
        rank: int,
        local_rank: int,
        world_size: int,
        start_step: int,
        optimizer: AdamW,
        scheduler: LambdaLR,
        tokenizer: TokenPCAPByteTokenizer
):
    """Training loop optimized for on-the-fly tokenization and 2x A5000 32GB."""
    device = torch.device(f"cuda:{local_rank}")
    model.train()
    scaler = GradScaler()

    # Create an iterator from the DataLoader to manually control steps
    data_iterator = iter(data_loader)

    progress_bar = tqdm(
        range(start_step, config['training_steps']),
        desc="Training",
        disable=(rank != 0),
        unit="step",
        initial=start_step
    )

    # Track tokenization performance
    tokenization_times = []
    batch_load_times = []

    for step in progress_bar:
        try:
            accumulated_loss = 0.0
            step_start_time = time.time()

            # Gradient accumulation loop
            for micro_step in range(config['gradient_accumulation_steps']):
                try:
                    # Get batch with on-the-fly tokenization
                    batch_start_time = time.time()
                    batch = next(data_iterator)
                    batch_load_time = time.time() - batch_start_time
                    batch_load_times.append(batch_load_time)

                except StopIteration:
                    # DataLoader exhausted, reset it
                    if rank == 0:
                        print(f"\nDataLoader exhausted at step {step}, micro_step {micro_step}. Resetting.")
                    data_iterator = iter(data_loader)
                    batch = next(data_iterator)

                # Move to device
                batch = batch.to(device, non_blocking=True)

                # Handle chunks that are longer than model's max_seqlen
                if batch.size(1) > config['max_seqlen']:
                    # Randomly sample a window from the chunk for variety
                    start_idx = torch.randint(0, batch.size(1) - config['max_seqlen'] + 1, (1,)).item()
                    batch = batch[:, start_idx:start_idx + config['max_seqlen']]

                # Create input and target tensors
                input_tensor = batch[:, :-1]
                target_tensor = batch[:, 1:]

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(token_values=input_tensor, target=target_tensor)
                    loss = loss / config['gradient_accumulation_steps']

                accumulated_loss += loss.item()
                scaler.scale(loss).backward()

            # Gradient clipping and optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # More aggressive memory cleanup to prevent CUDA corruption
            if (step + 1) % 50 == 0:  # Every 50 steps instead of 100
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                # Additional cleanup for multi-GPU
                if world_size > 1:
                    dist.barrier()  # Synchronize all processes

            step_time = time.time() - step_start_time

            # Logging, evaluation, and checkpointing
            if rank == 0:
                current_lr = scheduler.get_last_lr()[0]

                # Calculate performance metrics
                avg_batch_load_time = sum(batch_load_times[-config['gradient_accumulation_steps']:]) / config[
                    'gradient_accumulation_steps']

                # Log training metrics
                wandb.log({
                    "train_loss": accumulated_loss,
                    "learning_rate": current_lr,
                    "step": step,
                    "tokens_processed": (step + 1) * config['effective_batch_tokens'],
                    "step_time": step_time,
                    "avg_batch_load_time": avg_batch_load_time,
                    "tokenization_overhead": avg_batch_load_time / step_time if step_time > 0 else 0
                })

                progress_bar.set_postfix(
                    loss=f"{accumulated_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    tokens=f"{(step + 1) * config['effective_batch_tokens'] / 1e6:.1f}M",
                    load_time=f"{avg_batch_load_time:.3f}s"
                )

                # Evaluation every eval_freq steps
                if (step + 1) % config['eval_freq'] == 0:
                    print(f"\nRunning evaluation at step {step + 1}...")

                    # Create small eval subset
                    eval_dataset = RawPCAPDataset(
                        raw_data_dir=config['raw_data_dir'],
                        tokenizer=tokenizer,
                        chunk_size=config['chunk_size'],
                        max_length=config['max_seqlen'],
                        cache_tokenized=True,  # Enable cache for eval
                        cache_size=500
                    )

                    # Use only a subset for evaluation
                    eval_size = min(100, len(eval_dataset))
                    eval_indices = list(range(eval_size))
                    eval_subset = torch.utils.data.Subset(eval_dataset, eval_indices)

                    eval_collate_fn = OnTheFlyCollate(
                        pad_token_id=tokenizer.pad_token_id,
                        max_length=config['max_seqlen']
                    )

                    eval_loader = DataLoader(
                        eval_subset,
                        batch_size=config['batch_size_per_gpu'],
                        shuffle=False,
                        num_workers=2,
                        collate_fn=eval_collate_fn
                    )

                    eval_metrics = evaluate_model_blt_style(
                        model, eval_loader, device, rank, world_size
                    )

                    # Log evaluation metrics
                    wandb.log({
                        "eval_loss_local": eval_metrics['eval_loss_local'],
                        "eval_loss_global": eval_metrics['eval_loss_global'],
                        "eval_bpb_local": eval_metrics['bpb_local'],
                        "eval_bpb_global": eval_metrics['bpb_global'],
                        "eval_perplexity_local": eval_metrics['perplexity_local'],
                        "eval_perplexity_global": eval_metrics['perplexity_global'],
                        "step": step
                    })

                    # Log cache statistics if available
                    if hasattr(data_loader.dataset, 'get_cache_stats'):
                        cache_stats = data_loader.dataset.get_cache_stats()
                        wandb.log({
                            "cache_size": cache_stats.get('cache_size', 0),
                            "cache_hit_ratio": cache_stats.get('cache_hit_ratio', 0),
                            "step": step
                        })

                    model.train()  # Back to training mode

                # Checkpointing
                if (step + 1) % config['checkpoint_freq'] == 0:
                    checkpoint_path = Path(config['checkpoint_dir']) / f"checkpoint_step_{step + 1}.pt"
                    torch.save({
                        'step': step + 1,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'config': config,
                        'tokenizer_config': {
                            'vocab_size': tokenizer.vocab_size,
                            'pad_token_id': tokenizer.pad_token_id
                        }
                    }, checkpoint_path)
                    print(f"\nSaved checkpoint to {checkpoint_path}")

        except Exception as e:
            if rank == 0:
                print(f"Error at step {step}: {e}")

                # Check if it's a CUDA error
                if "CUDA" in str(e):
                    print("CUDA error detected. Attempting recovery...")
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        print("CUDA cache cleared. Continuing training...")
                    except:
                        print("CUDA recovery failed. Skipping step.")
                else:
                    print("Skipping step.")
            continue

    if rank == 0:
        print("Training finished.")
        print(f"Average batch load time: {sum(batch_load_times) / len(batch_load_times):.3f}s")


if __name__ == "__main__":
    import time

    # Initialize distributed training (even for single GPU)
    rank, world_size, local_rank = setup_distributed()

    # CONSERVATIVE CONFIG FOR 2x A5000 32GB - STABILITY FOCUSED
    train_config = {
        # Data paths
        "raw_data_dir": "/home/abanisenioluwa_oroj1/Downloads/flows",  # Updated path

        # Tokenization settings
        "chunk_size": 8192,  # Size of chunks to create from files (bytes)
        "cache_tokenized": True,  # Cache tokenized chunks in memory
        "cache_size": 1200,  # Very conservative for stability

        # Training hyperparameters - CONSERVATIVE FOR STABILITY
        "training_steps": 5000,
        "learning_rate": 2e-4,  # BLT's proven LR
        "warmup_steps": 100,  # BLT's warmup
        "batch_size_per_gpu": 3,  # Reduced back to conservative
        "gradient_accumulation_steps": 27,  # 3 * 27 * 8192 * 2 = 1,327,104 tokens (~1.33M)
        "max_seqlen": 8192,
        "gradient_clip_norm": 10.0,  # BLT's setting
        "eval_freq": 500,  # Less frequent evaluation

        # Checkpointing
        "checkpoint_freq": 250,  # Every 250 steps
        "checkpoint_dir": "checkpoints_onthefly",

        # Performance settings - VERY CONSERVATIVE
        "num_workers": 1,  # Minimal workers to reduce complexity
        "prefetch_factor": 1,  # Minimal prefetch
    }

    # Calculate effective batch size in tokens
    train_config['effective_batch_tokens'] = (
            train_config['batch_size_per_gpu'] *
            train_config['gradient_accumulation_steps'] *
            train_config['max_seqlen'] *
            world_size
    )

    if rank == 0:
        print(f"Training Configuration (2x A5000 32GB - Conservative/Stable):")
        print(f"  Raw data directory: {train_config['raw_data_dir']}")
        print(f"  Chunk size: {train_config['chunk_size']} bytes")
        print(f"  Tokenization cache: {train_config['cache_tokenized']} (max {train_config['cache_size']} chunks)")
        print(f"  Training steps: {train_config['training_steps']:,}")
        print(f"  Learning rate: {train_config['learning_rate']} (BLT-matched)")
        print(f"  Warmup steps: {train_config['warmup_steps']} (BLT-matched)")
        print(f"  Gradient clip: {train_config['gradient_clip_norm']} (BLT-matched)")
        print(f"  Mixed precision: bfloat16")
        print(f"  Evaluation frequency: every {train_config['eval_freq']} steps")
        print(f"  Batch size per GPU: {train_config['batch_size_per_gpu']}")
        print(f"  Gradient accumulation steps: {train_config['gradient_accumulation_steps']}")
        print(
            f"  Effective batch size: {train_config['effective_batch_tokens']:,} tokens ({train_config['effective_batch_tokens'] / 1e6:.2f}M)")
        print(f"  Max sequence length: {train_config['max_seqlen']}")
        print(f"  Model: 768d, 14L, 12H (~100M params)")
        print(f"  Total tokens: {train_config['training_steps'] * train_config['effective_batch_tokens'] / 1e9:.1f}B")
        print(f"  World size: {world_size}")
        print(f"  Hardware: 2x A5000 32GB (CONSERVATIVE CONFIG for stability)")
        print(f"  Memory utilization: ~60-70% per GPU (reduced for stability)")

        # Hardware diagnostics
        print(f"\n=== GPU Hardware Check ===")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory // 1024 ** 3} GB")
            print(f"    SM: {props.major}.{props.minor}")
            print(f"    Multiprocessors: {props.multi_processor_count}")

            # Test GPU memory allocation
            try:
                test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
                del test_tensor
                torch.cuda.empty_cache()
                print(f"    Status: ✅ OK")
            except Exception as e:
                print(f"    Status: ❌ ERROR - {e}")
        print(f"=== End GPU Check ===\n")

        Path(train_config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer
    tokenizer = TokenPCAPByteTokenizer()
    if rank == 0:
        print(f"Tokenizer initialized: vocab_size={tokenizer.vocab_size}, pad_token_id={tokenizer.pad_token_id}")

    # Create model (WITHOUT torch.compile for A5000 compatibility)
    entropy_model, model_args = create_entropy_model(tokenizer, rank)
    # entropy_model = torch.compile(entropy_model)  # DISABLED: Causes issues with xformers on A5000
    entropy_model.to(local_rank)
    ddp_model = DDP(entropy_model, device_ids=[local_rank], find_unused_parameters=False)

    # Initialize optimizer
    optimizer = AdamW(
        ddp_model.parameters(),
        lr=train_config['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8
    )

    # Initialize scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=train_config['warmup_steps'],
        total_steps=train_config['training_steps'],
        min_ratio=0.1
    )

    # --- ON-THE-FLY TOKENIZATION PIPELINE ---
    if rank == 0:
        print("Setting up on-the-fly tokenization pipeline...")

    # Create dataset with on-the-fly tokenization
    train_loader = create_dataloader_from_raw(
        raw_data_dir=train_config['raw_data_dir'],
        tokenizer=tokenizer,
        batch_size=train_config['batch_size_per_gpu'],
        chunk_size=train_config['chunk_size'],
        max_length=train_config['max_seqlen'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        use_iterable=False,
        cache_tokenized=train_config['cache_tokenized'],
        cache_size=train_config['cache_size']
    )

    if rank == 0:
        print(f"Dataset loaded: {len(train_loader.dataset):,} chunks")
        print(f"DataLoader created with {len(train_loader):,} batches per epoch")

    # Load checkpoint if exists
    start_step = 0
    checkpoint_dir = Path(train_config['checkpoint_dir'])
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"), key=os.path.getmtime)
        if checkpoints:
            latest_checkpoint_path = checkpoints[-1]
            if rank == 0:
                print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=f"cuda:{local_rank}")
            ddp_model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_step = checkpoint['step']
            if 'scaler_state_dict' in checkpoint:
                scaler = GradScaler()
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Initialize wandb
    if rank == 0:
        project_name = "pcap_entropy_model_2xa5000_onthefly"
        run_name = f"2xa5000-onthefly-{train_config['effective_batch_tokens'] // 1000}k-tokens-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        wandb_config = train_config.copy()
        wandb_config["model_args"] = model_args.model_dump()
        wandb_config["world_size"] = world_size
        wandb_config["dataset_size"] = len(train_loader.dataset)
        wandb_config["tokenizer_vocab_size"] = tokenizer.vocab_size

        wandb.init(
            project=project_name,
            name=run_name,
            config=wandb_config,
            resume="allow",
            id=run_name
        )

    # Start training
    if rank == 0:
        print("Starting training with on-the-fly tokenization on 2x A5000 32GB...")
        print("NOTE: torch.compile disabled for xformers compatibility")
        print("CONSERVATIVE configuration for maximum stability")
        print("If CUDA errors persist, consider single GPU fallback (see debug guide)")
        print("=" * 60)

    train_entropy_model(
        model=ddp_model,
        data_loader=train_loader,
        config=train_config,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        start_step=start_step,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer
    )

    # Save final model with proper error handling and memory cleanup
    if rank == 0:
        final_model_path = "pcap_entropy_model_2xa5000_onthefly_final.pt"

        try:
            print("Synchronizing CUDA and clearing cache before saving...")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Move model to CPU to avoid CUDA corruption during save
            print("Moving model to CPU for safe saving...")
            cpu_model_state = {}
            for key, value in ddp_model.module.state_dict().items():
                cpu_model_state[key] = value.cpu()

            print("Saving final model...")
            torch.save({
                'model_state_dict': cpu_model_state,
                'model_args': model_args.model_dump(),
                'tokenizer_config': {
                    'vocab_size': tokenizer.vocab_size,
                    'pad_token_id': tokenizer.pad_token_id
                },
                'training_completed': True,
                'final_step': train_config['training_steps']
            }, final_model_path)

            print(f"Final model saved successfully to {final_model_path}")

        except Exception as e:
            print(f"Error saving final model: {e}")
            print("Attempting alternative save method...")

            try:
                # Alternative: save just the state dict
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                alternative_path = "pcap_entropy_model_2xa5000_statedict_only.pt"
                torch.save(ddp_model.module.cpu().state_dict(), alternative_path)
                print(f"Alternative save successful: {alternative_path}")
            except Exception as e2:
                print(f"Alternative save also failed: {e2}")
                print("Model training completed but final save failed. Check checkpoints directory for saved models.")

        wandb.finish()

    cleanup_distributed()