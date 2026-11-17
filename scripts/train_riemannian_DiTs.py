"""
Riemannian Flow Matching (RFM) DiTs training script using PyTorch Lightning.

Trains the Riemannian DiTs model on particle physics events: conditional
flow matching with velocity prediction on the per-particle product manifold
R (log energy-scale) × S² (direction) × R (log mass, massive types only).

Usage:
    # Training from scratch
    uv run python scripts/train_riemannian_DiTs.py --config configs/train_riemannian_DiTs_2to4lep.yaml

    # Quick test
    uv run python scripts/train_riemannian_DiTs.py --config configs/train_riemannian_DiTs_2to4lep.yaml --test-mode --epochs 3

    # Resume training from checkpoint
    uv run python scripts/train_riemannian_DiTs.py --config configs/train_riemannian_DiTs_2to4lep.yaml \
        --resume-from checkpoints/riemannian_DiTs/last.ckpt --wandb-id <original_run_id>

    # Finetune from pretrained checkpoint (fresh optimizer/scheduler, epoch 0)
    uv run python scripts/train_riemannian_DiTs.py --config configs/train_riemannian_DiTs_2to4lep.yaml \
        --finetune-from checkpoints/riemannian_DiTs/best.ckpt
"""

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shellflow.data import ParticleDataset
from shellflow.data.particle_vector_collator import (
    ParticleVectorCollator,
    ParticleVectorCollatorGeometric,
)
from shellflow.data.token_packing_sampler import (
    TokenPackingSampler,
    pad_batch_to_static,
)
from shellflow.models.riemannian_DiTs_model import create_riemannian_dits
from shellflow.training import get_default_callbacks
from shellflow.training.riemannian_DiTs_lightning import RiemannianDiTsModule
from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Riemannian Diffusion Transformer training with PyTorch Lightning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Checkpoint to resume from"
    )
    parser.add_argument(
        "--finetune-from",
        type=str,
        default=None,
        help="Checkpoint to finetune from (weights only)",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name from dataset_registry.yaml (e.g. 4lep, exactly3lep_reduced)",
    )
    parser.add_argument(
        "--config-path", type=str, default=None, help="Path to dataloader_config.yaml"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--test-mode", action="store_true", help="Quick test with limited events"
    )
    parser.add_argument(
        "--test-mode-samples",
        type=int,
        default=1000,
        help="Number of samples in test mode",
    )
    parser.add_argument(
        "--lazy-loading", action="store_true", default=None, help="Use lazy loading"
    )
    parser.add_argument("--no-lazy-loading", dest="lazy_loading", action="store_false")

    # Training
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument(
        "--grad-clip", type=float, default=None, help="Gradient clipping max norm"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["cosine", "onecycle", "constant"],
    )

    # Model
    parser.add_argument("--d-model", type=int, default=None, help="Model dimension")
    parser.add_argument(
        "--num-heads", type=int, default=None, help="Number of attention heads"
    )
    parser.add_argument(
        "--num-layers", type=int, default=None, help="Number of transformer layers"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Directory to save checkpoints"
    )

    # Logging
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="W&B project name"
    )
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    parser.add_argument(
        "--wandb-id", type=str, default=None, help="W&B run ID to resume"
    )

    # torch.compile
    parser.add_argument(
        "--compile-model",
        action="store_true",
        default=None,
        help="Enable torch.compile",
    )
    parser.add_argument(
        "--no-compile-model", dest="compile_model", action="store_false"
    )

    # Geometric (sparse) batching
    parser.add_argument(
        "--geometric",
        action="store_true",
        help="Use geometric (sparse) collator — no padding, varlen attention",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Use flat compiled pipeline — token packing + torch.compile + varlen attention",
    )

    # Generation
    parser.add_argument(
        "--no-generation", action="store_true", help="Disable generation monitoring"
    )
    parser.add_argument("--gen-num-samples", type=int, default=None)
    parser.add_argument("--gen-every-n-epochs", type=int, default=None)

    # System
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu", "auto"]
    )
    parser.add_argument(
        "--precision", type=str, default=None, choices=["32", "16-mixed", "bf16-mixed"]
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="DataLoader workers"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    return parser.parse_args()


def load_config_from_yaml(path: str) -> dict:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    torch.set_float32_matmul_precision("medium")

    args = parse_args()

    # Load YAML config
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        cfg = load_config_from_yaml(args.config)
    else:
        cfg = {}

    # Extract config sections
    model_cfg = cfg.get("model", {})
    opt_cfg = cfg.get("optimizer", {})
    sched_cfg = cfg.get("scheduler", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    wandb_cfg = cfg.get("wandb", {})
    gen_cfg = cfg.get("generation", {})
    es_cfg = cfg.get("early_stopping", {})
    quality_weighting_cfg = cfg.get("quality_weighting", {})

    # Model params (CLI overrides YAML)
    d_model = args.d_model or model_cfg.get("d_model", 128)
    num_heads = args.num_heads or model_cfg.get("num_heads", 8)
    num_layers = args.num_layers or model_cfg.get("num_layers", 6)
    max_particles = model_cfg.get("max_particles", 30)
    mlp_ratio = model_cfg.get("mlp_ratio", 4.0)
    dropout = model_cfg.get("dropout", 0.1)
    num_extra_features = model_cfg.get("num_extra_features", 13)
    met_embed_dim = model_cfg.get("met_embed_dim", 64)
    num_met_tokens = model_cfg.get("num_met_tokens", 4)

    # RFM params
    ema_decay = model_cfg.get("ema_decay", None)
    sigma_prior = model_cfg.get("sigma_prior", 1.0)

    # Optimizer params
    lr = args.lr or opt_cfg.get("lr", 3e-4)
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else opt_cfg.get("weight_decay", 0.01)
    )
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = opt_cfg.get("eps", 1e-8)

    # Scheduler params
    scheduler_name = args.scheduler or sched_cfg.get("name", "cosine")
    warmup_pct = sched_cfg.get("warmup_pct", 0.05)
    warmup_steps = sched_cfg.get("warmup_steps", None)
    min_lr = sched_cfg.get("min_lr", 1e-6)

    # Data params — resolve from dataset registry
    import yaml

    dataset_name = args.dataset or data_cfg.get("dataset")
    if not dataset_name:
        raise ValueError(
            "Must specify dataset via --dataset flag or data.dataset in config YAML"
        )
    registry_path = (
        Path(__file__).parent.parent / "src/shellflow/config/dataset_registry.yaml"
    )
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    if dataset_name not in registry:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {list(registry.keys())}"
        )
    ds_entry = registry[dataset_name]
    config_path = args.config_path or ds_entry["config_path"]
    data_dir_override = ds_entry["data_dir"]
    logger.info(f"Dataset '{dataset_name}' resolved from registry")
    train_split = data_cfg.get("train_split", 0.95)
    batch_size = args.batch_size or data_cfg.get("batch_size", 1024)
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else data_cfg.get("num_workers", 4)
    )
    lazy_loading = (
        args.lazy_loading
        if args.lazy_loading is not None
        else data_cfg.get("lazy_loading", True)
    )
    normalize = data_cfg.get("normalize", False)
    pin_memory = data_cfg.get("pin_memory", True)
    persistent_workers = data_cfg.get("persistent_workers", True)
    prefetch_factor = data_cfg.get("prefetch_factor", 2)

    # Auto-scale workers to avoid I/O contention with multi-GPU
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if n_gpus > 1:
        cpu_count = os.cpu_count() or 8
        num_workers = min(num_workers, max(2, cpu_count // n_gpus))
        logger.info(
            f"Auto-scaled num_workers to {num_workers} ({n_gpus} GPUs, {cpu_count} CPUs)"
        )

    # Training params
    epochs = args.epochs or train_cfg.get("epochs", 100)
    grad_clip = (
        args.grad_clip
        if args.grad_clip is not None
        else train_cfg.get("grad_clip", 2.0)
    )
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    log_every_n_steps = train_cfg.get("log_every_n_steps", 50)
    val_checks_per_epoch = train_cfg.get("val_checks_per_epoch", 1)
    # Optional virtual-epoch cap: limit training batches per epoch so the
    # epoch-keyed callbacks (validation, checkpoint, generation, higgs) fire at a
    # sane cadence on very large datasets where a natural epoch is huge. Default
    # None = no cap (full natural epoch, unchanged behavior for existing runs).
    limit_train_batches = train_cfg.get("limit_train_batches", None)
    # Cap validation batches per check (spot-check val on huge datasets so a
    # frequent virtual epoch doesn't pay a full-val-set tax). Default None = full.
    limit_val_batches = train_cfg.get("limit_val_batches", None)
    compile_model = (
        args.compile_model
        if args.compile_model is not None
        else train_cfg.get("compile_model", True)
    )
    compile_mode = train_cfg.get("compile_mode", "default")
    compile_dynamic = train_cfg.get("compile_dynamic", False)
    test_mode = args.test_mode or train_cfg.get("test_mode", False)
    test_mode_samples = args.test_mode_samples or train_cfg.get(
        "test_mode_samples", 1000
    )

    # Checkpoint params
    checkpoint_dir = args.checkpoint_dir or ckpt_cfg.get(
        "dir", "checkpoints/riemannian_DiTs"
    )
    save_every_n_epochs = ckpt_cfg.get("save_every_n_epochs", 10)
    save_top_k = ckpt_cfg.get("save_top_k", 3)

    # Early stopping
    early_stopping_enabled = es_cfg.get("enabled", False)
    early_stopping_patience = es_cfg.get("patience", 10)
    early_stopping_min_delta = es_cfg.get("min_delta", 0.0001)

    # Generation params
    generation_enabled = not args.no_generation and gen_cfg.get("enabled", True)
    gen_num_samples = args.gen_num_samples or gen_cfg.get("num_samples", 10_000)
    gen_batch_size = gen_cfg.get("batch_size", 64)
    gen_n_steps = gen_cfg.get("n_gen_steps", 50)
    gen_eta = gen_cfg.get("eta", 0.0)
    gen_every_n_epochs = args.gen_every_n_epochs or gen_cfg.get("every_n_epochs", 5)

    # W&B params
    use_wandb = not args.no_wandb and wandb_cfg.get("enabled", True)
    wandb_project = args.wandb_project or wandb_cfg.get(
        "project", "shellflow_diffusion"
    )
    wandb_name = args.wandb_name or wandb_cfg.get("name", None)
    wandb_id = args.wandb_id or wandb_cfg.get("run_id", None)
    wandb_tags = wandb_cfg.get("tags", ["riemannian"])

    # System
    device = args.device or cfg.get("device", "auto")
    precision = args.precision or cfg.get("precision", "bf16-mixed")

    # Resume/finetune
    resume_ckpt_path = args.resume_from or cfg.get("resume", {}).get(
        "checkpoint_path", None
    )
    finetune_ckpt_path = args.finetune_from or cfg.get("resume", {}).get(
        "finetune_from", None
    )

    if resume_ckpt_path and finetune_ckpt_path:
        raise ValueError(
            "Cannot use both checkpoint_path (resume) and finetune_from simultaneously"
        )

    if test_mode:
        wandb_tags = list(wandb_tags) + ["test"]

    logger.info("=" * 70)
    logger.info("Riemannian Diffusion Transformer Training")
    logger.info("=" * 70)

    L.seed_everything(seed, workers=True)

    # Accelerator
    if device == "auto":
        accelerator = "auto"
        devices = "auto"
    elif device == "cuda":
        accelerator = "gpu"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    # DDP rank/world_size (set by Lightning's DDP launcher or torchrun).
    # Lightning's "ddp" subprocess launcher keeps GLOBAL_RANK 0 in the original
    # parent process and only re-executes the script for ranks 1..N-1. In the
    # parent process WORLD_SIZE is NOT yet set when this code runs, so falling
    # back to 1 would make the LR scheduler's total_steps 4× too large on rank 0
    # (cosine cycle never reaches min_lr because rank 0 dominates parameter sync
    # under DDP). Detect actual device count from CUDA_VISIBLE_DEVICES instead.
    ddp_rank = int(os.environ.get("LOCAL_RANK", 0))
    env_world_size = int(os.environ.get("WORLD_SIZE", 0))
    if env_world_size > 0:
        ddp_world_size = env_world_size  # subprocess: env var is authoritative
    elif torch.cuda.is_available():
        ddp_world_size = max(torch.cuda.device_count(), 1)  # parent: count visible CUDA
    else:
        ddp_world_size = 1  # CPU

    logger.info(f"Accelerator: {accelerator}, Devices: {devices}")

    # Load dataset
    logger.info(f"Loading dataset from config: {config_path}")
    full_dataset = ParticleDataset(
        config_path=config_path,
        normalize=normalize,
        lazy_loading=lazy_loading,
        data_dir_override=data_dir_override,
    )

    if test_mode:
        logger.info(f"TEST MODE: Using {test_mode_samples:,} events")
        indices = list(range(min(test_mode_samples, len(full_dataset))))
        full_dataset = Subset(full_dataset, indices)

    # Train/val split
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    # random_split stores each split's indices as a PYTHON LIST. With a 783M-event
    # union split, every batch access in a forked DataLoader worker touches those
    # int objects → CPython refcount writes privatize the shared pages (copy-on-
    # write) → host RAM leaks ~GB/min across the 64 workers until the OOM-killer
    # fires `-9` (~2.5h). Storing the indices as a numpy array (ONE object, a
    # contiguous int64 buffer with no per-element refcounts) means worker reads
    # don't privatize pages. Content-neutral: identical split, indices, and order.
    train_dataset.indices = np.asarray(train_dataset.indices, dtype=np.int64)
    val_dataset.indices = np.asarray(val_dataset.indices, dtype=np.int64)

    logger.info(
        f"Train: {len(train_dataset):,} events, Val: {len(val_dataset):,} events"
    )

    # Collator + Dataloaders
    use_gpu = accelerator in ("gpu", "auto")
    use_flat = args.flat or model_cfg.get("use_flat_forward", False)

    if use_flat:
        # Flat compiled pipeline: token packing + static shapes
        T_fixed = model_cfg.get("T_fixed", 8192)
        min_particles = model_cfg.get("min_particles", 3)
        B_fixed = T_fixed // min_particles + 1

        geo_collator = ParticleVectorCollatorGeometric()

        # Compute particle counts for token packing sampler
        logger.info("Computing particle counts for token packing...")
        # Get underlying ParticleDataset (may be wrapped in Subset for test mode)
        base_dataset = (
            full_dataset.dataset if hasattr(full_dataset, "dataset") else full_dataset
        )
        all_counts = TokenPackingSampler.compute_particle_counts(base_dataset)
        # If subseted, only use the relevant indices
        if hasattr(full_dataset, "indices"):
            particle_counts = all_counts[full_dataset.indices]
        else:
            particle_counts = all_counts
        # Split counts to match train/val split
        # random_split returns Subsets with .indices relative to full_dataset
        train_indices = (
            list(train_dataset.indices)
            if hasattr(train_dataset, "indices")
            else list(range(len(train_dataset)))
        )
        val_indices = (
            list(val_dataset.indices)
            if hasattr(val_dataset, "indices")
            else list(range(len(val_dataset)))
        )
        train_counts = particle_counts[train_indices]
        val_counts = particle_counts[val_indices]

        train_sampler = TokenPackingSampler(
            train_counts,
            T_fixed=T_fixed,
            shuffle=True,
            seed=seed,
            drop_last=True,
            rank=ddp_rank,
            world_size=ddp_world_size,
        )
        val_sampler = TokenPackingSampler(
            val_counts,
            T_fixed=T_fixed,
            shuffle=False,
            seed=seed,
            drop_last=False,
            rank=ddp_rank,
            world_size=ddp_world_size,
        )

        def flat_collate_fn(events):
            batch = geo_collator(events)
            return pad_batch_to_static(batch, T_fixed, B_fixed)

        common_kwargs = dict(
            collate_fn=flat_collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory and use_gpu,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor
            if lazy_loading and num_workers > 0
            else None,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            **common_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            **common_kwargs,
        )

        collator = geo_collator  # for generation callback
        logger.info(
            f"Flat compiled pipeline: T_fixed={T_fixed}, B_fixed={B_fixed}, "
            f"~{T_fixed / train_counts.mean():.0f} events/batch"
        )

        # Ensure model config has flat flags
        model_cfg["use_flat_forward"] = True
        model_cfg["geometric_hybrid"] = False

    elif args.geometric:
        collator = ParticleVectorCollatorGeometric()
        logger.info("Using geometric (sparse) collator — no padding, varlen attention")
        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin_memory and use_gpu,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor
            if lazy_loading and num_workers > 0
            else None,
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    else:
        collator = ParticleVectorCollator(max_particles=max_particles)
        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=pin_memory and use_gpu,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor
            if lazy_loading and num_workers > 0
            else None,
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    logger.info(
        f"Train batches: {len(train_loader):,}, Val batches: {len(val_loader):,}"
    )

    steps_per_epoch = len(train_loader) // ddp_world_size
    if limit_train_batches is not None:
        # Virtual epoch: Lightning caps each rank's loader to this many batches
        # (= optimizer steps per epoch), so the scheduler horizon must match.
        steps_per_epoch = min(steps_per_epoch, int(limit_train_batches))
    total_steps = epochs * steps_per_epoch
    logger.info(
        f"Total training steps: {total_steps:,} "
        f"({steps_per_epoch:,} steps/epoch x {epochs} epochs"
        + (
            f"; capped by limit_train_batches={limit_train_batches}"
            if limit_train_batches is not None
            else ""
        )
        + ")"
    )

    # Load manifold stats from dataloader config (chi, logm, e_ref normalization)
    import yaml

    with open(config_path, "r") as f:
        dl_config = yaml.safe_load(f)
    manifold_stats = dl_config.get("manifold_stats")
    if manifold_stats:
        logger.info("Loaded manifold_stats from dataloader config")

    # Count params
    model_config_dict = {
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_ratio": mlp_ratio,
        "dropout": dropout,
        "num_extra_features": num_extra_features,
        "met_embed_dim": met_embed_dim,
        "num_met_tokens": num_met_tokens,
        "manifold_stats": manifold_stats,
    }
    if model_cfg.get("backbone") == "lgatr_slim":
        # TF32 rounds fp32 matmul inputs to 10 mantissa bits, which cancels the fp32
        # vector stream of the slim layers.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        from shellflow.models.riemannian_DiTs_model_slimLGATr import (
            create_riemannian_dits_slim,
        )

        temp_model = create_riemannian_dits_slim({**model_cfg, **model_config_dict})
    else:
        temp_model = create_riemannian_dits(model_config_dict)
    n_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    del temp_model

    logger.info(
        f"Model: {num_layers}L x {num_heads}H x {d_model}D, {n_params:,} params"
    )
    logger.info(f"RFM: sigma_prior={sigma_prior}")

    # Add RFM params to model config dict (read by Lightning module)
    model_config_dict["sigma_prior"] = sigma_prior
    if ema_decay is not None:
        model_config_dict["ema_decay"] = ema_decay

    # Forward all remaining model YAML keys to the Lightning module.
    # Keys already in model_config_dict (from CLI overrides) are not overwritten.
    for key, value in model_cfg.items():
        if key not in model_config_dict:
            model_config_dict[key] = value

    # Create Lightning module
    optimizer_config = {
        "name": "adamw",
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": list(betas),
        "eps": eps,
    }
    scheduler_config = {
        "name": scheduler_name,
        "warmup_pct": warmup_pct,
        "warmup_steps": warmup_steps,
        "min_lr": min_lr,
    }
    training_config = {
        "epochs": epochs,
        "grad_clip": grad_clip,
        "seed": seed,
        "compile_model": compile_model,
        "compile_mode": compile_mode,
        "test_mode": test_mode,
        "test_mode_samples": test_mode_samples,
    }

    module = RiemannianDiTsModule(
        model_config=model_config_dict,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        training_config=training_config,
        total_steps=total_steps,
        quality_weighting_config=quality_weighting_cfg,
    )

    # Load pretrained weights for finetuning
    if finetune_ckpt_path:
        logger.info(f"Loading pretrained weights from: {finetune_ckpt_path}")
        ckpt = torch.load(finetune_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        # Strip torch.compile prefix if present
        if any("._orig_mod." in k for k in state_dict):
            state_dict = {
                k.replace("._orig_mod.", "."): v for k, v in state_dict.items()
            }
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"  Missing keys: {missing}")
        if unexpected:
            logger.warning(f"  Unexpected keys: {unexpected}")
        logger.info("  Weights loaded, fresh optimizer/scheduler, epoch 0")

    # W&B logger
    wandb_logger = None
    if use_wandb:
        run_name = wandb_name or f"rfm_d{d_model}_l{num_layers}"
        if test_mode:
            run_name += "_test"

        full_config = {
            "model": model_config_dict,
            "rfm": {"sigma_prior": sigma_prior},
            "optimizer": {"lr": lr, "weight_decay": weight_decay, "betas": betas},
            "scheduler": {"name": scheduler_name, "warmup_pct": warmup_pct},
            "training": {"epochs": epochs, "grad_clip": grad_clip, "seed": seed},
            "data": {
                "batch_size": batch_size,
                "train_split": train_split,
                "config_path": config_path,
            },
            "n_params": n_params,
            "total_steps": total_steps,
            "train_events": len(train_dataset),
            "val_events": len(val_dataset),
        }

        # W&B resume
        wandb_resume = None
        if resume_ckpt_path and wandb_id:
            wandb_resume = "must"
            logger.info(f"Resuming W&B run: {wandb_id}")

        wandb_logger = WandbLogger(
            project=wandb_project,
            name=run_name,
            id=wandb_id,
            resume=wandb_resume,
            tags=wandb_tags,
            save_code=wandb_cfg.get("save_code", True),
            config=full_config,
        )
        logger.info(f"W&B: project={wandb_project}, run={run_name}")

    # Callbacks
    callbacks = get_default_callbacks(
        config=full_config if use_wandb else {},
        monitor="val/loss/total",
        mode="min",
        checkpoint_dir=checkpoint_dir,
        save_top_k=save_top_k,
        save_every_n_epochs=save_every_n_epochs,
        enable_early_stopping=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
    )

    # Generation callback
    if generation_enabled:
        from shellflow.training.riemannian_DiTs_callback import (
            RiemannianDiTsSamplePlotCallback,
        )

        # Generation uses padded collator (ODE sampling loops over model.forward())
        gen_collator = (
            ParticleVectorCollator(max_particles=max_particles)
            if (args.geometric or use_flat)
            else collator
        )
        # Higgs end-of-training plot config (opt-in via generation.higgs.*).
        higgs_cfg = gen_cfg.get("higgs", {})
        # DM gen-KDE (produces p̂_gen for the training weight w*) — a SEPARATE
        # generation from the monitoring plots, configured under its own
        # TOP-LEVEL `gen_kde:` block (distinct purpose: training-loss reweighting
        # vs result-checking). Generated from the TRAIN split (no train/val leak),
        # step-based + EMA-accumulated. refresh_steps=None → legacy epoch-based
        # refresh. num_samples/n_steps fall back to the monitoring values when omitted.
        gen_kde_cfg = cfg.get("gen_kde", {}) or {}
        gen_kde_refresh_steps = gen_kde_cfg.get("refresh_steps", None)
        gen_kde_ema = float(gen_kde_cfg.get("ema", 0.3))
        gen_kde_num_samples = gen_kde_cfg.get("num_samples", None)
        gen_kde_n_steps = gen_kde_cfg.get("n_steps", None)
        # Dedicated 2to4lep reference for the OSSF invariant-mass panels ONLY
        # (generation.monitor). On a union run, set it so those panels draw 50k
        # from already-multilepton events (full OSSF stats). Omit it when the
        # training data is already 2to4lep — the cut is then skipped.
        monitor_cfg = gen_cfg.get("monitor", {}) or {}
        monitor_dataset = None
        if monitor_cfg.get("data_dir") and monitor_cfg.get("dataloader_config"):
            monitor_dataset = ParticleDataset(
                config_path=monitor_cfg["dataloader_config"],
                normalize=True,
                lazy_loading=True,
                data_dir_override=monitor_cfg["data_dir"],
            )
            logger.info(
                f"OSSF inv-mass panels: dedicated 2to4lep reference "
                f"({monitor_cfg['data_dir']}, {len(monitor_dataset):,} events)"
            )
        callbacks.append(
            RiemannianDiTsSamplePlotCallback(
                val_dataset=val_dataset,
                collator=gen_collator,
                num_samples=gen_num_samples,
                batch_size=gen_batch_size,
                n_gen_steps=gen_n_steps,
                eta=gen_eta,
                seed=seed,
                every_n_epochs=gen_every_n_epochs,
                gen_kde_dataset=train_dataset,
                gen_kde_refresh_steps=(
                    int(gen_kde_refresh_steps)
                    if gen_kde_refresh_steps is not None
                    else None
                ),
                gen_kde_ema=gen_kde_ema,
                gen_kde_num_samples=(
                    int(gen_kde_num_samples)
                    if gen_kde_num_samples is not None
                    else None
                ),
                gen_kde_n_steps=(
                    int(gen_kde_n_steps) if gen_kde_n_steps is not None else None
                ),
                monitor_dataset=monitor_dataset,
                higgs_enabled=bool(higgs_cfg.get("enabled", False)),
                higgs_dataloader_config=higgs_cfg.get(
                    "dataloader_config",
                    "src/shellflow/config/dataloader_config_06_2to4lep_reduced.yaml",
                ),
                higgs_data_dir=higgs_cfg.get(
                    "data_dir",
                    "data/run2_data_4lep_reduced",
                ),
                higgs_raw_data_path=higgs_cfg.get("raw_data_path"),
                higgs_batch_size=int(higgs_cfg.get("batch_size", 16384)),
                higgs_shard_dir=higgs_cfg.get("shard_dir", "/tmp/higgs_shards"),
                higgs_every_n_epochs=(
                    int(higgs_cfg["every_n_epochs"])
                    if "every_n_epochs" in higgs_cfg
                    and higgs_cfg["every_n_epochs"] is not None
                    else None
                ),
            )
        )
        logger.info(
            f"Generation: {gen_num_samples:,} samples every {gen_every_n_epochs} epoch(s)"
        )
        if gen_kde_refresh_steps is not None:
            logger.info(
                f"DM gen-KDE: step-based refresh every {int(gen_kde_refresh_steps):,} "
                f"steps from TRAIN reference (EMA λ={gen_kde_ema}), {gen_num_samples:,} samples"
            )
        else:
            logger.info("DM gen-KDE: epoch-based refresh (legacy)")
        if bool(higgs_cfg.get("enabled", False)):
            logger.info(
                f"Higgs end-of-training plot ENABLED "
                f"(data_dir={higgs_cfg.get('data_dir', 'default')}, "
                f"batch_size={higgs_cfg.get('batch_size', 16384)})"
            )
    else:
        logger.info("Generation monitoring: disabled")

    # DDP epoch seeding: update sampler shuffle seed each epoch
    if use_flat and ddp_world_size > 1:

        class _SamplerEpochCallback(L.Callback):
            def on_train_epoch_start(self, trainer, pl_module):
                train_sampler.set_epoch(trainer.current_epoch)

        callbacks.append(_SamplerEpochCallback())

    # Validation interval
    val_check_interval = None
    if val_checks_per_epoch > 1:
        val_check_interval = 1.0 / val_checks_per_epoch

    # DDP strategy with extended NCCL collective timeout. Default is 30 min;
    # bump to 2 h so periodic-mode generation hooks (especially the Higgs
    # callback's rank-0 plot + W&B upload, which can take 10-30 min on a
    # full 4-lep dataset) don't trip the watchdog when other ranks reach
    # the next collective (sync_dist=True metric or backward-pass allreduce)
    # and wait for rank 0 to participate.
    # static_graph: the model uses the same params every step (no
    # find_unused_parameters; aux head exercised every step via the phantom
    # term), so DDP can skip its per-step graph traversal and fuse the gradient
    # all-reduce more aggressively. gradient_as_bucket_view avoids a grad copy.
    # Both are pure-efficiency (training math unchanged).
    _ddp_kwargs = dict(
        timeout=timedelta(hours=2),
        static_graph=True,
        gradient_as_bucket_view=True,
    )
    strategy = "auto"
    if isinstance(devices, list) and len(devices) > 1:
        strategy = DDPStrategy(**_ddp_kwargs)
    elif isinstance(devices, int) and devices > 1:
        strategy = DDPStrategy(**_ddp_kwargs)

    # Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=grad_clip,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        use_distributed_sampler=not use_flat,
    )

    # Log resume info
    if resume_ckpt_path:
        logger.info(f"Resuming from: {resume_ckpt_path}")
    elif finetune_ckpt_path:
        logger.info(f"Finetuning from: {finetune_ckpt_path}")

    # Train
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt_path,
    )

    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info(
        f"Best val loss: {trainer.callback_metrics.get('val/loss/total', 'N/A')}"
    )
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info("=" * 70)

    if wandb_logger:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
