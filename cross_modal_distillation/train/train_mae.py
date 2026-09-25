"""
train_mae.py — Teacher spike MAE pretraining / single-session fine-tuning.

Paper reference: Section 3.1–3.2, Appendix A.1–A.2.

Usage (Hydra)
-------------
# Pretrain on multiple sessions:
    python -m cross_modal_distillation.train.train_mae mode=pretrain

# Fine-tune on a single held-out session:
    python -m cross_modal_distillation.train.train_mae mode=finetune \
        teacher_ckpt=<path/to/pretrained.ckpt> \
        dataset.new_session=MonkeyI_20160622_01

Both modes use the same MAE objective (Poisson NLL on masked spike patches).
The difference is:
  - pretrain : all session space embeddings are trained from scratch on pooled
               multi-session data.
  - finetune : model weights are loaded from a pretrained checkpoint; a new
               space embedding is registered for the target session via
               tokenizer.update_for_new_sessions(); all parameters are then
               updated jointly (paper A.1: "all model parameters are updated
               during fine-tuning").

Configuration
-------------
All hyperparameters are loaded from:
    cross_modal_distillation/configs/train/teacher_mae.yaml
Override any value on the command line via Hydra syntax, e.g.:
    python -m cross_modal_distillation.train.train_mae model.mask_ratio=0.75

Dataset
-------
The training loop expects a DataLoader that yields PairedBatchItem (from
collate_paired_fn) or a plain BatchItem where batch.spikes is the spike tensor.
Replace the dataset construction block marked "# DATASET" with your concrete
PairedSpikeLFPDataset subclass once data is available.
"""

import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cross_modal_distillation.data.collate import collate_paired_fn
from cross_modal_distillation.data.makin_sessions import (
    DISTILL_SESSION,
    TEACHER_PRETRAIN_SESSIONS,
    require_makin_session,
)
from cross_modal_distillation.data.paired_dataset import (
    MakinPairedDataset,
    MakinSpikeDataset,
    SessionBatchSampler,
)
from cross_modal_distillation.models.spike_mae import SpikeMAEModel
from cross_modal_distillation.train.optim import build_optimizer_and_scheduler
from cross_modal_distillation.utility.utils import init_logger

log = init_logger("TrainMAE")


def run_epoch(
    model: SpikeMAEModel,
    loader: DataLoader,
    optimizer,
    device: str,
    is_train: bool = True,
) -> float:
    """Run one epoch of MAE training or evaluation. Returns mean loss."""
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            # ------------------------------------------------------------------
            # Unpack batch — works with PairedBatchItem (spikes field) or any
            # dict/namedtuple that has 'spikes' and 'subject_sessions'.
            # ------------------------------------------------------------------
            if hasattr(batch, "spikes"):
                spikes = batch.spikes
                subject_sessions = batch.subject_sessions
                position_ids = getattr(batch, "position_ids", None)
            else:
                raise TypeError(
                    "Batch must have 'spikes' and 'subject_sessions' attributes. "
                    "Use collate_paired_fn or a compatible collate function."
                )

            spikes = spikes.to(device)
            if position_ids is not None and torch.is_tensor(position_ids):
                position_ids = position_ids.to(device)

            if is_train:
                optimizer.zero_grad()

            loss, _ = model(
                spikes=spikes,
                subject_sessions=subject_sessions,
                position_ids=position_ids,
            )

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


@hydra.main(
    config_path="../configs/train",
    config_name="teacher_mae",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    mode = cfg.get("mode", "pretrain")   # "pretrain" | "finetune"
    dataset_cfg = cfg.dataset
    new_session = require_makin_session(dataset_cfg.get("new_session"), [DISTILL_SESSION])
    output_dir = Path(cfg.get("output_dir", "./results/checkpoints/teacher")) / new_session
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Finetune / held-out session: {new_session}")

    if mode == "finetune":
        train_dataset = MakinPairedDataset(
            lfp_metadata_path=dataset_cfg.lfp_metadata_path,
            spike_segment_dir=dataset_cfg.spike_segment_dir,
            session_info_path=dataset_cfg.session_info_path,
            split=dataset_cfg.get("split", "train"),
            sessions=[new_session],
        )
    elif mode == "pretrain":
        train_dataset = MakinSpikeDataset(
            spike_metadata_path=dataset_cfg.spike_metadata_path,
            session_info_path=dataset_cfg.session_info_path,
            split=dataset_cfg.get("split", "train"),
            sessions=list(TEACHER_PRETRAIN_SESSIONS),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}.  Choose 'pretrain' or 'finetune'.")
    session_d_spike_dict = train_dataset.get_session_d_spike_dict()
    log.info(
        f"MAE dataset: {len(train_dataset)} segments, "
        f"sessions={list(session_d_spike_dict)}"
    )

    batch_size = cfg.training.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=SessionBatchSampler(
            train_dataset.subject_sessions, batch_size=batch_size, shuffle=True
        ),
        num_workers=cfg.training.get("num_workers", 4),
        collate_fn=collate_paired_fn,
    )

    # ---------------------------------------------------------------------- #
    # Model                                                                   #
    # ---------------------------------------------------------------------- #
    model_cfg = cfg.model
    if mode == "pretrain":
        model = SpikeMAEModel(
            session_d_input_dict=session_d_spike_dict,
            spatial_patch_size=model_cfg.spatial_patch_size,
            d_encoder=model_cfg.d_encoder,
            d_predictor=model_cfg.d_predictor,
            num_encoder_layers=model_cfg.num_encoder_layers,
            num_predictor_layers=model_cfg.num_predictor_layers,
            num_heads=model_cfg.num_heads,
            max_count=model_cfg.max_count,
            mask_ratio=model_cfg.mask_ratio,
            dropout=model_cfg.dropout,
            use_flash_attention=model_cfg.get("use_flash_attention", False),
            max_pos=model_cfg.get("max_pos", 1024),
        )
    elif mode == "finetune":
        teacher_ckpt = cfg.get("teacher_ckpt")
        assert teacher_ckpt, "mode=finetune requires teacher_ckpt to be set."
        model = SpikeMAEModel.load_checkpoint(teacher_ckpt)
        if new_session in model.tokenizer.session_d_input_dict:
            raise ValueError(
                f"'{new_session}' is already in the teacher checkpoint. "
                "Re-run mode=pretrain with this dataset.new_session so it is held out."
            )
        n_neurons = session_d_spike_dict[new_session]
        model.tokenizer.update_for_new_sessions({new_session: n_neurons})
        log.info(f"Registered new session embedding for '{new_session}' ({n_neurons} units)")
    else:
        raise ValueError(f"Unknown mode: {mode}.  Choose 'pretrain' or 'finetune'.")

    model = model.to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model.parameters(),
        max_lr=cfg.training.max_lr,
        start_factor=cfg.training.start_factor,
        warmup_epochs=cfg.training.warmup_epochs,
        decay_factor=cfg.training.decay_factor,
        weight_decay=cfg.training.weight_decay,
    )

    # ---------------------------------------------------------------------- #
    # Training loop with early stopping                                       #
    # ---------------------------------------------------------------------- #
    best_loss = float("inf")
    patience_counter = 0
    early_stop_patience = cfg.training.get("early_stop_patience", 50)
    max_epochs = cfg.training.get("max_epochs", 600)

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, is_train=True)
        scheduler.step()

        log.info(f"[{mode}] Epoch {epoch}/{max_epochs}  train_loss={train_loss:.5f}")

        # Checkpoint on improvement
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            ckpt_path = str(output_dir / f"best_{mode}.ckpt")
            model_config = {
                "session_d_input_dict": model.tokenizer.session_d_input_dict,
                "spatial_patch_size"  : model.spatial_patch_size,
                "d_encoder"           : model.d_encoder,
                "d_predictor"         : model.d_predictor,
                "num_encoder_layers"  : model_cfg.num_encoder_layers,
                "num_predictor_layers": model_cfg.num_predictor_layers,
                "num_heads"           : model_cfg.num_heads,
                "max_count"           : model.max_count,
                "mask_ratio"          : model.mask_ratio,
                "dropout"             : model_cfg.dropout,
            }
            model.save_checkpoint(ckpt_path, model_config)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                log.info(f"Early stopping at epoch {epoch} (patience={early_stop_patience})")
                break

    log.info(f"Training complete. Best loss={best_loss:.5f}, saved to {output_dir}")


if __name__ == "__main__":
    main()
