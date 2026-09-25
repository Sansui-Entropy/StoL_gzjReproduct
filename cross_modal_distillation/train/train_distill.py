"""
train_distill.py — LFP student cross-modal knowledge distillation.

Paper reference: Section 3.3, Eq. 1, Appendix A.1–A.2.

Usage (Hydra)
-------------
    python -m cross_modal_distillation.train.train_distill \
        teacher_ckpt=<path/to/finetuned_teacher.ckpt> \
        dataset.session=MonkeyI_20160622_01

Configuration
-------------
All hyperparameters loaded from:
    cross_modal_distillation/configs/train/distill.yaml
Override on the command line with Hydra syntax.

Training steps (paper Section 3.3):
  1. Load a fine-tuned SpikeMAEModel teacher — fully frozen.
  2. Build a fresh LFPStudentModel for the target session.
  3. Optimise Eq. 1:
         L = MSE(y, ŷ) + 5 · (1 − cosine(z_lfp, z_spike))
     where z_spike comes from the frozen teacher.encode().

Dataset
-------
Replace the "# DATASET" block below with your concrete PairedSpikeLFPDataset
subclass once data is available.
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cross_modal_distillation.data.collate import collate_paired_fn
from cross_modal_distillation.data.makin_sessions import DISTILL_SESSION, require_makin_session
from cross_modal_distillation.data.paired_dataset import MakinPairedDataset, SessionBatchSampler
from cross_modal_distillation.models.spike_mae import SpikeMAEModel
from cross_modal_distillation.models.distillation import DistillationModel, LFPStudentModel
from cross_modal_distillation.train.optim import build_optimizer_and_scheduler
from cross_modal_distillation.utility.utils import init_logger

log = init_logger("TrainDistill")


def run_epoch(
    model: DistillationModel,
    loader: DataLoader,
    optimizer,
    device: str,
    is_train: bool = True,
) -> tuple:
    """Run one distillation epoch. Returns (mean_total, mean_ae, mean_align)."""
    model.student.train(is_train)
    model.teacher.eval()   # teacher always eval

    total, ae_total, align_total = 0.0, 0.0, 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            # ------------------------------------------------------------------
            # Unpack PairedBatchItem (from collate_paired_fn)
            # ------------------------------------------------------------------
            if not (hasattr(batch, "spikes") and hasattr(batch, "lfp")):
                raise TypeError(
                    "Batch must have 'spikes' and 'lfp' attributes. "
                    "Use collate_paired_fn."
                )

            spikes = batch.spikes.to(device)
            lfp = batch.lfp.to(device)
            subject_sessions = batch.subject_sessions
            position_ids = getattr(batch, "position_ids", None)
            if torch.is_tensor(position_ids):
                position_ids = position_ids.to(device)

            if is_train:
                optimizer.zero_grad()

            loss, loss_ae, loss_align = model(
                spikes=spikes,
                lfp=lfp,
                subject_sessions_spike=subject_sessions,
                subject_sessions_lfp=subject_sessions,
                position_ids_spike=position_ids,
                position_ids_lfp=position_ids,
            )

            if is_train:
                loss.backward()
                optimizer.step()

            total += loss.item()
            ae_total += loss_ae.item()
            align_total += loss_align.item()
            n_batches += 1

    denom = max(n_batches, 1)
    return total / denom, ae_total / denom, align_total / denom


@hydra.main(
    config_path="../configs/train",
    config_name="distill",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dataset_cfg = cfg.dataset
    target_session = require_makin_session(dataset_cfg.get("session"), [DISTILL_SESSION])
    output_dir = Path(cfg.get("output_dir", "./results/checkpoints/distill")) / target_session
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MakinPairedDataset(
        lfp_metadata_path=dataset_cfg.lfp_metadata_path,
        spike_segment_dir=dataset_cfg.spike_segment_dir,
        session_info_path=dataset_cfg.session_info_path,
        split=dataset_cfg.get("split", "train"),
        sessions=[target_session],
    )
    session_d_lfp_dict = dataset.get_session_d_lfp_dict()
    n_lfp = session_d_lfp_dict[target_session]
    log.info(
        f"Distill dataset: {len(dataset)} segments from {target_session}, n_lfp={n_lfp}"
    )

    loader = DataLoader(
        dataset,
        batch_sampler=SessionBatchSampler(
            dataset.subject_sessions,
            batch_size=cfg.training.batch_size,
            shuffle=True,
        ),
        num_workers=cfg.training.get("num_workers", 4),
        collate_fn=collate_paired_fn,
    )

    # ---------------------------------------------------------------------- #
    # Teacher — load pretrained / fine-tuned checkpoint, fully frozen        #
    # ---------------------------------------------------------------------- #
    teacher_ckpt = cfg.get("teacher_ckpt")
    assert teacher_ckpt, "teacher_ckpt must be set (path to fine-tuned SpikeMAEModel)."
    teacher = SpikeMAEModel.load_checkpoint(teacher_ckpt).to(device)
    if target_session not in teacher.tokenizer.session_d_input_dict:
        raise ValueError(
            f"Teacher checkpoint has no embedding for '{target_session}'. "
            "Fine-tune the teacher on this session before distillation."
        )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    log.info(f"Teacher loaded from '{teacher_ckpt}' and frozen.")

    # ---------------------------------------------------------------------- #
    # Student — fresh LFP encoder                                             #
    # ---------------------------------------------------------------------- #
    model_cfg = cfg.model
    student = LFPStudentModel(
        session_d_input_dict=session_d_lfp_dict,
        n_lfp=n_lfp,
        spatial_patch_size=model_cfg.spatial_patch_size,
        d_hidden=model_cfg.d_hidden,
        num_encoder_layers=model_cfg.num_encoder_layers,
        num_heads=model_cfg.num_heads,
        kernel_size=model_cfg.get("kernel_size", 3),
        dilation=model_cfg.get("dilation", 1),
        dropout=model_cfg.dropout,
        use_flash_attention=model_cfg.get("use_flash_attention", False),
        max_pos=model_cfg.get("max_pos", 1024),
    ).to(device)

    distill_model = DistillationModel(
        teacher=teacher,
        student=student,
        lambda_align=cfg.training.get("lambda_align", 5.0),
    )

    optimizer, scheduler = build_optimizer_and_scheduler(
        distill_model.student_parameters(),
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
    max_epochs = cfg.training.get("max_epochs", 400)

    for epoch in range(1, max_epochs + 1):
        loss, loss_ae, loss_align = run_epoch(
            distill_model, loader, optimizer, device, is_train=True
        )
        scheduler.step()

        log.info(
            f"Epoch {epoch}/{max_epochs}  "
            f"loss={loss:.5f}  ae={loss_ae:.5f}  align={loss_align:.5f}"
        )

        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            ckpt_path = str(output_dir / "best_distill.ckpt")
            config_dict = {
                "session_d_input_dict": session_d_lfp_dict,
                "n_lfp"               : n_lfp,
                "spatial_patch_size"  : model_cfg.spatial_patch_size,
                "d_hidden"            : model_cfg.d_hidden,
                "num_encoder_layers"  : model_cfg.num_encoder_layers,
                "num_heads"           : model_cfg.num_heads,
                "kernel_size"         : model_cfg.get("kernel_size", 3),
                "dilation"            : model_cfg.get("dilation", 1),
                "dropout"             : model_cfg.dropout,
                "lambda_align"        : cfg.training.get("lambda_align", 5.0),
            }
            distill_model.save_checkpoint(ckpt_path, config_dict)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    log.info(f"Distillation complete. Best loss={best_loss:.5f}")


if __name__ == "__main__":
    main()
