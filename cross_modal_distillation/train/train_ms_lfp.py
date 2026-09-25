"""
Train the undistilled multi-session LFP model (MS-LFP baseline).

Loss is LFP reconstruction MSE only. The 18 evaluation sessions are included,
matching the official comparison against the distilled student.

    python -m cross_modal_distillation.train.train_ms_lfp
"""

from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cross_modal_distillation.data.makin_sessions import MAKIN_EVAL_SESSIONS
from cross_modal_distillation.data.paired_dataset import MakinLFPDataset, collate_lfp_fn
from cross_modal_distillation.models.distillation import LFPStudentModel
from cross_modal_distillation.train.optim import build_optimizer_and_scheduler
from cross_modal_distillation.utility.utils import init_logger

log = init_logger("TrainMSLFP")


def run_epoch(model, loader, optimizer, device, is_train):
    model.train(is_train)
    total = 0.0
    n_batches = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for lfp, subject_sessions in loader:
            lfp = lfp.to(device)
            if is_train:
                optimizer.zero_grad()
            _, lfp_hat = model(lfp, subject_sessions)
            loss = F.mse_loss(lfp_hat, lfp)
            if is_train:
                loss.backward()
                optimizer.step()
            total += loss.item()
            n_batches += 1
    return total / max(n_batches, 1)


@hydra.main(config_path="../configs/train", config_name="ms_lfp", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.get("output_dir", "./results/checkpoints/ms_lfp"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MakinLFPDataset(
        lfp_metadata_path=cfg.dataset.lfp_metadata_path,
        split=cfg.dataset.get("split", "train"),
        sessions=list(MAKIN_EVAL_SESSIONS),
    )
    session_d_lfp_dict = dataset.session_d_lfp_dict
    n_values = set(session_d_lfp_dict.values())
    if len(n_values) != 1:
        raise ValueError(f"MS-LFP expects one channel count, found {session_d_lfp_dict}")
    n_lfp = int(next(iter(n_values)))
    log.info(f"MS-LFP dataset: {len(dataset)} train segments, {len(session_d_lfp_dict)} sessions")

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.get("num_workers", 4),
        collate_fn=collate_lfp_fn,
    )

    model_cfg = cfg.model
    model = LFPStudentModel(
        session_d_input_dict=session_d_lfp_dict,
        n_lfp=n_lfp,
        spatial_patch_size=model_cfg.spatial_patch_size,
        d_hidden=model_cfg.d_hidden,
        num_encoder_layers=model_cfg.num_encoder_layers,
        num_heads=model_cfg.num_heads,
        kernel_size=model_cfg.kernel_size,
        dilation=model_cfg.dilation,
        dropout=model_cfg.dropout,
        use_flash_attention=model_cfg.get("use_flash_attention", False),
        max_pos=model_cfg.get("max_pos", 1024),
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model.parameters(),
        max_lr=cfg.training.max_lr,
        start_factor=cfg.training.start_factor,
        warmup_epochs=cfg.training.warmup_epochs,
        decay_factor=cfg.training.decay_factor,
        weight_decay=cfg.training.weight_decay,
    )

    best_loss = float("inf")
    patience_counter = 0
    early_stop_patience = cfg.training.get("early_stop_patience", 50)
    max_epochs = cfg.training.get("max_epochs", 400)
    for epoch in range(1, max_epochs + 1):
        loss = run_epoch(model, loader, optimizer, device, is_train=True)
        scheduler.step()
        log.info(f"Epoch {epoch}/{max_epochs}  loss={loss:.5f}")
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            config_dict = {
                "session_d_input_dict": session_d_lfp_dict,
                "n_lfp": n_lfp,
                "spatial_patch_size": model_cfg.spatial_patch_size,
                "d_hidden": model_cfg.d_hidden,
                "num_encoder_layers": model_cfg.num_encoder_layers,
                "num_heads": model_cfg.num_heads,
                "kernel_size": model_cfg.kernel_size,
                "dilation": model_cfg.dilation,
                "dropout": model_cfg.dropout,
            }
            path = str(output_dir / "best_ms_lfp.ckpt")
            torch.save({"state_dict": model.state_dict(), "config": config_dict}, path)
            log.info(f"Saved {path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                log.info(f"Early stopping at epoch {epoch}")
                break
    log.info(f"MS-LFP complete. Best loss={best_loss:.5f}")


if __name__ == "__main__":
    main()
