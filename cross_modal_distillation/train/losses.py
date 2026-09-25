"""
Loss functions for teacher MAE pretraining and LFP student distillation.

Poisson MAE loss
----------------
Used during SpikeMAEModel.forward() (implemented inline there).
Exposed here as a standalone function for unit testing or alternative usage.

Distillation loss
-----------------
Eq. 1 of the paper:
    L = (1/ny·T) · MSE(y, ŷ)
      + lambda_align · (1 − (1/T) · Σ_t cosine(z_lfp_t, z_spike_t))

Fully-supervised distillation loss
-----------------------------------
Eq. 3 of the paper (optional extension, replaces autoencoding term with
a behavior regression term):
    L_sup = (1/ny·T) · MSE(z, f_psi(z_lfp))
           + lambda_align · (1 − (1/T) · Σ_t cosine(z_lfp_t, z_spike_t))
"""

import torch
import torch.nn.functional as F


def poisson_mae_loss(
    lambda_: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Poisson NLL for masked spike patches.

    Args:
        lambda_    : (N_valid,) predicted Poisson rates (softplus output, >0)
        targets    : (N_valid,) true spike counts (float)
        valid_mask : (N_valid,) bool, True = this element is NOT spatial padding

    Returns:
        scalar mean Poisson NLL over valid elements
    """
    lam = lambda_[valid_mask]
    tgt = targets[valid_mask].float()
    if lam.numel() == 0:
        return torch.tensor(0.0, device=lambda_.device, requires_grad=True)
    return F.poisson_nll_loss(lam, tgt, log_input=False, full=False, reduction="mean")


def distillation_loss(
    z_lfp: torch.Tensor,
    z_spike: torch.Tensor,
    lfp_hat: torch.Tensor,
    lfp: torch.Tensor,
    lambda_align: float = 5.0,
) -> tuple:
    """
    Eq. 1 distillation loss.

    Args:
        z_lfp       : (B, T, D) student LFP representations (pooled)
        z_spike     : (B, T, D) teacher spike representations (pooled, no grad)
        lfp_hat     : (B, T, N_lfp) student reconstructed LFP
        lfp         : (B, T, N_lfp) ground-truth z-scored LFP
        lambda_align: weight for alignment term (default 5, paper A.2)

    Returns:
        (loss, loss_ae, loss_align) — all scalars
    """
    loss_ae = F.mse_loss(lfp_hat, lfp, reduction="mean")
    cos_sim = F.cosine_similarity(z_lfp, z_spike, dim=-1)  # (B, T)
    loss_align = 1.0 - cos_sim.mean()
    loss = loss_ae + lambda_align * loss_align
    return loss, loss_ae, loss_align


def supervised_distillation_loss(
    z_lfp: torch.Tensor,
    z_spike: torch.Tensor,
    behavior_hat: torch.Tensor,
    behavior: torch.Tensor,
    lambda_align: float = 5.0,
) -> tuple:
    """
    Eq. 3 fully-supervised distillation loss (Appendix A.11).
    Replaces the autoencoding term with a behavior regression term.

    Args:
        z_lfp        : (B, T, D) student LFP representations
        z_spike      : (B, T, D) teacher spike representations
        behavior_hat : (B, T, n_beh) predicted behavior
        behavior     : (B, T, n_beh) ground-truth behavior
        lambda_align : weight for alignment term (default 5)

    Returns:
        (loss, loss_beh, loss_align) — all scalars
    """
    loss_beh = F.mse_loss(behavior_hat, behavior, reduction="mean")
    cos_sim = F.cosine_similarity(z_lfp, z_spike, dim=-1)
    loss_align = 1.0 - cos_sim.mean()
    loss = loss_beh + lambda_align * loss_align
    return loss, loss_beh, loss_align
