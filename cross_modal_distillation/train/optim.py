"""
Optimizer and learning-rate schedule factory (paper Appendix A.2).

Schedule:
  1. Linear warmup: LR rises from (start_factor × max_lr) to max_lr over
     warmup_epochs epochs.
  2. Exponential decay: LR multiplied by decay_factor each epoch thereafter.

Paper values (A.2):
  - max_lr       = 0.000625
  - start_factor = 0.3   →  initial LR = 0.3 × 0.000625 ≈ 1.875e-4
  - warmup_epochs= 30
  - decay_factor = 0.995
  - optimizer    = AdamW
  - weight_decay starts at 0.1, ramps to 0.4 over 1000 epochs
    (in practice no run reached 1000 epochs, so weight_decay≈0.1 always)

Usage
-----
    optimizer, scheduler = build_optimizer_and_scheduler(model.parameters())
    for epoch in range(max_epochs):
        train_one_epoch(...)
        scheduler.step()
"""

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, SequentialLR


def build_optimizer_and_scheduler(
    params,
    max_lr: float = 6.25e-4,
    start_factor: float = 0.3,
    warmup_epochs: int = 30,
    decay_factor: float = 0.995,
    weight_decay: float = 0.1,
):
    """
    Build AdamW optimizer with linear-warmup + exponential-decay schedule.

    Args:
        params        : iterable of parameters (e.g. model.parameters())
        max_lr        : peak learning rate after warmup (paper: 0.000625)
        start_factor  : fraction of max_lr used at epoch 0 (paper: 0.3)
        warmup_epochs : epochs to reach max_lr (paper: 30)
        decay_factor  : per-epoch multiplicative factor after warmup (paper: 0.995)
        weight_decay  : AdamW weight decay (paper: starts at 0.1)

    Returns:
        optimizer  : AdamW
        scheduler  : SequentialLR (warmup then exponential decay)
    """
    optimizer = AdamW(params, lr=max_lr, weight_decay=weight_decay)

    # --- Warmup phase (linear ramp from start_factor to 1.0) ---
    def warmup_lambda(epoch):
        if epoch >= warmup_epochs:
            return 1.0
        # linear interpolation from start_factor to 1.0
        return start_factor + (1.0 - start_factor) * epoch / max(1, warmup_epochs - 1)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # --- Exponential decay phase (applied each epoch after warmup) ---
    def exp_lambda(epoch):
        return decay_factor ** epoch

    exp_scheduler = LambdaLR(optimizer, lr_lambda=exp_lambda)

    # SequentialLR: first run warmup for warmup_epochs steps, then exp decay
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, exp_scheduler],
        milestones=[warmup_epochs],
    )

    return optimizer, scheduler
