"""
DistillationModel — Cross-modal knowledge distillation (paper Section 3.3 / Fig. 2 / Eq. 1)

Architecture:
  ┌─ Frozen teacher (SpikeMAEModel.encode) ──────────────────┐
  │  Spike counts (B,T,N_spike) → tokenizer → encoder → pool │ z_spike (B,T,D)
  └──────────────────────────────────────────────────────────┘
                                           ↓
                                    cosine similarity ──────────┐
                                           ↑                    │
  ┌─ LFP student ─────────────────────────────────────────┐    │
  │  LFP (B,T,N_lfp) → conv-tokenizer → encoder → pool   │ z_lfp (B,T,D)
  │                          │                            │    │
  │              f_phi (Linear, D→N_lfp)                  │    │
  │              reconstruct ŷ ≈ y (MSE)                  │    │
  └───────────────────────────────────────────────────────┘    │
                                                               ↓
  L = (1/ny·T)·MSE(y,ŷ) + λ·(1 - mean_t cosine(z_lfp, z_spike))
  λ = 5  (paper A.2)

Design notes:
- Teacher weights are fully frozen; gradients never flow into SpikeMAEModel.
- LFP tokenizer uses dilated causal conv for value embedding (paper 3.3 / 3.1).
- pool_by_position() is shared with spike_mae.py.
- n_lfp (number of LFP channels) is passed at construction; it may differ
  across sessions — use one DistillationModel per session, or pass the largest
  n_lfp and mask unused channels in the loss (handled externally).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cross_modal_distillation.models.custom_transformer import CustomTransformer
from cross_modal_distillation.models.tokenizer import PatchTokenizer
from cross_modal_distillation.models.spike_mae import SpikeMAEModel, pool_by_position
from cross_modal_distillation.utility.utils import init_logger

std_logger = init_logger("Distillation")


class LFPStudentModel(nn.Module):
    """
    Single-session LFP encoder student.

    Uses a dilated causal Conv1d as the value embedding (paper 3.3:
    "we use a dilated convolutional layer for the value embeddings of LFP
    signals instead of learnable embeddings").

    Parameters
    ----------
    session_d_input_dict : dict[str, int]
        Maps 'subject_session' → number of LFP channels (n_lfp).
    n_lfp : int
        Number of LFP channels for this session (used by reconstruction head).
    spatial_patch_size : int
        S for LFP (default 32, paper A.2).
    d_hidden : int
        Encoder hidden dimension (default 256, same as teacher).
    num_encoder_layers : int
        Encoder depth (default 10).
    num_heads : int
        Attention heads.
    kernel_size : int
        Causal conv kernel size (default 3, following repo convention).
    dilation : int
        Causal conv dilation (default 1).
    dropout : float
    use_flash_attention : bool
    max_pos : int
        Maximum RoPE position index.
    """

    def __init__(
        self,
        session_d_input_dict: Dict[str, int],
        n_lfp: int,
        spatial_patch_size: int = 32,
        d_hidden: int = 256,
        num_encoder_layers: int = 10,
        num_heads: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        max_pos: int = 1024,
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_lfp = n_lfp

        # Tokenizer — causal conv value embedding, session space embedding
        self.tokenizer = PatchTokenizer(
            spatial_patch_size=spatial_patch_size,
            session_d_input_dict=session_d_input_dict,
            d_hidden=d_hidden,
            learn_patch_embedding=True,
            use_conv_for_input=True,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        # Encoder (RoPE, 10 layers, d=256)
        self.encoder = CustomTransformer(
            num_layers=num_encoder_layers,
            d_hidden=d_hidden,
            num_heads=num_heads,
            dropout=dropout,
            attention_module_name="RotarySelfAttention",
            use_flash_attention=use_flash_attention,
            max_pos=max_pos,
            use_final_norm=True,
        )

        # f_phi: linear reconstruction head D → n_lfp  (paper Eq.1)
        self.recon_head = nn.Linear(d_hidden, n_lfp, bias=True)

    def forward(
        self,
        lfp: torch.Tensor,
        subject_sessions: List[str],
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lfp             : (B, T, N_lfp) continuous LFP signal (z-scored)
            subject_sessions: list[str] of length B
            position_ids    : (B, T) optional

        Returns:
            z_lfp   : (B, T, D) pooled LFP representations
            lfp_hat : (B, T, N_lfp) reconstructed LFP (for autoencoding loss)
        """
        _, tokens, pos_ids, _, seq_lens = self.tokenizer(
            x=lfp,
            subject_sessions=subject_sessions,
            position_ids=position_ids,
        )
        _, enc_out = self.encoder(
            x=tokens, position_ids=pos_ids, seq_lens=seq_lens
        )
        z_lfp = pool_by_position(enc_out, pos_ids)   # (B, T, D)
        lfp_hat = self.recon_head(z_lfp)             # (B, T, n_lfp)
        return z_lfp, lfp_hat


class DistillationModel(nn.Module):
    """
    Cross-modal knowledge distillation model (paper Eq. 1).

    Wraps a *frozen* SpikeMAEModel teacher and a trainable LFPStudentModel.

    Parameters
    ----------
    teacher : SpikeMAEModel
        Pretrained and (optionally) fine-tuned spike MAE model.
        Its parameters are frozen inside this module.
    student : LFPStudentModel
        Freshly initialized LFP student to be trained via distillation.
    lambda_align : float
        Weight of the cosine alignment term in Eq. 1 (default 5, paper A.2).
    """

    def __init__(
        self,
        teacher: SpikeMAEModel,
        student: LFPStudentModel,
        lambda_align: float = 5.0,
    ):
        super().__init__()
        self.lambda_align = lambda_align

        # Freeze teacher completely
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.student = student

    def forward(
        self,
        spikes: torch.Tensor,
        lfp: torch.Tensor,
        subject_sessions_spike: List[str],
        subject_sessions_lfp: List[str],
        position_ids_spike: Optional[torch.Tensor] = None,
        position_ids_lfp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the distillation loss (Eq. 1):

            L = (1/ny·T) · MSE(y, f_phi(z_lfp))
              + lambda · (1 - (1/T) · sum_t cosine(z_lfp_t, z_spike_t))

        Args:
            spikes                : (B, T, N_spike) spike counts
            lfp                   : (B, T, N_lfp)   z-scored LFP
            subject_sessions_spike: list[str] — session keys for teacher tokenizer
            subject_sessions_lfp  : list[str] — session keys for student tokenizer
            position_ids_spike    : (B, T) optional
            position_ids_lfp      : (B, T) optional

        Returns:
            loss       : scalar total distillation loss
            loss_ae    : scalar autoencoding (MSE) component
            loss_align : scalar cosine alignment component
        """
        # Teacher: no gradient, full eval
        with torch.no_grad():
            self.teacher.eval()
            z_spike = self.teacher.encode(
                spikes=spikes,
                subject_sessions=subject_sessions_spike,
                position_ids=position_ids_spike,
            )   # (B, T, D)

        # Student
        z_lfp, lfp_hat = self.student(
            lfp=lfp,
            subject_sessions=subject_sessions_lfp,
            position_ids=position_ids_lfp,
        )   # z_lfp: (B, T, D),  lfp_hat: (B, T, N_lfp)

        # Autoencoding loss: (1 / n_lfp·T) · ||y - f_phi(z_lfp)||^2  (Eq.1 first term)
        loss_ae = F.mse_loss(lfp_hat, lfp, reduction="mean")

        # Cosine alignment loss: 1 - (1/T) · sum_t <z_lfp_t, z_spike_t> / (||z_lfp_t||·||z_spike_t||)
        # F.cosine_similarity operates on last dim; average over time and batch.
        # z_spike and z_lfp should have the same (B, T, D) shape.
        cos_sim = F.cosine_similarity(z_lfp, z_spike, dim=-1)   # (B, T)
        loss_align = 1.0 - cos_sim.mean()

        loss = loss_ae + self.lambda_align * loss_align
        return loss, loss_ae, loss_align

    def student_parameters(self):
        """Return only student parameters (for optimizer)."""
        return self.student.parameters()

    def encode_lfp(
        self,
        lfp: torch.Tensor,
        subject_sessions: List[str],
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract pooled LFP representations from the trained student.
        Used for downstream linear decoding (paper A.4).

        Returns:
            z_lfp : (B, T, D)
        """
        z_lfp, _ = self.student(
            lfp=lfp,
            subject_sessions=subject_sessions,
            position_ids=position_ids,
        )
        return z_lfp

    def save_checkpoint(self, path: str, config: dict) -> None:
        """Save full model (teacher + student) weights and config."""
        torch.save({"state_dict": self.state_dict(), "config": config}, path)
        std_logger.info(f"DistillationModel checkpoint saved to {path}")

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        teacher: SpikeMAEModel,
        student: LFPStudentModel,
        lambda_align: float = 5.0,
    ) -> "DistillationModel":
        """Load student weights from a saved checkpoint into a new DistillationModel."""
        ckpt = torch.load(path, map_location="cpu")
        model = cls(teacher=teacher, student=student, lambda_align=lambda_align)
        model.load_state_dict(ckpt["state_dict"])
        std_logger.info(f"DistillationModel loaded from {path}")
        return model
