"""
SpikeMAEModel — Spike Masked Autoencoder (Teacher model, paper Fig. 7 / Appendix A.1)

Architecture:
  Input spike counts (B, T, N_neurons)
      │
  PatchTokenizer  [use_embedding_for_input=True, learn_patch_embedding=True]
      │  → tokens (B, T*P, d_encoder=256)
  Random token drop  (mask_ratio=0.60, across time AND space)
      │  → visible tokens only
  RoPE encoder  (10 layers, d=256)
      │
  Insert mask_token + space_embedding at masked positions
      │
  Linear(256→192)  [논문이 명시하지 않은 차원 접합 레이어, 필요상 추가]
      │
  RoPE predictor  (4 layers, d=192)
      │
  Linear(192→64)  [64-dimensional down-projection, paper A.2]
      │
  softplus(·)  → Poisson rate λ  (our convention: λ = softplus(z))
      │
  Poisson NLL loss on masked positions, pad dims excluded

Training modes:
  - pretrain : MAE on multiple sessions (call forward())
  - finetune : MAE on single held-out session (same forward(), reset space embeddings
               via tokenizer.update_for_new_sessions() before calling)

Inference (for distillation teacher):
  - encode()  : no masking, returns pooled (B, T, d_encoder) representations
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from cross_modal_distillation.models.custom_transformer import CustomTransformer
from cross_modal_distillation.models.tokenizer import PatchTokenizer
from cross_modal_distillation.utility.utils import init_logger

std_logger = init_logger("SpikeMAE")


# ---------------------------------------------------------------------------
# Helper: scatter-mean pooling (same logic as Model.pool_embeddings)
# ---------------------------------------------------------------------------

def pool_by_position(embeddings: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings that share the same time-step position.

    Args:
        embeddings  : (B, N_tokens, D)
        position_ids: (B, N_tokens) — integer time-step index for every token

    Returns:
        pooled : (B, T_max+1, D)  where T_max = max(position_ids)
    """
    B, N, D = embeddings.shape
    max_pos = int(position_ids.max().item())

    if len(position_ids.shape) == 3:
        if position_ids.shape[-1] == 1:
            position_ids = position_ids.squeeze(-1)
        else:
            raise ValueError("position_ids must be 2-D or 3-D with last dim=1")

    idx = repeat(position_ids.long(), "b n -> b n d", d=D)
    pooled = torch.zeros(B, max_pos + 1, D, device=embeddings.device, dtype=embeddings.dtype)
    pooled = pooled.scatter_reduce(1, idx, embeddings, reduce="mean")
    return pooled  # (B, T, D)


# ---------------------------------------------------------------------------
# SpikeMAEModel
# ---------------------------------------------------------------------------

class SpikeMAEModel(nn.Module):
    """
    Multi-session Spike Masked Autoencoder.

    Parameters
    ----------
    session_d_input_dict : dict[str, int]
        Maps "subject_session" → number of neurons (n_spike).
        Required to build per-session space embeddings in the tokenizer.
    spatial_patch_size : int
        S in the paper (default 64 for spikes).
    d_encoder : int
        Hidden dim of the encoder transformer (default 256).
    d_predictor : int
        Hidden dim of the predictor transformer (default 192).
    num_encoder_layers : int
        Depth of the encoder (default 10).
    num_predictor_layers : int
        Depth of the predictor (default 4).
    num_heads : int
        Attention heads (default 8, must divide d_encoder and d_predictor).
    max_count : int
        Maximum spike count per bin (clips higher values, default 5).
    mask_ratio : float
        Fraction of tokens to mask during MAE training (default 0.60).
    dropout : float
        Dropout rate in transformer layers.
    use_flash_attention : bool
        Whether to use Flash-Attention for memory efficiency.
    max_pos : int
        Maximum sequence position for RoPE (must cover T * P tokens).
    """

    def __init__(
        self,
        session_d_input_dict: Dict[str, int],
        spatial_patch_size: int = 64,
        d_encoder: int = 256,
        d_predictor: int = 192,
        num_encoder_layers: int = 10,
        num_predictor_layers: int = 4,
        num_heads: int = 8,
        max_count: int = 5,
        mask_ratio: float = 0.60,
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        max_pos: int = 1024,
    ):
        super().__init__()

        self.spatial_patch_size = spatial_patch_size
        self.d_encoder = d_encoder
        self.d_predictor = d_predictor
        self.mask_ratio = mask_ratio
        self.max_count = max_count

        # ------------------------------------------------------------------ #
        # Tokenizer  (Fig. 1)
        # use_embedding_for_input=True  → count lookup-table value embedding
        # learn_patch_embedding=True    → session-specific space embedding
        # ------------------------------------------------------------------ #
        self.tokenizer = PatchTokenizer(
            spatial_patch_size=spatial_patch_size,
            session_d_input_dict=session_d_input_dict,
            d_hidden=d_encoder,
            learn_patch_embedding=True,
            use_embedding_for_input=True,
            max_count=max_count,
        )

        # ------------------------------------------------------------------ #
        # Learnable mask token — inserted at dropped positions (Appendix A.1)
        # ------------------------------------------------------------------ #
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_encoder))
        nn.init.normal_(self.mask_token, std=0.02)

        # ------------------------------------------------------------------ #
        # Encoder  (RoPE transformer, 10 layers, d=256)
        # ------------------------------------------------------------------ #
        self.encoder = CustomTransformer(
            num_layers=num_encoder_layers,
            d_hidden=d_encoder,
            num_heads=num_heads,
            dropout=dropout,
            attention_module_name="RotarySelfAttention",
            use_flash_attention=use_flash_attention,
            max_pos=max_pos,
            use_final_norm=True,
        )

        # ------------------------------------------------------------------ #
        # Enc→Pred projection  (256→192, paper does not name this layer but
        # the dimension change is required to feed into the predictor)
        # ------------------------------------------------------------------ #
        self.enc_to_pred = nn.Linear(d_encoder, d_predictor, bias=True)

        # ------------------------------------------------------------------ #
        # Predictor  (RoPE transformer, 4 layers, d=192)
        # ------------------------------------------------------------------ #
        self.predictor = CustomTransformer(
            num_layers=num_predictor_layers,
            d_hidden=d_predictor,
            num_heads=num_heads,
            dropout=dropout,
            attention_module_name="RotarySelfAttention",
            use_flash_attention=use_flash_attention,
            max_pos=max_pos,
            use_final_norm=True,
        )

        # ------------------------------------------------------------------ #
        # Reconstruction head  (paper A.2: "64-dimensional down-projection")
        # softplus(z) is used as the Poisson rate λ (our convention)
        # ------------------------------------------------------------------ #
        self.recon_head = nn.Linear(d_predictor, spatial_patch_size, bias=True)

    # ---------------------------------------------------------------------- #
    # Internal utilities
    # ---------------------------------------------------------------------- #

    def _random_mask(
        self, tokens: torch.Tensor, seq_lens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Randomly drop mask_ratio fraction of tokens (independently per sample).

        Handles both batched tensors (B>1, fixed length) and the variable-length
        convention (B=1, all samples concatenated along dim=1 with seq_lens).

        Returns
        -------
        visible_tokens  : tokens at unmasked positions
        mask_bool       : (B, N_total) bool mask — True = this token was MASKED
        visible_seq_lens: list of visible token counts per sample
        masked_seq_lens : list of masked token counts per sample
        """
        B, N_total, D = tokens.shape
        device = tokens.device

        if B == 1 and len(seq_lens) > 1:
            # Variable-length path: process each sample individually
            visible_list, mask_list = [], []
            vis_lens, mask_lens_out = [], []

            chunks = torch.split(tokens[0], seq_lens, dim=0)  # each (n_i, D)
            for chunk in chunks:
                n_i = chunk.shape[0]
                num_mask = max(1, int(self.mask_ratio * n_i))
                perm = torch.randperm(n_i, device=device)
                mask_idx = perm[:num_mask]
                vis_idx = perm[num_mask:]

                m = torch.zeros(n_i, dtype=torch.bool, device=device)
                m[mask_idx] = True

                visible_list.append(chunk[vis_idx])
                mask_list.append(m)
                vis_lens.append(n_i - num_mask)
                mask_lens_out.append(num_mask)

            visible_tokens = torch.cat(visible_list, dim=0).unsqueeze(0)  # (1, N_vis, D)
            mask_bool = torch.cat(mask_list, dim=0).unsqueeze(0)           # (1, N_total)
            return visible_tokens, mask_bool, vis_lens, mask_lens_out

        else:
            # Batched path (all samples same length)
            n_per = seq_lens[0]
            num_mask = max(1, int(self.mask_ratio * n_per))

            # independent shuffle per batch item
            perm = torch.stack(
                [torch.randperm(n_per, device=device) for _ in range(B)]
            )  # (B, n_per)
            mask_idx = perm[:, :num_mask]   # (B, num_mask)
            vis_idx  = perm[:, num_mask:]   # (B, n_vis)

            # gather visible / mask-bool
            visible_tokens = torch.gather(
                tokens, 1, vis_idx.unsqueeze(-1).expand(-1, -1, D)
            )  # (B, n_vis, D)

            mask_bool = torch.zeros(B, n_per, dtype=torch.bool, device=device)
            mask_bool.scatter_(1, mask_idx, True)

            vis_lens_out = [n_per - num_mask] * B
            mask_lens_out = [num_mask] * B
            return visible_tokens, mask_bool, vis_lens_out, mask_lens_out

    def _add_space_embeddings_at_positions(
        self,
        full_tokens: torch.Tensor,   # (B, N_total, D) — with mask_token at masked spots
        patch_ids: torch.Tensor,     # (B, N_total) — patch index for each token
        subject_sessions: List[str],
        seq_lens: List[int],
    ) -> torch.Tensor:
        """
        Add session-specific space embeddings to tokens at masked positions.
        Paper Appendix A.1: 'add their corresponding space embeddings to the mask tokens'.
        """
        if self.tokenizer.patch_embeddings is None:
            return full_tokens

        B = full_tokens.shape[0]
        result = full_tokens.clone()

        if B == 1 and len(seq_lens) > 1:
            pid_chunks = torch.split(patch_ids[0], seq_lens, dim=0)
            offset = 0
            for i, (ss, pid_chunk) in enumerate(zip(subject_sessions, pid_chunks)):
                sp_emb = self.tokenizer.patch_embeddings[ss](pid_chunk)  # (n_i, D)
                result[0, offset: offset + seq_lens[i]] += sp_emb
                offset += seq_lens[i]
        else:
            for i, ss in enumerate(subject_sessions):
                sp_emb = self.tokenizer.patch_embeddings[ss](patch_ids[i])  # (N, D)
                result[i] += sp_emb

        return result

    # ---------------------------------------------------------------------- #
    # Forward (MAE training)
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        spikes: torch.Tensor,
        subject_sessions: List[str],
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MAE forward pass used during both pretraining and fine-tuning.

        Args:
            spikes          : (B, T, N_neurons) spike count tensor (non-negative int)
            subject_sessions: list of 'subject_session' strings of length B
            position_ids    : (B, T) optional time-step indices

        Returns:
            loss   : scalar Poisson NLL loss
            lambda_: (B, N_masked, S) predicted Poisson rates at masked positions
        """
        # 1. Tokenize -------------------------------------------------------
        data_patched, tokens, pos_ids_tok, patch_ids, seq_lens = self.tokenizer(
            x=spikes,
            subject_sessions=subject_sessions,
            position_ids=position_ids,
        )
        # tokens      : (B, T*P, d_encoder)
        # data_patched: (B, T*P, S)  — raw counts per patch (may include pad)
        # pos_ids_tok : (B, T*P)     — time-step position of each token
        # patch_ids   : (B, T*P)     — spatial patch index of each token

        B, N_total, D = tokens.shape

        # 2. Random mask -----------------------------------------------------
        visible_tokens, mask_bool, vis_lens, mask_lens = self._random_mask(tokens, seq_lens)

        # 3. Encoder on visible tokens only ----------------------------------
        if B == 1 and len(vis_lens) > 1:
            # variable-length: pass vis_lens list
            vis_pos_ids = pos_ids_tok[mask_bool.logical_not()].unsqueeze(0)  # approx
            _, enc_out = self.encoder(
                x=visible_tokens, position_ids=vis_pos_ids, seq_lens=vis_lens
            )
        else:
            vis_pos_ids = pos_ids_tok[~mask_bool].view(B, -1)
            _, enc_out = self.encoder(
                x=visible_tokens, position_ids=vis_pos_ids, seq_lens=vis_lens
            )

        # 4. Rebuild full sequence with mask tokens --------------------------
        # Allocate full buffer, fill visible positions, then insert mask_token
        full = tokens.clone()  # keep original for gathering positions
        mask_token_expanded = self.mask_token.expand(B, N_total, D)

        if B == 1 and len(seq_lens) > 1:
            # variable length: scatter enc_out back
            full[mask_bool] = self.mask_token.squeeze(0).expand(mask_bool.sum(), D)
            vis_mask = ~mask_bool
            full[vis_mask] = enc_out.squeeze(0).reshape(-1, D)
        else:
            full[mask_bool] = self.mask_token.reshape(1, D).expand(int(mask_bool.sum()), D)
            full[~mask_bool] = enc_out.reshape(B * enc_out.shape[1], D)

        # Add space embeddings to masked positions (paper A.1)
        full = self._add_space_embeddings_at_positions(
            full, patch_ids, subject_sessions, seq_lens
        )

        # 5. Predictor -------------------------------------------------------
        full_pred = self.enc_to_pred(full)   # (B, N_total, d_predictor)
        _, pred_out = self.predictor(
            x=full_pred, position_ids=pos_ids_tok, seq_lens=seq_lens
        )
        # pred_out : (B, N_total, d_predictor)

        # 6. Reconstruction head → Poisson rates ----------------------------
        logits = self.recon_head(pred_out)           # (B, N_total, S)
        lambda_ = F.softplus(logits)                 # Poisson rate, strictly positive

        # 7. Poisson NLL on masked, non-pad positions -----------------------
        loss = self._mae_loss(
            lambda_=lambda_,
            targets=data_patched,
            mask_bool=mask_bool,
            spikes_raw=spikes,
            subject_sessions=subject_sessions,
            seq_lens=seq_lens,
        )

        return loss, lambda_[mask_bool]

    def _mae_loss(
        self,
        lambda_: torch.Tensor,
        targets: torch.Tensor,
        mask_bool: torch.Tensor,
        spikes_raw: torch.Tensor,
        subject_sessions: List[str],
        seq_lens: List[int],
    ) -> torch.Tensor:
        """
        Poisson NLL on masked positions, excluding spatial padding dimensions.

        paper A.1: "During loss computation, any padded dimensions introduced
        during spatial patching are excluded."
        """
        B = spikes_raw.shape[0]
        device = spikes_raw.device
        total_loss = torch.tensor(0.0, device=device)
        n_valid = 0

        # Build per-session pad mask once (on CPU, small)
        # pad_mask[ss] : (num_patches, S) bool, True = valid neuron
        pad_masks = {}
        for ss in set(subject_sessions):
            d_in = self.tokenizer.session_d_input_dict[ss]
            pad_masks[ss] = self.tokenizer.get_spatial_pad_mask(d_in).to(device)

        if B == 1 and len(seq_lens) > 1:
            # Variable-length: iterate over samples
            chunks_lam = torch.split(lambda_[0], seq_lens, dim=0)
            chunks_tgt = torch.split(targets[0], seq_lens, dim=0)
            chunks_msk = torch.split(mask_bool[0], seq_lens, dim=0)

            for i, ss in enumerate(subject_sessions):
                pm = pad_masks[ss]  # (num_patches_for_ss, S)
                lam_i = chunks_lam[i][chunks_msk[i]]   # (n_masked, S)
                tgt_i = chunks_tgt[i][chunks_msk[i]]   # (n_masked, S)

                # pad mask indexed by patch_id within masked tokens
                # (we already excluded pad in tokenizer via pad_value,
                # but we also mask the loss explicitly)
                num_patches = pm.shape[0]
                # repeat pad_mask to match masked tokens
                pm_flat = pm.reshape(-1, self.spatial_patch_size)  # (P, S)
                # We don't have direct patch-level index here for masked tokens;
                # build a repeating valid mask of same size as lam_i
                n_masked = lam_i.shape[0]
                valid_mask = pm_flat.repeat(
                    (n_masked // num_patches) + 1, 1
                )[:n_masked]  # (n_masked, S)

                lam_valid = lam_i[valid_mask]
                tgt_valid = tgt_i[valid_mask].float()
                if tgt_valid.numel() > 0:
                    total_loss = total_loss + F.poisson_nll_loss(
                        lam_valid, tgt_valid, log_input=False, full=False, reduction="sum"
                    )
                    n_valid += tgt_valid.numel()
        else:
            for b_idx, ss in enumerate(subject_sessions):
                pm = pad_masks[ss]  # (num_patches, S)
                num_patches = pm.shape[0]
                # Replicate pad mask across time steps so shape matches N_total
                T_steps = seq_lens[b_idx] // num_patches if num_patches > 0 else 1
                pm_expanded = pm.unsqueeze(0).expand(T_steps, -1, -1)  # (T, P, S)
                pm_flat = pm_expanded.reshape(-1, self.spatial_patch_size)  # (T*P, S)

                mask_b = mask_bool[b_idx]                  # (N_total,)
                lam_b  = lambda_[b_idx][mask_b]            # (n_masked, S)
                tgt_b  = targets[b_idx][mask_b].float()    # (n_masked, S)

                # Align pad mask to masked positions
                n_masked = lam_b.shape[0]
                valid_m = pm_flat.repeat(
                    (n_masked // pm_flat.shape[0]) + 1, 1
                )[:n_masked]  # (n_masked, S)

                lam_valid = lam_b[valid_m]
                tgt_valid = tgt_b[valid_m]
                if tgt_valid.numel() > 0:
                    total_loss = total_loss + F.poisson_nll_loss(
                        lam_valid, tgt_valid, log_input=False, full=False, reduction="sum"
                    )
                    n_valid += tgt_valid.numel()

        return total_loss / max(n_valid, 1)

    # ---------------------------------------------------------------------- #
    # encode()  — no masking, returns pooled representations (for distillation)
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def encode(
        self,
        spikes: torch.Tensor,
        subject_sessions: List[str],
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode spike signals without masking and return time-step pooled
        representations.  Used as the frozen teacher in Fig. 2.

        Args:
            spikes          : (B, T, N_neurons)
            subject_sessions: list[str] of length B
            position_ids    : (B, T) optional

        Returns:
            pooled : (B, T, d_encoder)  — mean-pooled over spatial patches
        """
        _, tokens, pos_ids_tok, _, seq_lens = self.tokenizer(
            x=spikes,
            subject_sessions=subject_sessions,
            position_ids=position_ids,
        )
        _, enc_out = self.encoder(
            x=tokens, position_ids=pos_ids_tok, seq_lens=seq_lens
        )
        pooled = pool_by_position(enc_out, pos_ids_tok)  # (B, T, D)
        return pooled

    # ---------------------------------------------------------------------- #
    # Checkpoint helpers
    # ---------------------------------------------------------------------- #

    def save_checkpoint(self, path: str, config: dict) -> None:
        """Save model weights + config dict to a .ckpt file."""
        torch.save({"state_dict": self.state_dict(), "config": config}, path)
        std_logger.info(f"SpikeMAEModel checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str) -> "SpikeMAEModel":
        """
        Load a SpikeMAEModel from a checkpoint saved by save_checkpoint().
        The checkpoint must contain 'config' with all __init__ kwargs.
        """
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        std_logger.info(f"SpikeMAEModel loaded from {path}")
        return model
