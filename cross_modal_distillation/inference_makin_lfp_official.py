"""
Monkey I LFP inference using official MS-LFP / Distilled-LFP checkpoints.

Builds the Makin-only 10s then 5s datasets (no FlintCO), extracts embeddings,
and fits a linear decoder from embeddings to cursor velocity (paper A.4 / Fig. 5).

Usage (from repo root, after `pip install -e .`):

    python -m cross_modal_distillation.inference_makin_lfp_official
"""

import os
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from cross_modal_distillation.build import build_config, build_dataloader, build_module
from cross_modal_distillation.models.model import Model
from cross_modal_distillation.utility.utils import init_logger

log = init_logger("InferenceMakinLFP")

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "cross_modal_distillation" / "checkpoints"
# Writable working copy (original MakinRT/ is owned by another user).
DATA_DIR = Path("/gfs/share/datasets/MakinRT_gzjTest")
RESULT_DIR = DATA_DIR / "results"

TEST_SESSIONS_MONKEY_I = [
    "MonkeyI_20160624_03",
    "MonkeyI_20160627_01",
    "MonkeyI_20160916_01",
    "MonkeyI_20160921_01",
    "MonkeyI_20160927_04",
    "MonkeyI_20160927_06",
    "MonkeyI_20160930_02",
    "MonkeyI_20160930_05",
    "MonkeyI_20161005_06",
    "MonkeyI_20161006_02",
    "MonkeyI_20161007_02",
    "MonkeyI_20161011_03",
    "MonkeyI_20161014_04",
    "MonkeyI_20161017_02",
    "MonkeyI_20161024_03",
    "MonkeyI_20161025_04",
    "MonkeyI_20161026_03",
    "MonkeyI_20161027_03",
]

DISTILL_SESSION = "MonkeyI_20160622_01"


def decode_kinem(embedding_dir, metadata_df, subject_sessions):
    train_scores, test_scores = [], []
    for ss in subject_sessions:
        metadata_ss = metadata_df[metadata_df["subject_session"] == ss].reset_index(
            drop=True
        )
        if len(metadata_ss) == 0:
            log.warning(f"No metadata rows for {ss}, skip decoding.")
            train_scores.append(float("nan"))
            test_scores.append(float("nan"))
            continue

        train_kinem, train_embeddings = [], []
        test_kinem, test_embeddings = [], []

        for _, meta_row in metadata_ss.iterrows():
            data = torch.load(meta_row.path, weights_only=True)
            emb_path = os.path.join(embedding_dir, meta_row.segment_filename)
            if not os.path.exists(emb_path):
                log.warning(f"Missing embedding {emb_path}")
                continue
            emb = torch.load(emb_path, weights_only=True)
            kinem = data["kinem"]
            emb_flat = emb.reshape(-1, emb.shape[-1])
            if meta_row.split == "train":
                train_kinem.append(kinem)
                train_embeddings.append(emb_flat)
            else:
                test_kinem.append(kinem)
                test_embeddings.append(emb_flat)

        if not train_kinem or not test_kinem:
            log.warning(f"{ss}: empty train or test split, skip.")
            train_scores.append(float("nan"))
            test_scores.append(float("nan"))
            continue

        train_kinem = torch.cat(train_kinem)
        train_embeddings = torch.cat(train_embeddings)
        test_kinem = torch.cat(test_kinem)
        test_embeddings = torch.cat(test_embeddings)

        lr = LinearRegression()
        lr.fit(train_embeddings.numpy(), train_kinem.numpy())
        train_score = lr.score(train_embeddings.numpy(), train_kinem.numpy())
        test_score = lr.score(test_embeddings.numpy(), test_kinem.numpy())
        log.info(f"{ss}  train R2={train_score:.4f}  val+test R2={test_score:.4f}")
        train_scores.append(train_score)
        test_scores.append(test_score)
    return train_scores, test_scores


def pick_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def main():
    device = pick_device()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Device={device}")
    log.info("Building 10s Makin-only dataset (triggers LFP preprocessing if needed)...")
    cfg_10s = build_config(config_name="ms_lfp_makin_10s")
    build_module(config=cfg_10s)

    log.info("Building 5s dataset from 10s segments...")
    cfg_5s = build_config(config_name="ms_lfp_makin")
    dataset_5s = build_module(config=cfg_5s)
    dataset_5s.reduce_metadata(subject_sessions=TEST_SESSIONS_MONKEY_I)
    metadata_df = dataset_5s.metadata._metadata_df.copy()
    log.info(f"5s segments for inference: {len(dataset_5s)}")

    loader = build_dataloader(
        dataset=dataset_5s,
        collate_fn_name="collate_with_metadata_fn",
        batch_size=32,
        num_workers=0,
    )

    # ----- MS-LFP baseline -----
    ms_emb_dir = str(RESULT_DIR / "ms_lfp_embeddings")
    log.info("Loading MS-LFP checkpoint...")
    ms_lfp = Model(
        ckpt_path=str(CKPT_DIR / "ms_lfp.ckpt"),
        ckpt_metadata_path=str(CKPT_DIR / "ms_lfp_metadata.csv"),
    )
    ms_lfp.get_embeddings(
        loader,
        save_embeddings=True,
        save_dir=ms_emb_dir,
        device=device,
    )
    _, ms_scores = decode_kinem(ms_emb_dir, metadata_df, TEST_SESSIONS_MONKEY_I)

    # ----- Distilled LFP (trained on 20160622_01) -----
    dist_emb_dir = str(RESULT_DIR / "distilled_lfp_embeddings")
    log.info("Loading Distilled LFP checkpoint (MonkeyI_20160622_01)...")
    distilled = Model(
        ckpt_path=str(CKPT_DIR / "distilled_lfp_monkeyI_20160622_01.ckpt"),
        ckpt_metadata_path=str(CKPT_DIR / "distilled_lfp_monkeyI_20160622_01_metadata.csv"),
    )
    for ss in TEST_SESSIONS_MONKEY_I:
        log.info(f"Distilled inference on {ss} (tokenized as {DISTILL_SESSION})...")
        ds = deepcopy(dataset_5s)
        ds.reduce_metadata(subject_sessions=ss)
        ds.metadata._metadata_df["subject_session"] = DISTILL_SESSION
        dist_loader = build_dataloader(
            dataset=ds,
            collate_fn_name="collate_with_metadata_fn",
            batch_size=32,
            num_workers=0,
        )
        distilled.get_embeddings(
            dist_loader,
            save_embeddings=True,
            save_dir=dist_emb_dir,
            device=device,
        )

    _, dist_scores = decode_kinem(dist_emb_dir, metadata_df, TEST_SESSIONS_MONKEY_I)

    summary = pd.DataFrame(
        {
            "subject_session": TEST_SESSIONS_MONKEY_I,
            "ms_lfp_r2": ms_scores,
            "distilled_lfp_r2": dist_scores,
        }
    )
    out_csv = RESULT_DIR / "monkeyI_lfp_decoding.csv"
    summary.to_csv(out_csv, index=False)
    log.info(f"Wrote {out_csv}\n{summary.to_string(index=False)}")
    log.info(
        f"Mean R2  MS-LFP={summary.ms_lfp_r2.mean():.4f}  "
        f"Distilled={summary.distilled_lfp_r2.mean():.4f}"
    )


if __name__ == "__main__":
    main()
