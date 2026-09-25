"""
Same comparison as inference_makin_lfp_official.py, using checkpoints trained
in this repository instead of the released weights.

  ms_lfp_r2        undistilled MS-LFP, each session keeps its own name
  distilled_lfp_r2 student distilled on MonkeyI_20160622_01; every eval
                   session is tokenized as that session

Both numbers are val+test R2 of a linear velocity decoder fit on train.

    python -m cross_modal_distillation.inference_makin_lfp
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from cross_modal_distillation.data.makin_sessions import DISTILL_SESSION, MAKIN_EVAL_SESSIONS
from cross_modal_distillation.models.distillation import LFPStudentModel
from cross_modal_distillation.utility.utils import init_logger

log = init_logger("InferenceMakinLFP")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/gfs/share/datasets/MakinRT_gzjTest")
METADATA_PATH = DATA_DIR / "metadata_3a4b4.csv"
RESULT_DIR = DATA_DIR / "results"
DEFAULT_MS_CKPT = REPO_ROOT / "results" / "checkpoints" / "ms_lfp" / "best_ms_lfp.ckpt"
DEFAULT_DISTILL_CKPT = (
    REPO_ROOT / "results" / "checkpoints" / "distill" / DISTILL_SESSION / "best_distill.ckpt"
)


def pick_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def require_ckpt(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_lfp_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model = LFPStudentModel(
        session_d_input_dict=cfg["session_d_input_dict"],
        n_lfp=cfg["n_lfp"],
        spatial_patch_size=cfg["spatial_patch_size"],
        d_hidden=cfg["d_hidden"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_heads=cfg["num_heads"],
        kernel_size=cfg["kernel_size"],
        dilation=cfg["dilation"],
        dropout=cfg["dropout"],
    )
    state = ckpt["state_dict"]
    if any(key.startswith("student.") for key in state):
        state = {
            key[len("student.") :]: value
            for key, value in state.items()
            if key.startswith("student.")
        }
    model.load_state_dict(state)
    model.eval().to(device)
    return model, list(cfg["session_d_input_dict"]), int(cfg["n_lfp"])


def embed_rows(model, rows, session_name, n_lfp, device):
    embeddings = {}
    for row in rows.itertuples(index=False):
        data = torch.load(row.path, weights_only=False)
        lfp = data["lfp"].float()
        if lfp.shape[-1] != n_lfp:
            raise ValueError(
                f"{row.segment_filename} has {lfp.shape[-1]} channels, model expects {n_lfp}"
            )
        name = session_name if session_name is not None else row.subject_session
        with torch.no_grad():
            z = model(lfp.unsqueeze(0).to(device), [name])[0].squeeze(0).cpu()
        embeddings[row.segment_filename] = (z, data["kinem"])
    return embeddings


def heldout_r2(rows, embeddings):
    train_x, train_y, test_x, test_y = [], [], [], []
    for row in rows.itertuples(index=False):
        z, kinem = embeddings[row.segment_filename]
        flat = z.reshape(-1, z.shape[-1])
        if row.split == "train":
            train_x.append(flat)
            train_y.append(kinem)
        else:
            test_x.append(flat)
            test_y.append(kinem)
    if not train_x or not test_x:
        return float("nan")
    train_x = torch.cat(train_x).numpy()
    train_y = torch.cat(train_y).numpy()
    test_x = torch.cat(test_x).numpy()
    test_y = torch.cat(test_y).numpy()
    lr = LinearRegression().fit(train_x, train_y)
    return float(lr.score(test_x, test_y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms-ckpt", default=str(DEFAULT_MS_CKPT))
    parser.add_argument("--distill-ckpt", default=str(DEFAULT_DISTILL_CKPT))
    args = parser.parse_args()

    device = pick_device()
    ms_path = require_ckpt(args.ms_ckpt)
    distill_path = require_ckpt(args.distill_ckpt)
    ms_model, ms_sessions, ms_n = load_lfp_model(ms_path, device)
    distill_model, distill_sessions, distill_n = load_lfp_model(distill_path, device)
    if distill_sessions != [DISTILL_SESSION]:
        raise ValueError(
            f"Distilled checkpoint must contain only {DISTILL_SESSION}, found {distill_sessions}"
        )
    missing = [ss for ss in MAKIN_EVAL_SESSIONS if ss not in ms_sessions]
    if missing:
        raise ValueError(f"MS-LFP checkpoint is missing sessions: {missing}")
    log.info(f"Device={device}")
    log.info(f"MS-LFP {ms_path}")
    log.info(f"Distilled {distill_path} tokenized as {DISTILL_SESSION}")

    metadata = pd.read_csv(METADATA_PATH)
    ms_scores, distill_scores = [], []
    for subject_session in MAKIN_EVAL_SESSIONS:
        rows = metadata[metadata.subject_session == subject_session]
        if rows.empty:
            raise RuntimeError(f"No metadata rows for {subject_session}")
        ms_emb = embed_rows(ms_model, rows, None, ms_n, device)
        distill_emb = embed_rows(distill_model, rows, DISTILL_SESSION, distill_n, device)
        ms_r2 = heldout_r2(rows, ms_emb)
        distill_r2 = heldout_r2(rows, distill_emb)
        log.info(f"{subject_session}  ms_lfp_r2={ms_r2:.4f}  distilled_lfp_r2={distill_r2:.4f}")
        ms_scores.append(ms_r2)
        distill_scores.append(distill_r2)

    summary = pd.DataFrame(
        {
            "subject_session": MAKIN_EVAL_SESSIONS,
            "ms_lfp_r2": ms_scores,
            "distilled_lfp_r2": distill_scores,
        }
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULT_DIR / "monkeyI_lfp_decoding.csv"
    summary.to_csv(out_csv, index=False)
    log.info(f"Wrote {out_csv}\n{summary.to_string(index=False)}")
    log.info(
        f"Mean R2  MS-LFP={summary.ms_lfp_r2.mean():.4f}  "
        f"Distilled={summary.distilled_lfp_r2.mean():.4f}"
    )


if __name__ == "__main__":
    main()
