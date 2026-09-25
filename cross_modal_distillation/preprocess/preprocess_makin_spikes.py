"""
Bin MakinRT spike times onto the processed LFP time axis and cut 5-second
segments whose filenames match metadata_3a4b4.csv.

Spike times come from raw/<indy_session>.mat field `spikes` (5 units x 96
channels). Counts are placed on the same 10 ms grid as
processed_raw_data_*/MonkeyI_<session>.pt["t"]. Units whose mean rate on that
grid is below 1 Hz are dropped. Each 5-second LFP segment id `sid` maps to
samples [start, start+500) with

    start = (sid // 2) * 1000 + (sid % 2) * 500

which is how 10-second segments are later split into 5-second segments.

Usage (from the repository root):

    python -m cross_modal_distillation.preprocess.preprocess_makin_spikes
"""

import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch

from scipy import signal

from cross_modal_distillation.data.makin_sessions import TEACHER_PRETRAIN_SESSIONS
from cross_modal_distillation.utility.utils import init_logger

log = init_logger("PreprocessMakinSpikes")

ROOT = "/gfs/share/datasets/MakinRT_gzjTest"
METADATA_PATH = os.path.join(ROOT, "metadata_3a4b4.csv")
RAW_DIR = os.path.join(ROOT, "raw")
PROCESSED_RAW_DIR = os.path.join(
    ROOT,
    "processed_raw_data_10ms_zScLFP_zScKinem_LFP_lpCut5.0e+01Hz_hpCut5.0e-02Hz",
)
SPIKE_DIR = os.path.join(ROOT, "spike_segments_5s")
SESSION_INFO_PATH = os.path.join(ROOT, "spike_session_info.csv")
PRETRAIN_METADATA_PATH = os.path.join(ROOT, "spike_pretrain_metadata.csv")
MIN_RATE_HZ = 1.0
STEPS_10S = 1000
STEPS_5S = 500


def _load_unit_times(mat_file, spikes, unit, channel):
    ref = spikes[unit, channel]
    if not isinstance(ref, h5py.Reference) or not ref:
        return np.zeros(0, dtype=np.float64)
    arr = np.array(mat_file[ref]).astype(np.float64).ravel()
    if arr.size == 0:
        return arr
    # Empty MATLAB cells are stored as a length-2 zero vector.
    if arr.size <= 2 and np.all(arr == 0):
        return np.zeros(0, dtype=np.float64)
    return arr


def trial_trimmed_time(mat_path, delta_s=0.01):
    """Match MakinRTDataset trial trimming without running the LFP filters."""
    with h5py.File(mat_path, "r") as mat_file:
        t = np.array(mat_file["t"]).squeeze().astype(np.float64)
        target_pos = np.array(mat_file["target_pos"]).T
    num_steps = int((t[-1] - t[0]) / delta_s)
    _, t = signal.resample(np.zeros((len(t), 1)), num_steps, t=t)
    trial_start = np.where(np.diff(target_pos, axis=0).sum(axis=1) != 0)[0] + 1
    trial_start = (trial_start // (target_pos.shape[0] / num_steps)).astype(np.int32)
    return np.asarray(t[trial_start[0] : trial_start[-1]], dtype=np.float64)


def bin_counts(mat_path, t):
    t = np.asarray(t, dtype=np.float64)
    dt = float(np.median(np.diff(t)))
    edges = np.empty(t.shape[0] + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (t[:-1] + t[1:])
    edges[0] = t[0] - 0.5 * dt
    edges[-1] = t[-1] + 0.5 * dt
    duration = t.shape[0] * dt
    kept = []
    with h5py.File(mat_path, "r") as mat_file:
        spikes = mat_file["spikes"]
        n_units, n_channels = spikes.shape
        for unit in range(n_units):
            for channel in range(n_channels):
                times = _load_unit_times(mat_file, spikes, unit, channel)
                counts = np.histogram(times, bins=edges)[0]
                if counts.sum() / duration < MIN_RATE_HZ:
                    continue
                kept.append(counts.astype(np.float32))
    if not kept:
        raise RuntimeError(f"No units kept for {mat_path}")
    return np.stack(kept, axis=1)


def bin_session_spikes(subject_session):
    session = subject_session.split("_", 1)[1]
    mat_path = os.path.join(RAW_DIR, f"indy_{session}.mat")
    raw_path = os.path.join(PROCESSED_RAW_DIR, f"{subject_session}.pt")
    processed = torch.load(raw_path, weights_only=False)
    t = processed["t"].detach().cpu().numpy().astype(np.float64)
    return bin_counts(mat_path, t)


def _assign_splits(rows):
    frame = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = int(0.1 * len(frame))
    test_size = int(0.1 * len(frame))
    frame["split"] = "train"
    frame.loc[:val_size, "split"] = "val"
    frame.loc[val_size : (val_size + test_size), "split"] = "test"
    return frame


def write_teacher_pretrain_spikes(info_rows):
    os.makedirs(SPIKE_DIR, exist_ok=True)
    meta_rows = []
    for subject_session in TEACHER_PRETRAIN_SESSIONS:
        session = subject_session.split("_", 1)[1]
        mat_path = os.path.join(RAW_DIR, f"indy_{session}.mat")
        log.info(f"Spike-only binning for teacher session {subject_session}")
        t = trial_trimmed_time(mat_path)
        counts = bin_counts(mat_path, t)
        info_rows.append(
            {
                "subject_session": subject_session,
                "n_spike": int(counts.shape[1]),
                "n_time": int(counts.shape[0]),
            }
        )
        n_parent = counts.shape[0] // STEPS_10S
        session_rows = []
        for parent in range(n_parent):
            for half in range(2):
                segment_id = parent * 2 + half
                start = segment_start(segment_id)
                segment = counts[start : start + STEPS_5S]
                filename = f"{subject_session}_{segment_id}.pt"
                path = os.path.join(SPIKE_DIR, filename)
                torch.save({"spikes": torch.from_numpy(np.ascontiguousarray(segment))}, path)
                session_rows.append(
                    {
                        "subject_session": subject_session,
                        "segment_filename": filename,
                        "path": path,
                        "split": "train",
                    }
                )
        meta_rows.append(_assign_splits(session_rows))
        log.info(f"{subject_session}: {counts.shape[1]} units, {n_parent * 2} segments")
    pd.concat(meta_rows, ignore_index=True).to_csv(PRETRAIN_METADATA_PATH, index=False)


def segment_start(segment_id):
    parent, half = divmod(segment_id, 2)
    return parent * STEPS_10S + half * STEPS_5S


def main():
    os.makedirs(SPIKE_DIR, exist_ok=True)
    metadata = pd.read_csv(METADATA_PATH)
    session_rows = defaultdict(list)
    for row in metadata.itertuples(index=False):
        session_rows[row.subject_session].append(row)

    info_rows = []
    for subject_session, rows in session_rows.items():
        log.info(f"Binning spikes for {subject_session} ({len(rows)} segments)")
        counts = bin_session_spikes(subject_session)
        n_neurons = int(counts.shape[1])
        info_rows.append(
            {"subject_session": subject_session, "n_spike": n_neurons, "n_time": int(counts.shape[0])}
        )
        log.info(f"{subject_session}: kept {n_neurons} units, T={counts.shape[0]}")

        for row in rows:
            segment_id = int(row.segment_filename[:-3].split("_")[-1])
            start = segment_start(segment_id)
            segment = counts[start : start + STEPS_5S]
            if segment.shape[0] != STEPS_5S:
                raise RuntimeError(
                    f"{row.segment_filename} slice {start}:{start + STEPS_5S} "
                    f"has length {segment.shape[0]}"
                )
            out_path = os.path.join(SPIKE_DIR, row.segment_filename)
            torch.save(
                {"spikes": torch.from_numpy(np.ascontiguousarray(segment))},
                out_path,
            )

    write_teacher_pretrain_spikes(info_rows)
    pd.DataFrame(info_rows).to_csv(SESSION_INFO_PATH, index=False)
    log.info(f"Wrote paired spike segments to {SPIKE_DIR}")
    log.info(f"Session info: {SESSION_INFO_PATH}")


if __name__ == "__main__":
    main()
