"""
Preprocess Monkey I (MakinRT) LFP only. Does not load FlintCO or run inference.

The original repository has no standalone preprocessing entry point. Filtering,
downsampling, segmentation, and metadata writing live inside
MakinRTDataset.__init__ and run when the dataset is constructed. This script
triggers that pipeline twice:

1. ms_lfp_makin_10s: extract LFP from raw nwb+mat and cut 10-second segments.
2. ms_lfp_makin: cut 5-second segments from those 10-second segments so the
   train/val/test split matches the paper.

Raw files are read from, and processed files are written to,
/gfs/share/datasets/MakinRT_gzjTest (see ms_lfp_makin_10s.yaml).
Sessions that already have a processed .pt file are skipped unless
force_reprocess_stage1 is set in the yaml.

Usage (from the repository root):

    python -m cross_modal_distillation.preprocess.preprocess_makin
"""

from einops import rearrange

from cross_modal_distillation.build import build_config, build_module
from cross_modal_distillation.data.makin_sessions import DISTILL_SESSION
from cross_modal_distillation.utility.utils import init_logger

log = init_logger("PreprocessMakin")


def _makin(dataset):
    if not dataset.datasets:
        raise RuntimeError("MultiSessionDataset has no inner MakinRT dataset.")
    return dataset.datasets[0]


def _has_session(dataset, subject_session):
    column = dataset.metadata._metadata_df["subject_session"]
    return subject_session in set(column)


def _append_10s(dataset, subject, session):
    subject_session = f"{subject}_{session}"
    if _has_session(dataset, subject_session):
        log.info(f"10s segments already include {subject_session}")
        return
    makin = _makin(dataset)
    log.info(f"Appending 10s segments for {subject_session}")
    makin.process_single_session_segments(subject=subject, session=session)
    makin.metadata.save(makin.metadata_path)


def _append_5s_from_10s(dataset, subject_session):
    if _has_session(dataset, subject_session):
        log.info(f"5s segments already include {subject_session}")
        return
    dataset = _makin(dataset)
    log.info(f"Appending 5s segments for {subject_session}")
    _, existing_hash = dataset.get_segments_processing_hash(
        segment_length=dataset.config.existing_data_segment_length,
        segment_from_existing_data=False,
    )
    existing = dataset.metadata.__class__(
        load_path=dataset.get_metadata_path(existing_hash)
    )
    new_rows = []
    frame = existing._metadata_df
    frame = frame[frame["subject_session"] == subject_session]
    for row in frame.itertuples(index=False):
        existing_segment = __import__("torch").load(row.path, weights_only=False)
        segment_id = dataset.get_segment_id_from_path(row.path)
        new_segment_data = {}
        num_new = None
        for name, value in existing_segment.items():
            new_steps = int(dataset.config.segment_length / (dataset.config.delta / 1000))
            usable = value.shape[0] - (value.shape[0] % new_steps)
            pieces = rearrange(value[:usable], "(b t) n -> b t n", t=new_steps)
            new_segment_data[name] = pieces
            num_new = len(pieces)
        for split_id in range(num_new):
            data_dict = {k: v[split_id].clone() for k, v in new_segment_data.items()}
            path, filename = dataset.save_segment_data(
                data_dict=data_dict,
                subject=row.subject,
                session=row.session,
                segment_id=segment_id * num_new + split_id,
            )
            meta = row._asdict()
            meta["path"] = path
            meta["segment_filename"] = filename
            meta["segments_processing_str"] = dataset.segments_processing_str
            new_rows.append(meta)
    import pandas as pd

    dataset.metadata.concat(new_metadata_df=pd.DataFrame(new_rows))
    dataset.metadata.save(dataset.metadata_path)


def main():
    subject, session = DISTILL_SESSION.split("_", 1)
    log.info("Stage 1-2: extract LFP and write 10-second segments (Makin only).")
    cfg_10s = build_config(config_name="ms_lfp_makin_10s")
    ds_10s = build_module(config=cfg_10s)
    _append_10s(ds_10s, subject, session)
    log.info(f"10s segments ready: {len(ds_10s)}")

    log.info("Stage 2b: chunk 5-second segments from the 10-second segments.")
    cfg_5s = build_config(config_name="ms_lfp_makin")
    ds_5s = build_module(config=cfg_5s)
    _append_5s_from_10s(ds_5s, DISTILL_SESSION)
    log.info(f"5s segments ready: {len(ds_5s)}")
    log.info("MakinRT preprocessing finished.")


if __name__ == "__main__":
    main()
