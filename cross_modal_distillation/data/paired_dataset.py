"""
PairedSpikeLFPDataset — Abstract base class for spike + LFP paired datasets.

Concrete subclasses must:
  1. Implement __len__() and __getitem__() returning the dict described below.
  2. Populate self.session_d_spike_dict and self.session_d_lfp_dict with
     {subject_session: n_channels} mappings so that model tokenizers can be
     built without inspecting the data directly.

__getitem__ contract
--------------------
Each sample must be a dict with the following keys:

    {
        "spikes"          : torch.Tensor of shape (T, N_spike),  dtype=torch.float32
                            Non-negative spike counts, binned at delta-ms resolution.
        "lfp"             : torch.Tensor of shape (T, N_lfp),    dtype=torch.float32
                            Z-scored continuous LFP signal, same time axis as spikes.
        "subject_session" : str
                            Identifier of the form "{Subject}_{session}", e.g.
                            "MonkeyI_20160622_01".  Must match keys in tokenizer
                            session_d_input_dict.
        "segment_filename": str
                            Unique file-level identifier used for saving embeddings.
    }

Both "spikes" and "lfp" share the same T (number of time steps).  Spatial
dimensions (N_spike, N_lfp) may differ across sessions but must be consistent
within a single session.

How to add a real dataset
--------------------------
Subclass PairedSpikeLFPDataset and implement the three required methods.
The existing MakinRTDataset / FlintCODataset preprocessing pipelines in
makin_dataset.py / flint_dataset.py already store processed LFP as
data["lfp"] (shape T × N_lfp) and cursor velocity data["kinem"].
To obtain paired spike counts you need to additionally:
  - Load sorted unit spike trains from the raw .nwb / .mat files.
  - Bin them in delta-ms (e.g. 10 ms) non-overlapping windows.
  - Discard units with mean firing rate < 1 Hz (paper A.3.1).
  - Align the binned spike matrix to the same time grid as the LFP.
  - Z-score is NOT applied to spike counts (they are raw integers).

Example skeleton
----------------
    class MakinPairedDataset(PairedSpikeLFPDataset):
        def __init__(self, lfp_metadata_path, spike_segment_dir, ...):
            # Load metadata, build session dicts, etc.
            ...

        def __len__(self):
            return len(self.metadata)

        def __getitem__(self, idx):
            row = self.metadata.iloc[idx]
            lfp_data = torch.load(row.path)
            spike_data = torch.load(os.path.join(self.spike_dir, row.segment_filename))
            return {
                "spikes"          : spike_data["spikes"].float(),
                "lfp"             : lfp_data["lfp"].float(),
                "subject_session" : row.subject_session,
                "segment_filename": row.segment_filename,
            }
"""

import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class PairedSpikeLFPDataset(Dataset, ABC):
    """
    Abstract base for datasets that provide paired (spike, LFP) segments.

    Attributes
    ----------
    session_d_spike_dict : dict[str, int]
        {subject_session: n_spike_channels}.  Must be set by subclasses
        before the tokenizer is built.
    session_d_lfp_dict : dict[str, int]
        {subject_session: n_lfp_channels}.  Must be set by subclasses.
    """

    session_d_spike_dict: Dict[str, int] = {}
    session_d_lfp_dict: Dict[str, int] = {}

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of paired segments."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """
        Return a single paired sample.

        Must contain keys: "spikes", "lfp", "subject_session", "segment_filename".
        See module docstring for full specification.
        """
        ...

    def get_session_d_spike_dict(self) -> Dict[str, int]:
        """Convenience accessor for building the teacher tokenizer."""
        return self.session_d_spike_dict

    def get_session_d_lfp_dict(self) -> Dict[str, int]:
        """Convenience accessor for building the student tokenizer."""
        return self.session_d_lfp_dict


class MakinPairedDataset(PairedSpikeLFPDataset):
    """
    Paired 5-second MakinRT segments.

    LFP paths and train/val/test splits come from the existing LFP metadata.
    Spike tensors are loaded from spike_segment_dir using the same
    segment_filename, so each sample shares one time axis.
    """

    def __init__(
        self,
        lfp_metadata_path,
        spike_segment_dir,
        session_info_path,
        split="train",
        sessions=None,
        exclude_sessions=None,
    ):
        metadata = pd.read_csv(lfp_metadata_path)
        if split is not None:
            metadata = metadata[metadata["split"] == split]
        if sessions:
            metadata = metadata[metadata["subject_session"].isin(list(sessions))]
        if exclude_sessions:
            metadata = metadata[~metadata["subject_session"].isin(list(exclude_sessions))]
        if metadata.empty:
            raise ValueError(
                "MakinPairedDataset filter removed every row. "
                f"split={split}, sessions={sessions}, exclude_sessions={exclude_sessions}"
            )

        self.metadata = metadata.reset_index(drop=True)
        self.spike_segment_dir = spike_segment_dir
        self.subject_sessions = self.metadata["subject_session"].tolist()

        info = pd.read_csv(session_info_path)
        n_spike = dict(zip(info["subject_session"], info["n_spike"].astype(int)))
        present = sorted(self.metadata["subject_session"].unique())
        self.session_d_spike_dict = {ss: int(n_spike[ss]) for ss in present}
        self.session_d_lfp_dict = {
            ss: int(self.metadata.loc[self.metadata["subject_session"] == ss, "d_lfp"].iloc[0])
            for ss in present
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        lfp_data = torch.load(row.path, weights_only=False)
        spike_path = os.path.join(self.spike_segment_dir, row.segment_filename)
        spike_data = torch.load(spike_path, weights_only=False)
        return {
            "spikes": spike_data["spikes"].float(),
            "lfp": lfp_data["lfp"].float(),
            "subject_session": row.subject_session,
            "segment_filename": row.segment_filename,
        }


class MakinLFPDataset(Dataset):
    """LFP-only 5-second segments for the undistilled MS-LFP baseline."""

    def __init__(self, lfp_metadata_path, split="train", sessions=None):
        metadata = pd.read_csv(lfp_metadata_path)
        if split is not None:
            metadata = metadata[metadata["split"] == split]
        if sessions:
            metadata = metadata[metadata["subject_session"].isin(list(sessions))]
        if metadata.empty:
            raise ValueError("MakinLFPDataset filter removed every row.")
        self.metadata = metadata.reset_index(drop=True)
        self.subject_sessions = self.metadata["subject_session"].tolist()
        present = sorted(self.metadata["subject_session"].unique())
        self.session_d_lfp_dict = {
            ss: int(self.metadata.loc[self.metadata["subject_session"] == ss, "d_lfp"].iloc[0])
            for ss in present
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        data = torch.load(row.path, weights_only=False)
        return {
            "lfp": data["lfp"].float(),
            "subject_session": row.subject_session,
            "segment_filename": row.segment_filename,
        }


def collate_lfp_fn(batch):
    lfp = torch.stack([item["lfp"] for item in batch], dim=0)
    return lfp, [item["subject_session"] for item in batch]


class MakinSpikeDataset(PairedSpikeLFPDataset):
    """Spike-only 5-second segments for teacher pretraining."""

    def __init__(
        self,
        spike_metadata_path,
        session_info_path,
        split="train",
        sessions=None,
    ):
        metadata = pd.read_csv(spike_metadata_path)
        if split is not None:
            metadata = metadata[metadata["split"] == split]
        if sessions:
            metadata = metadata[metadata["subject_session"].isin(list(sessions))]
        if metadata.empty:
            raise ValueError(
                "MakinSpikeDataset filter removed every row. "
                f"split={split}, sessions={sessions}"
            )
        self.metadata = metadata.reset_index(drop=True)
        self.subject_sessions = self.metadata["subject_session"].tolist()
        info = pd.read_csv(session_info_path)
        n_spike = dict(zip(info["subject_session"], info["n_spike"].astype(int)))
        present = sorted(self.metadata["subject_session"].unique())
        self.session_d_spike_dict = {ss: int(n_spike[ss]) for ss in present}
        self.session_d_lfp_dict = {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        spike_data = torch.load(row.path, weights_only=False)
        spikes = spike_data["spikes"].float()
        return {
            "spikes": spikes,
            "lfp": torch.zeros(spikes.shape[0], 1),
            "subject_session": row.subject_session,
            "segment_filename": row.segment_filename,
        }


class SessionBatchSampler(Sampler):
    """Yield index batches that stay inside one session, so N_spike matches."""

    def __init__(self, subject_sessions, batch_size, shuffle=True):
        self.groups = defaultdict(list)
        for idx, session in enumerate(subject_sessions):
            self.groups[session].append(idx)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        batches = []
        sessions = list(self.groups)
        if self.shuffle:
            random.shuffle(sessions)
        for session in sessions:
            indices = list(self.groups[session])
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size])
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.groups.values()
        )
