import inspect
import sys
from collections import namedtuple
from functools import partial

import torch
import torch.nn.functional as F

BatchItem = namedtuple(
    "BatchItem",
    ["inputs", "subject_sessions", "position_ids", "segment_filenames"],
)

# Named tuple for paired (spike, LFP) batches used in distillation training.
PairedBatchItem = namedtuple(
    "PairedBatchItem",
    [
        "spikes",            # Tensor (B, T, N_spike) or list of Tensors
        "lfp",               # Tensor (B, T, N_lfp)  or list of Tensors
        "subject_sessions",  # list[str] of length B
        "position_ids",      # Tensor (B, T) or list, shared for spike & LFP
        "segment_filenames", # list[str] of length B
    ],
)


def collate_with_metadata_fn(batch):
    inputs, position_ids, subject_sessions = [], [], []
    input_dims, input_seq_lens = [], []
    segment_filenames = []

    for datapoint in batch:
        inputs.append(datapoint["input"])
        subject_sessions.append(datapoint["subject_session"])

        num_steps = datapoint["input"].shape[0]
        position_ids.append(torch.arange(0, num_steps, 1))

        input_dims.append(datapoint["input"].shape[-1])
        input_seq_lens.append(datapoint["input"].shape[0])

        segment_filenames.append(datapoint["segment_filename"])

    # means that all trials have the same number of dimensions, so we can form a tensor
    # otherwise, keep them in a list and tokenizer will handle the rest
    if (torch.tensor(input_dims) == input_dims[0]).all() and (
        torch.tensor(input_seq_lens) == input_seq_lens[0]
    ).all():
        inputs = torch.stack(inputs, dim=0)
        if len(position_ids) > 0:
            position_ids = torch.stack(position_ids, dim=0)

    if len(position_ids) == 0:
        position_ids = None

    batch = BatchItem(
        inputs=inputs,
        position_ids=position_ids,
        subject_sessions=subject_sessions,
        segment_filenames=segment_filenames,
    )
    return batch


def collate_paired_fn(batch):
    """
    Collate function for PairedSpikeLFPDataset batches.

    Each element of `batch` is a dict with keys:
        "spikes"          : Tensor (T, N_spike)
        "lfp"             : Tensor (T, N_lfp)
        "subject_session" : str
        "segment_filename": str

    If all samples share the same (T, N_spike) and (T, N_lfp), they are
    stacked into (B, T, N) tensors.  Otherwise they are left as lists for
    the variable-length path in the tokenizer.
    """
    spikes_list, lfp_list = [], []
    subject_sessions, segment_filenames = [], []
    spike_shapes, lfp_shapes = [], []

    for dp in batch:
        spikes_list.append(dp["spikes"])
        lfp_list.append(dp["lfp"])
        subject_sessions.append(dp["subject_session"])
        segment_filenames.append(dp["segment_filename"])
        spike_shapes.append(dp["spikes"].shape)
        lfp_shapes.append(dp["lfp"].shape)

    # Stack if all shapes match; otherwise keep as list
    if len(set(spike_shapes)) == 1:
        spikes = torch.stack(spikes_list, dim=0)          # (B, T, N_spike)
        position_ids = torch.arange(spikes.shape[1]).unsqueeze(0).expand(
            spikes.shape[0], -1
        )  # (B, T)
    else:
        spikes = spikes_list
        position_ids = [torch.arange(s.shape[0]) for s in spikes_list]

    if len(set(lfp_shapes)) == 1:
        lfp = torch.stack(lfp_list, dim=0)                # (B, T, N_lfp)
    else:
        lfp = lfp_list

    return PairedBatchItem(
        spikes=spikes,
        lfp=lfp,
        subject_sessions=subject_sessions,
        position_ids=position_ids,
        segment_filenames=segment_filenames,
    )


def get_collate_fn(collate_fn_name, **partial_kwargs):

    current_module = sys.modules[__name__]
    funcs = {
        name: obj
        for name, obj in inspect.getmembers(current_module, inspect.isfunction)
    }
    collate_fn = funcs.get(collate_fn_name, None)
    if collate_fn:
        return partial(collate_fn, **partial_kwargs)
    else:
        return None
