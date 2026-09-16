"""Named transform-file contracts shared by preprocessing and loading."""

from __future__ import annotations


SPLIT_PROTOCOLS = ("legacy", "v4_disjoint")


def transform_filename(sequence_id: str, split: str, protocol: str) -> str:
    if protocol not in SPLIT_PROTOCOLS:
        raise ValueError(f"unknown split protocol {protocol!r}")
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown dataset split {split!r}")
    namespace = "" if protocol == "legacy" else f"_{protocol}"
    return f"transforms_{sequence_id}{namespace}_{split}.json"
