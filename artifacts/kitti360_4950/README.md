# KITTI-360 4950 artifacts

This directory contains the completed STGC-NeRF scene 4950 training log and the integrity manifest for the checkpoints used for the reported evaluation.

| Release asset | Size bytes | SHA-256 | Purpose |
|---|---:|---|---|
| `stgc_nerf_ep0639.pth` | 1009062994 | `85d9c8bbe6fd8fa5bbda771eb9586058eda0cc66d3ed498d393c18b057583185` | Full epoch 639 training checkpoint |
| `stgc_nerf_ep0639_refine.pth` | 212320299 | `fda731ae5ac09bde08ec82a812b545a1a58e3eeebe2967b37ea631db0cd700b1` | Final ray-drop refined evaluation checkpoint |
| `log_stgc_nerf.txt` | 86725 | `f543a1b39f53bb16394b5805b5f2c49a9edf636293942e14b5e18e9d47571af8` | Complete training, refinement, and test log |

The checkpoint files are attached to the private GitHub release [`stgc-kitti360-4950-checkpoints-20260914`](https://github.com/hubigwaterdeep/STGC-NeRF-private/releases/tag/stgc-kitti360-4950-checkpoints-20260914).
