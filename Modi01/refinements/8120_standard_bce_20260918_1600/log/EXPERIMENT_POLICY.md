# Refinement experiment policy

Three authorized Modi01 candidates; standard STGC BCE-only refinement, 1000 steps.

Final 30000-step EMA field frozen; UNet only; separate output checkpoints.

Official legacy KITTI-360 8120–8170; time=(frame_id-8120)/50; val/test overlap: post-hoc development.

train: 47 frames; IDs: [8120, 8121, 8122, 8123, 8124, 8125, 8126, 8127, 8128, 8129, 8131, 8132, 8133, 8134, 8135, 8136, 8137, 8138, 8139, 8141, 8142, 8143, 8144, 8145, 8146, 8147, 8148, 8149, 8151, 8152, 8153, 8154, 8155, 8156, 8157, 8158, 8159, 8161, 8162, 8163, 8164, 8165, 8166, 8167, 8168, 8169, 8170]
Manifest: /home/zijiewu/Code/STGC-NeRF-private/data/kitti360/transforms_8120_train.json

val: 4 frames; IDs: [8130, 8140, 8150, 8160]
Manifest: /home/zijiewu/Code/STGC-NeRF-private/data/kitti360/transforms_8120_val.json

test: 4 frames; IDs: [8130, 8140, 8150, 8160]
Manifest: /home/zijiewu/Code/STGC-NeRF-private/data/kitti360/transforms_8120_test.json

No development/test training, early stopping or checkpoint selection. No model promotion.
No file digests are computed. Full normalization: ../data_protocol.json
