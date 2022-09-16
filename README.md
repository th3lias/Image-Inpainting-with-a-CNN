# Im-In

This project includes data preparation, augmentation and model training from start to finish.
The model is a simple CNN with skip connections. See: https://arxiv.org/abs/1608.06993

The model learns to inpaint random grids of variable width and height on images with resolution 100x100.
Default hparams are set but can be modified. It also utilizes parallelization and multiprocessing on the GPU.
