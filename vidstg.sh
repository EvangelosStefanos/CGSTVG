#!/bin/bash

## TRAINING
python3 -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/train_net.py \
    --config-file "experiments/vidstg.yaml" \
    INPUT.RESOLUTION 224 \
    OUTPUT_DIR output/vidstg \
    TENSORBOARD_DIR output/vidstg \
    | tee tee_log_train.txt
