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

## EVALUATION
# python3 -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     scripts/test_net.py \
#     --config-file "experiments/vidstg.yaml" \
#     INPUT.RESOLUTION 100 \
#     MODEL.WEIGHT "output/vidstg/model_final.pth" \
#     OUTPUT_DIR output/vidstg \
#     | tee tee_log_eval.txt
