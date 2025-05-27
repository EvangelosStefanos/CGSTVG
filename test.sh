#!/bin/bash

## EVALUATION
python3 -m torch.distributed.launch \
    --nproc_per_node=2 \
    scripts/test_net.py \
    --config-file "experiments/vidstg.yaml" \
    MODEL.WEIGHT "output/model_final.pth" \
    | tee tee_log_eval.txt
