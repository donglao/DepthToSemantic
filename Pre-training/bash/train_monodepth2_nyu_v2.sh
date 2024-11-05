#!/bin/bash

python external_src/monodepth2/train_nyu_v2.py \
--model_name mono_352x448 \
--log_dir trained_monodepth2 \
--num_layers 18 \
--png \
--height 352 \
--width 448 \
--disparity_smoothness 1.0 \
--min_depth 0.1 \
--max_depth 10.0 \
--weights_init pretrained \
--pose_model_type separate_resnet \
--num_workers 10 \
