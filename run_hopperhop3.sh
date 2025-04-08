#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python3 src/train.py task=hopper-hop cmixup_alpha=0.9 seed=1 exp_name=hopper-hop-cmixup-0.9 &
CUDA_VISIBLE_DEVICES=2 python3 src/train.py task=hopper-hop cmixup_alpha=0.9 seed=2 exp_name=hopper-hop-cmixup-0.9 &
CUDA_VISIBLE_DEVICES=2 python3 src/train.py task=hopper-hop cmixup_alpha=0.9 seed=3 exp_name=hopper-hop-cmixup-0.9 &
CUDA_VISIBLE_DEVICES=2 python3 src/train.py task=hopper-hop cmixup_alpha=0.9 seed=4 exp_name=hopper-hop-cmixup-0.9 &

CUDA_VISIBLE_DEVICES=3 python3 src/train.py task=hopper-hop cmixup_alpha=1.0 seed=1 exp_name=hopper-hop-cmixup-1.0 &
CUDA_VISIBLE_DEVICES=3 python3 src/train.py task=hopper-hop cmixup_alpha=1.0 seed=2 exp_name=hopper-hop-cmixup-1.0 &
CUDA_VISIBLE_DEVICES=3 python3 src/train.py task=hopper-hop cmixup_alpha=1.0 seed=3 exp_name=hopper-hop-cmixup-1.0 &
CUDA_VISIBLE_DEVICES=3 python3 src/train.py task=hopper-hop cmixup_alpha=1.0 seed=4 exp_name=hopper-hop-cmixup-1.0
