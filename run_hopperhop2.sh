#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 src/train.py task=hopper-hop cmixup_alpha=0.5 seed=1 exp_name=hopper-hop-cmixup-0.5 &
CUDA_VISIBLE_DEVICES=0 python3 src/train.py task=hopper-hop cmixup_alpha=0.5 seed=2 exp_name=hopper-hop-cmixup-0.5 &
CUDA_VISIBLE_DEVICES=0 python3 src/train.py task=hopper-hop cmixup_alpha=0.5 seed=3 exp_name=hopper-hop-cmixup-0.5 &
CUDA_VISIBLE_DEVICES=0 python3 src/train.py task=hopper-hop cmixup_alpha=0.5 seed=4 exp_name=hopper-hop-cmixup-0.5 &

CUDA_VISIBLE_DEVICES=1 python3 src/train.py task=hopper-hop cmixup_alpha=0.6 seed=1 exp_name=hopper-hop-cmixup-0.6 &
CUDA_VISIBLE_DEVICES=1 python3 src/train.py task=hopper-hop cmixup_alpha=0.6 seed=2 exp_name=hopper-hop-cmixup-0.6 &
CUDA_VISIBLE_DEVICES=1 python3 src/train.py task=hopper-hop cmixup_alpha=0.6 seed=3 exp_name=hopper-hop-cmixup-0.6 &
CUDA_VISIBLE_DEVICES=1 python3 src/train.py task=hopper-hop cmixup_alpha=0.6 seed=4 exp_name=hopper-hop-cmixup-0.6 &

CUDA_VISIBLE_DEVICES=2 python3 src/train.py task=hopper-hop cmixup_alpha=0.7 seed=1 exp_name=hopper-hop-cmixup-0.7 &
CUDA_VISIBLE_DEVICES=2 python3 src/train.py task=hopper-hop cmixup_alpha=0.7 seed=2 exp_name=hopper-hop-cmixup-0.7 &
CUDA_VISIBLE_DEVICES=2 python3 src/train.py task=hopper-hop cmixup_alpha=0.7 seed=3 exp_name=hopper-hop-cmixup-0.7 &
CUDA_VISIBLE_DEVICES=2 python3 src/train.py task=hopper-hop cmixup_alpha=0.7 seed=4 exp_name=hopper-hop-cmixup-0.7 &

CUDA_VISIBLE_DEVICES=3 python3 src/train.py task=hopper-hop cmixup_alpha=0.8 seed=1 exp_name=hopper-hop-cmixup-0.8 &
CUDA_VISIBLE_DEVICES=3 python3 src/train.py task=hopper-hop cmixup_alpha=0.8 seed=2 exp_name=hopper-hop-cmixup-0.8 &
CUDA_VISIBLE_DEVICES=3 python3 src/train.py task=hopper-hop cmixup_alpha=0.8 seed=3 exp_name=hopper-hop-cmixup-0.8 &
CUDA_VISIBLE_DEVICES=3 python3 src/train.py task=hopper-hop cmixup_alpha=0.8 seed=4 exp_name=hopper-hop-cmixup-0.8
