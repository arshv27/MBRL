#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.1 seed=1 exp_name=finger-turn-hard-cmixup-0.1 &
CUDA_VISIBLE_DEVICES=0 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.1 seed=2 exp_name=finger-turn-hard-cmixup-0.1 &
CUDA_VISIBLE_DEVICES=0 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.1 seed=3 exp_name=finger-turn-hard-cmixup-0.1 &
CUDA_VISIBLE_DEVICES=0 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.1 seed=4 exp_name=finger-turn-hard-cmixup-0.1 &

CUDA_VISIBLE_DEVICES=1 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.2 seed=1 exp_name=finger-turn-hard-cmixup-0.2 &
CUDA_VISIBLE_DEVICES=1 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.2 seed=2 exp_name=finger-turn-hard-cmixup-0.2 &
CUDA_VISIBLE_DEVICES=1 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.2 seed=3 exp_name=finger-turn-hard-cmixup-0.2 &
CUDA_VISIBLE_DEVICES=1 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.2 seed=4 exp_name=finger-turn-hard-cmixup-0.2 &

CUDA_VISIBLE_DEVICES=2 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.3 seed=1 exp_name=finger-turn-hard-cmixup-0.3 &
CUDA_VISIBLE_DEVICES=2 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.3 seed=2 exp_name=finger-turn-hard-cmixup-0.3 &
CUDA_VISIBLE_DEVICES=2 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.3 seed=3 exp_name=finger-turn-hard-cmixup-0.3 &
CUDA_VISIBLE_DEVICES=2 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.3 seed=4 exp_name=finger-turn-hard-cmixup-0.3 &

CUDA_VISIBLE_DEVICES=3 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.4 seed=1 exp_name=finger-turn-hard-cmixup-0.4 &
CUDA_VISIBLE_DEVICES=3 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.4 seed=2 exp_name=finger-turn-hard-cmixup-0.4 &
CUDA_VISIBLE_DEVICES=3 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.4 seed=3 exp_name=finger-turn-hard-cmixup-0.4 &
CUDA_VISIBLE_DEVICES=3 python3 src/train.py task=finger-turn-hard cmixup_alpha=0.4 seed=4 exp_name=finger-turn-hard-cmixup-0.4
