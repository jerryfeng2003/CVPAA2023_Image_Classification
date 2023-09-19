#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python main_dnd.py --cfg "./configs/WR2021.yml" --exp_name "WR2021_swinv2_b"
