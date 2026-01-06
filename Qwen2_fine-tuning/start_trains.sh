#!/bin/bash
# 使用分布式训练启动脚本
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=29501 trains.py