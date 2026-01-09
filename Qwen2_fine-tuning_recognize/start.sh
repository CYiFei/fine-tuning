# 使用torchrun启动多GPU训练
torchrun --nproc_per_node=2 --master_port=29500 trains.py --model_dir ./models/qwen/Qwen2-1___5B-Instruct --output_dir ./output