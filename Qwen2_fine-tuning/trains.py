import json
import pandas as pd
import os
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
import swanlab
from swanlab.integration.transformers import SwanLabCallback


def cleanup_distributed():
    """清理分布式训练环境"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|system|>\n你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型\n<|user|>\n{example['input']}<|assistant|>\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_and_preprocess_data():
    """
    优化的数据加载和预处理函数
    """
    import multiprocessing as mp
    
    print("开始加载和预处理数据...")
    
    # 读取数据集
    train_df = pd.read_json("new_train.jsonl", lines=True)
    print(f"数据集大小: {len(train_df)}")
    
    # 使用多进程进行数据预处理
    train_ds = Dataset.from_pandas(train_df)
    
    # 并行处理数据集，使用更多CPU核心
    num_cores = min(8, mp.cpu_count())  # 限制使用的核心数
    print(f"使用 {num_cores} 个CPU核心进行数据预处理")
    
    train_dataset = train_ds.map(
        process_func,
        batched=False,
        num_proc=num_cores,  # 使用多进程
        remove_columns=train_ds.column_names,
        load_from_cache_file=False,  # 不使用缓存，确保每次都处理
        desc="预处理数据"
    )
    
    print("数据预处理完成")
    return train_dataset


def main():
    # 检查GPU数量
    n_gpu = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # 检查SwanLab登录状态
    try:
        # 如果未登录，会提示用户登录
        swanlab.login()
        print("SwanLab已登录")
        swanlab_available = True
    except:
        print("SwanLab未登录或配置错误，跳过SwanLab日志记录")
        print("如需使用SwanLab记录训练过程，请访问 https://swanlab.cn 获取API密钥并运行 'swanlab login' 命令")
        swanlab_available = False

    print(f"检测到 {n_gpu} 个GPU")
    
    # 加载预训练的tokenizer和模型
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
    
    # 根据是否为分布式训练环境来设置device_map
    if local_rank != -1:
        # 分布式训练环境下，不使用device_map='auto'
        model = AutoModelForCausalLM.from_pretrained(
            "./qwen/Qwen2-1___5B-Instruct/", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        # 单GPU训练环境下，可以使用device_map='auto'
        model = AutoModelForCausalLM.from_pretrained(
            "./qwen/Qwen2-1___5B-Instruct/", 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    # 使用优化的数据加载函数
    train_dataset = load_and_preprocess_data()

    # 设置LORA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1,  # Dropout 比例
    )

    model = get_peft_model(model, config)

    # 根据GPU数量调整训练参数
    per_device_batch_size = 4
    if n_gpu > 1:
        # 在多GPU情况下可以增加每GPU的批处理大小
        per_device_batch_size = 4  # 每个GPU处理8个样本
        gradient_accumulation_steps = 2  # 减少梯度累积步数以保持总有效批大小
    else:
        gradient_accumulation_steps = 4

    # 训练参数
    args = TrainingArguments(
        output_dir="./output/Qwen2",
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        # 多GPU优化
        dataloader_num_workers=4 if n_gpu > 1 else 2,
        dataloader_pin_memory=True,
        # 启用DDP
        ddp_find_unused_parameters=False,
        # 如果有多个GPU，启用DDP
        local_rank=local_rank,
        # 优化数据加载
        dataloader_drop_last=True,
        remove_unused_columns=False,  # 保持列不被自动删除
    )

    # SwanLab回调
    swanlab_callback = SwanLabCallback(
        project="Qwen2-fintune",
        experiment_name="Qwen2-1.5B-Instruct",
        description="使用通义千问Qwen2-1.5B-Instruct模型在zh_cls_fudan-news数据集上微调。",
        config={
            "model": "qwen/Qwen2-1.5B-Instruct",
            "dataset": "huangjintao/zh_cls_fudan-news",
            "n_gpu": n_gpu,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )

    try:
        # 开始训练
        trainer.train()

        # 保存最终模型
        trainer.save_model()
        print("模型训练完成并已保存")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
    finally:
        # 清理分布式训练环境
        cleanup_distributed()
             
if __name__ == "__main__":
    main()