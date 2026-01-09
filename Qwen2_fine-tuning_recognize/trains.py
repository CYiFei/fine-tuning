import json
import pandas as pd
import os
import torch
import torch.distributed as dist
from modelscope import snapshot_download, AutoTokenizer
from transformers import (
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.transformers import SwanLabCallback  # 修复弃用警告
import swanlab
import argparse
import multiprocessing as mp


def cleanup_distributed():
    """清理分布式训练环境"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    elif 'LOCAL_RANK' in os.environ:
        # 如果设置了LOCAL_RANK但没有初始化分布式环境，也要确保清除相关环境变量
        os.environ.pop('LOCAL_RANK', None)

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input_text = data["text"]
            entities = data["entities"]
            match_names = ["地点", "人名", "地理实体", "组织"]
            
            entity_sentence = ""
            for entity in entities:
                entity_json = dict(entity)
                entity_text = entity_json["entity_text"]
                entity_names = entity_json["entity_names"]
                for name in entity_names:
                    if name in match_names:
                        entity_label = name
                        break
                
                entity_sentence += f"""{{"entity_text": "{entity_text}", "entity_label": "{entity_label}"}}"""
            
            if entity_sentence == "":
                entity_sentence = "没有找到任何实体"
            
            message = {
                "instruction": """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """,
                "input": f"文本:{input_text}",
                "output": entity_sentence,
            }
            
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example, tokenizer):
    """
    将数据集进行预处理, 处理成模型可以接受的格式
    """
    MAX_LENGTH = 384 
    input_ids, attention_mask, labels = [], [], []
    system_prompt = """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体"."""
    
    instruction = tokenizer(
        f"<|system|>\n{system_prompt}<|user|>\n{example['input']}<|assistant|>\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}



def load_and_preprocess_data(train_df, tokenizer):
    """
    优化的数据加载和预处理函数
    """
    print("开始预处理数据...")
    
    # 使用多进程进行数据预处理
    train_ds = Dataset.from_pandas(train_df)
    
    # 并行处理数据集，使用更多CPU核心
    num_cores = min(8, mp.cpu_count())  # 限制使用的核心数
    print(f"使用 {num_cores} 个CPU核心进行数据预处理")
    
    train_dataset = train_ds.map(
        lambda x: process_func(x, tokenizer),  # 传递tokenizer
        batched=False,
        num_proc=num_cores,  # 使用多进程
        remove_columns=train_ds.column_names,
        load_from_cache_file=False,  # 不使用缓存，确保每次都处理
        desc="预处理数据"
    )
    
    print("数据预处理完成")
    return train_dataset

def prepare_datasets(train_path, test_path=None, split_ratio=0.1, test_sample_size=20):
    """
    准备训练和测试数据集
    """
    total_df = pd.read_json(train_path, lines=True)
    
    if test_path is None:
        # 如果没有提供单独的测试集，从训练集中划分
        train_df = total_df[int(len(total_df) * split_ratio):]  # 取90%的数据做训练集
        test_df = total_df[:int(len(total_df) * split_ratio)].sample(n=test_sample_size)  # 随机取10%的数据中的20条做测试集
    else:
        train_df = pd.read_json(train_path, lines=True)
        test_df = pd.read_json(test_path, lines=True).sample(min(test_sample_size, len(pd.read_json(test_path, lines=True))))
    
    return train_df, test_df


def load_model_and_tokenizer(model_dir, quantization=False):
    """
    加载模型和分词器
    """
    global tokenizer
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    
    # 检查是否在分布式环境中，如果是，则不使用device_map="auto"
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # 配置量化选项（如果需要）
    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        if local_rank != -1:  # 分布式训练环境
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, 
                quantization_config=bnb_config,
                device_map={"": torch.cuda.current_device()},  # 为当前设备指定映射
                trust_remote_code=True
            )
        else:  # 单GPU环境
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, 
                quantization_config=bnb_config,
                device_map="auto", 
                trust_remote_code=True
            )
    else:
        if local_rank != -1:  # 分布式训练环境
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, 
                device_map={"": torch.cuda.current_device()},  # 为当前设备指定映射
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        else:  # 单GPU环境
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, 
                device_map="auto", 
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
    
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    
    return model, tokenizer

def setup_lora(model):
    """
    配置LoRA
    """
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
        lora_alpha=32,  # Lora alpha，具体作用参见 Lora 原理
        lora_dropout=0.1,  # Dropout 比例
    )

    model = get_peft_model(model, config)
    return model


def train_model(model, train_dataset, output_dir, num_gpus=1, tokenizer=None):
    """
    训练模型
    """
    # 检查是否为分布式训练环境
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # 根据GPU数量调整训练参数
    per_device_batch_size = 4  # 降低批处理大小以避免内存溢出
    if num_gpus > 1:
        # 在多GPU情况下可以适当增加每GPU的批处理大小
        per_device_batch_size = 4  # 显著减少每个GPU的批次大小以节省内存
        gradient_accumulation_steps = 2  # 增加梯度累积步数以保持总有效批大小
    else:
        gradient_accumulation_steps = 4  # 单GPU时增加梯度累积步数

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=max(1, per_device_batch_size // 2),
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=1e-4,
        save_strategy="steps",  # 保存策略
        save_total_limit=3,  # 最大保存的模型数量
        warmup_steps=50,  # 预热步数
        weight_decay=0.01,  # 权重衰减
        dataloader_pin_memory=True,  # 禁用内存固定以节省内存
        remove_unused_columns=False,
        label_names=["labels"],  # 指定标签列名称
        # 多GPU优化
        dataloader_num_workers=4 if num_gpus > 1 else 2,  # 减少数据加载工作进程数
        # 启用DDP
        ddp_find_unused_parameters=False,
        # 如果有多个GPU，启用DDP
        local_rank=local_rank,
        # 优化数据加载
        dataloader_drop_last=True,
        # 性能优化
        fp16=torch.cuda.is_available(),  # 启用混合精度训练（如果可用）
        dataloader_prefetch_factor=2 if num_gpus > 1 else 1,  # 降低预取因子
        # 内存优化
        gradient_checkpointing=True,  # 启用梯度检查点以节省内存
    )
    
    # 创建数据整理器 - 确保tokenizer被正确传递
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    
    # 初始化SwanLab回调
    swanlab_callback = SwanLabCallback(
        project="Qwen2-NER-finetune",
        experiment_name="Qwen2-1.5B-Instruct",
        description="使用通义千问Qwen2-1.5B-Instruct模型在NER数据集上微调，实现关键实体识别任务。",
        config={
            "model": model.config._name_or_path,
            "dataset": "qgyd2021/chinese_ner_sft",
            "num_gpus": num_gpus,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "total_batch_size": per_device_batch_size * num_gpus * gradient_accumulation_steps,
        },
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
    )
    
    # 开始训练
    trainer.train()
    
    return trainer

def main(args):
    # 检查模型路径是否存在
    if not os.path.exists(args.model_dir):
        print(f"错误：模型路径不存在: {args.model_dir}")
        print("请先下载模型，例如运行: python download.py")
        return
    
    # 检查数据文件是否存在
    if not os.path.exists(args.train_data_path):
        print(f"错误：训练数据文件不存在: {args.train_data_path}")
        return
    
    # 检查GPU数量
    n_gpu = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    print(f"检测到 {n_gpu} 个GPU设备")
    print(f"本地排名 (LOCAL_RANK): {local_rank}")
    
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
    
    # 确保数据集已准备好
    train_jsonl_new_path = "ccf_train.jsonl"
    if not os.path.exists(train_jsonl_new_path):
        dataset_jsonl_transfer(args.train_data_path, train_jsonl_new_path)
    
    # 准备数据集
    train_df, test_df = prepare_datasets(train_jsonl_new_path)
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.quantization)
    
    # 设置LoRA
    model = setup_lora(model)
    
    # 预处理训练数据 - 使用优化的多进程函数，传递tokenizer
    train_dataset = load_and_preprocess_data(train_df, tokenizer)
    
    # 训练模型
    trainer = train_model(model, train_dataset, args.output_dir, n_gpu, tokenizer)
    
    # 保存最终模型
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"模型已保存到: {args.output_dir}")
    
    # 清理分布式训练环境
    cleanup_distributed()
    
    # # 确保SwanLab正确关闭
    # try:
    #     swanlab.finish()
    #     print("SwanLab已关闭")
    # except:
    #     print("关闭SwanLab时出错")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练Qwen2 NER模型")
    parser.add_argument("--model_dir", type=str, default="./models/qwen/Qwen2-1___5B-Instruct", help="模型目录路径")
    parser.add_argument("--train_data_path", type=str, default="ccfbdci.jsonl", help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="./output/Qwen2-NER", help="输出目录")
    parser.add_argument("--quantization", action="store_true", help="是否启用4位量化")
    
    args = parser.parse_args()
    
    main(args)