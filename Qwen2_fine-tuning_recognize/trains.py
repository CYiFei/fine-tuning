import json
import pandas as pd
import os
import torch
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
from swanlab.integration.huggingface import SwanLabCallback
import swanlab
import argparse


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


def process_func(example):
    """
    将数据集进行预处理, 处理成模型可以接受的格式
    """
    MAX_LENGTH = 384 
    input_ids, attention_mask, labels = [], [], []
    system_prompt = """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体"."""
    
    instruction = tokenizer(
        f"<|system|>\n{system_prompt}<|endoftext|><|user|>\n{example['input']}<|endoftext|><|assistant|>\n",
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
    
    # 配置量化选项（如果需要）
    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            quantization_config=bnb_config,
            device_map="auto", 
            trust_remote_code=True
        )
    else:
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


def train_model(model, train_dataset, output_dir, num_gpus=1):
    """
    训练模型
    """
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        # 根据GPU数量调整参数
        dataloader_num_workers=min(4, num_gpus * 2) if num_gpus > 1 else 2,
    )
    
    # 初始化SwanLab回调
    swanlab_callback = SwanLabCallback(
        project="Qwen2-NER-finetune",
        experiment_name="Qwen2-1.5B-Instruct",
        description="使用通义千问Qwen2-1.5B-Instruct模型在NER数据集上微调，实现关键实体识别任务。",
        config={
            "model": model.config._name_or_path,
            "dataset": "qgyd2021/chinese_ner_sft",
            "num_gpus": num_gpus,
        },
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )
    
    # 开始训练
    trainer.train()
    
    return trainer


def main(args):
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
    
    # 预处理训练数据
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
    
    # 获取GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU设备")
    
    # 训练模型
    trainer = train_model(model, train_dataset, args.output_dir, num_gpus)
    
    # 保存最终模型
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练Qwen2 NER模型")
    parser.add_argument("--model_dir", type=str, default="./qwen/Qwen2-1___5B-Instruct", help="模型目录路径")
    parser.add_argument("--train_data_path", type=str, default="ccfbdci.jsonl", help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="./output/Qwen2-NER", help="输出目录")
    parser.add_argument("--quantization", action="store_true", help="是否启用4位量化")
    
    args = parser.parse_args()
    
    main(args)