# train_int4.py - Qwen2模型微调脚本

import json
import pandas as pd
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.huggingface import SwanLabCallback
import swanlab

def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|system|>\n你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型\n<|user|>\n{example['input']}",
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


def main():
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
        
    # 加载预训练的tokenizer和模型
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
    
    # 使用4bit量化来大幅减少显存占用
    from transformers import BitsAndBytesConfig
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16  # 使用float16而非bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "./qwen/Qwen2-1___5B-Instruct/", 
        device_map="auto", 
        torch_dtype=torch.float16,
        quantization_config=nf4_config,  # 启用4bit量化
        trust_remote_code=True,
        use_cache=True  # 在训练时启用缓存
    )
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    # 读取转换后的数据集
    train_df = pd.read_json("new_train.jsonl", lines=True)[:4000]  # 取前1000条做训练（可选）
    test_df = pd.read_json("new_test.jsonl", lines=True)[:10]  # 取前10条做主观评测

    # 数据预处理 - 使用更节省内存的方式
    train_ds = Dataset.from_pandas(train_df)
    
    # 为了控制内存使用，对数据集进行懒加载处理
    train_dataset = train_ds.map(
        process_func,
        batched=False,  # 避免批量处理占用过多内存
        remove_columns=train_ds.column_names,
        keep_in_memory=False,  # 不强制将数据保持在内存中
        num_proc=1  # 单进程处理，减少内存并发
    )
    
    # 清理内存
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 设置LORA - 减少参数以节省显存
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",  # 只保留关键模块，减少显存使用
        ],
        inference_mode=False,  # 训练模式
        r=4,  # Lora 秩 - 减小以节省显存
        lora_alpha=8,  # Lora alaph - 按比例调整
        lora_dropout=0.1,  # Dropout 比例
    )

    model = get_peft_model(model, config)

    # 训练参数 - 优化显存使用
    args = TrainingArguments(
        output_dir="./output/Qwen2",
        per_device_train_batch_size=4,  # 减小批次大小
        gradient_accumulation_steps=8,  # 增加梯度累积步数以保持有效批次大小
        logging_steps=10,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_pin_memory=False,  # 禁用pin memory节省内存
        remove_unused_columns=False,  # 避免不必要的列移除
        dataloader_num_workers=0,  # 设置为0以减少内存使用
        group_by_length=True,  # 根据长度分组，可以减少padding，节省显存
        dataloader_drop_last=True,  # 丢弃最后一个不完整的批次
        fp16=True,  # 启用混合精度训练
    )

    # SwanLab回调
    swanlab_callback = SwanLabCallback(
        project="Qwen2-fintune",
        experiment_name="Qwen2-1.5B-Instruct",
        description="使用通义千问Qwen2-1.5B-Instruct模型在zh_cls_fudan-news数据集上微调。",
        config={
            "model": "qwen/Qwen2-1.5B-Instruct",
            "dataset": "huangjintao/zh_cls_fudan-news",
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

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model()
    print("模型已保存到 ./output/Qwen2")
    
    # 最后再次清理显存
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()