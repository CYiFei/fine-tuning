# train.py - Qwen2模型微调脚本

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
        f"<|system|>\n你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型<|endoftext|>\n<|user|>\n{example['input']}<|endoftext|>\n<|assistant|>\n",
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


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

    return response


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
    model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    # 读取转换后的数据集
    train_df = pd.read_json("new_train.jsonl", lines=True)[:1000]  # 取前1000条做训练（可选）
    test_df = pd.read_json("new_test.jsonl", lines=True)[:10]  # 取前10条做主观评测

    # 数据预处理
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

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

    # 训练参数
    args = TrainingArguments(
        output_dir="./output/Qwen2",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
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

    # ====== 训练结束后的预测 ===== #
    test_text_list = []
    for index, row in test_df.iterrows():
        instruction = row["instruction"]
        input_value = row["input"]

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"},
        ]

        response = predict(messages, model, tokenizer)
        messages.append({"role": "assistant", "content": f"{response}"})
        result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
        test_text_list.append(swanlab.Text(result_text, caption=response))

    swanlab.log({"Prediction": test_text_list})
    swanlab.finish()


if __name__ == "__main__":
    main()