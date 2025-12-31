from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BitsAndBytesConfig
import gc  # 导入垃圾回收模块


def process_func(example):
    MAX_LENGTH = 128    # 进一步减少最大长度
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def clear_memory():
    """清理显存"""
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 使用4bit量化来大幅减少显存占用
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16  # 使用float16而非bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        './autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct', 
        device_map="auto", 
        torch_dtype=torch.float16,
        quantization_config=nf4_config,  # 启用4bit量化
        trust_remote_code=True,
        use_cache=False  # 禁用缓存
    )
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    tokenizer = AutoTokenizer.from_pretrained('./autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct', use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # # 限制数据集大小以进一步减少显存使用
    # df = pd.read_json('huanhuan.json')
    # # 只使用部分数据
    # df = df.head(min(1000, len(df)))  # 限制数据集大小为1000条记录
    # ds = Dataset.from_pandas(df)
    
    # 加载完整数据集，但控制内存使用
    df = pd.read_json('huanhuan.json')
    ds = Dataset.from_pandas(df)
    
    # 为了控制内存使用，我们可以对数据集进行懒加载处理
    # 不预先处理整个数据集，而是在训练时动态处理
    tokenized_id = ds.map(
        process_func,
        batched=False,  # 避免批量处理占用过多内存
        remove_columns=ds.column_names,
        # 设置较低的缓存大小
        keep_in_memory=False,  # 不强制将数据保持在内存中
        num_proc=1  # 单进程处理，减少内存并发
    )
    
    # 在数据预处理前清理显存
    clear_memory()
    
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj"],  # 只保留最关键的模块，进一步减少显存使用
        inference_mode=False, # 训练模式
        r=2, # 极大降低LoRA秩
        lora_alpha=4, # 相应降低alpha值
        lora_dropout=0.1# Dropout 比例
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters() # 打印总训练参数

    args = TrainingArguments(
        output_dir="./output/llama3_1_instruct_lora",
        per_device_train_batch_size=1,  # 最小批次大小
        gradient_accumulation_steps=32,  # 进一步增加梯度累积步数
        logging_steps=10,
        num_train_epochs=1,  # 减少训练轮数
        save_steps=100,  # 减少保存频率
        learning_rate=1e-4,  # 使用原始学习率
        save_on_each_node=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,  # 禁用pin memory节省内存
        remove_unused_columns=False,  # 避免不必要的列移除
        warmup_steps=5,  # 减少预热步数
        fp16=True,  # 启用混合精度训练
        report_to=None,  # 不报告到外部服务
        dataloader_num_workers=0,  # 设置为0以减少内存使用
        group_by_length=True,  # 根据长度分组，可以减少padding，节省显存
        dataloader_drop_last=True  # 丢弃最后一个不完整的批次
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train() # 开始训练 
    # 在训练参数中设置了自动保存策略此处并不需要手动保存。