# predict_after_train.py - Qwen2模型推理脚本（仅推理部分）

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import swanlab


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

    # 初始化SwanLab
    if swanlab_available:
        swanlab.init(
            project="Qwen2-fine-tuning",
            experiment_name="prediction",
            description="Qwen2 model prediction using fine-tuned model"
        )

    # 加载预训练的tokenizer和模型
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        "./qwen/Qwen2-1___5B-Instruct/", 
        device_map="auto", 
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 加载微调后的LORA权重
    model = PeftModel.from_pretrained(base_model, "./output/Qwen2")  # 假设LORA权重保存在该路径
    model = model.merge_and_unload()  # 合并LORA权重到基础模型
    model.eval()  # 设置为评估模式
    
    # 读取测试数据集
    test_df = pd.read_json("new_test.jsonl", lines=True)[:10]  # 取前10条做主观评测

    # ====== 推理预测 ===== #
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

    if swanlab_available:
        swanlab.log({"Prediction": test_text_list})
        swanlab.finish()
    
    print("推理完成") 
             
if __name__ == "__main__":
    main()