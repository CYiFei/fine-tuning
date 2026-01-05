# test_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

def test_model():
    # 基础模型路径
    base_model_path = './autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'
    
    # 检查可用的检查点
    output_dir = './output/llama3_1_instruct_lora'
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    if checkpoints:
        # 获取最新的检查点
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
        lora_path = os.path.join(output_dir, latest_checkpoint)
        print(f"使用最新的检查点: {lora_path}")
    else:
        # 如果没有检查点，使用最终输出目录
        lora_path = output_dir
        print(f"使用最终输出目录: {lora_path}")
    
    print("正在加载基础模型...")
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # 与训练时保持一致
        trust_remote_code=True
    )
    
    print("正在加载LoRA适配器...")
    # 加载LoRA权重
    try:
        model = PeftModel.from_pretrained(model, lora_path)
        print("成功加载LoRA模型")
    except Exception as e:
        print(f"加载LoRA模型失败: {e}")
        print(f"请确保路径 '{lora_path}' 包含adapter_config.json和adapter_model.safetensors文件")
        return

    print("正在加载分词器...")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 设置为评估模式
    model.eval()
    
    print("模型加载完成！开始测试：")
    print("="*50)
    
    def generate_response(prompt, max_new_tokens=256):
        """
        生成模型响应
        """
        # 构建输入格式，与训练时保持一致
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # 关键修复：将输入移动到模型所在的设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成响应
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # 添加一些参数来确保生成
                min_new_tokens=10  # 确保生成至少10个新token
            )
        
        # 解码完整输出
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Debug - Full response: {repr(full_response)}")  # 调试信息
        
        # 尝试多种方法提取助手的回复部分
        response = ""
        
        # 方法1: 查找assistant标记
        if "assistant<|end_header_id|>" in full_response:
            assistant_start = full_response.find("assistant<|end_header_id|>") + len("assistant<|end_header_id|>")
            response = full_response[assistant_start:].split("<|eot_id|>")[0].strip()
        
        # 方法2: 如果方法1失败，尝试从原始格式中提取
        if not response.strip():
            prompt_end_pos = full_response.find(prompt)
            if prompt_end_pos != -1:
                prompt_end = prompt_end_pos + len(prompt)
                response = full_response[prompt_end:].split("<|eot_id|>")[0].strip()
        
        # 方法3: 如果仍然没有响应，返回整个生成部分
        if not response.strip():
            # 找到用户输入后的内容
            user_pos = full_response.find("user<|end_header_id|>")
            if user_pos != -1:
                user_end = user_pos + len("user<|end_header_id|>")
                response = full_response[user_end:].split("<|eot_id|>")[0].strip()
        
        # 方法4: 如果还是没有，返回去除格式后的剩余部分
        if not response.strip():
            # 去除格式部分，返回生成的内容
            for prefix in [formatted_prompt, prompt]:
                if full_response.startswith(prefix):
                    response = full_response[len(prefix):].strip()
                    break
        
        print(f"Debug - Extracted response: {repr(response)}")  # 调试信息
        
        return response

    # 测试示例
    test_prompts = [
        "请介绍一下你自己",
        "今天天气怎么样？",
        "作为甄嬛，你会如何应对后宫的挑战？",
        "描述一下紫禁城的生活",
        "皇帝驾到，臣妾给你请安了"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n测试 {i}: {prompt}")
        response = generate_response(prompt)
        print(f"模型回复: {response}")
        print("-" * 30)
    
    # 交互式测试
    print("\n进入交互模式（输入 'quit' 退出）：")
    while True:
        user_input = input("\n您: ")
        if user_input.lower() == 'quit':
            break
        response = generate_response(user_input)
        print(f"模型: {response}")

if __name__ == "__main__":
    test_model()