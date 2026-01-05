# test_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig
import torch
import os

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

# 加载基础模型和LoRA适配器
base_model_path = './autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

# 加载量化配置（与训练时保持一致）
from transformers import BitsAndBytesConfig
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=nf4_config,
    trust_remote_code=True,
    use_cache=True
)

# 加载LoRA权重
try:
    model = PeftModel.from_pretrained(model, lora_path)
    print("成功加载LoRA模型")
except Exception as e:
    print(f"加载LoRA模型失败: {e}")
    print("请确保训练已完成并且检查点存在")
    exit(1)

# 设置为评估模式
model.eval()

def generate_response(prompt):
    """
    生成模型响应
    """
    # 构建输入格式，与训练时保持一致
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # 移动到GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成响应
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码完整输出
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 方法1: 查找助手回复部分
    if "assistant<|end_header_id|>" in full_response:
        assistant_start = full_response.find("assistant<|end_header_id|>") + len("assistant<|end_header_id|>")
        response = full_response[assistant_start:].split("<|eot_id|>")[0].strip()
    else:
        # 方法2: 从原始提示后开始截取
        prompt_end = full_response.find(prompt) + len(prompt)
        response = full_response[prompt_end:].split("<|eot_id|>")[0].strip()
    
    # 如果解析出的响应为空或太短，返回完整输出的后半部分
    if not response or len(response.strip()) < 5:
        # 简单截取完整响应的后部分
        parts = full_response.split("assistant<|end_header_id|>")
        if len(parts) > 1:
            response = parts[1].split("<|eot_id|>")[0].strip()
        else:
            # 如果以上方法都失败，返回整个输出减去输入部分
            response = full_response.replace(formatted_prompt, "").strip()
    
    return response

# 测试示例
test_prompts = [
    "请介绍一下你自己",
    "今天天气怎么样？",
    "作为甄嬛，你会如何应对后宫的挑战？",
    "描述一下紫禁城的生活"
]

print("开始测试模型效果：")
print("="*50)

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