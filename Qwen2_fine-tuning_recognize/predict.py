import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import json


def load_model_with_adapter(base_model_path, adapter_path, quantization=False):
    """
    加载基础模型和LoRA适配器
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
    
    if quantization:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA适配器
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # 加载LoRA适配器
        model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer


def predict_ner(text, model, tokenizer):
    """
    对输入文本进行NER预测
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 构建对话模板
    system_prompt = """你是一个文本实体识别领域的专家，你需要从给定的句子中提取 地点; 人名; 地理实体; 组织 实体. 以 json 格式输出, 如 {"entity_text": "南京", "entity_label": "地理实体"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体"."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"文本:{text}"},
    ]
    
    # 应用对话模板
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # 生成预测
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # 对于NER任务，通常不需要采样
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return generated_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Qwen2 NER模型预测")
    parser.add_argument("--base_model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--adapter_path", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--input_text", type=str, required=True, help="待识别的输入文本")
    parser.add_argument("--quantization", action="store_true", help="是否启用4位量化")
    
    args = parser.parse_args()
    
    # 加载模型
    print("正在加载模型...")
    model, tokenizer = load_model_with_adapter(args.base_model_path, args.adapter_path, args.quantization)
    
    # 执行预测
    print(f"正在对文本进行NER预测: {args.input_text}")
    result = predict_ner(args.input_text, model, tokenizer)
    
    print("\n预测结果:")
    print(result)
    
    # 尝试解析JSON格式的结果
    try:
        # 如果结果是多个JSON对象，按行分割解析
        lines = result.split('\n')
        entities = []
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    entity = json.loads(line)
                    entities.append(entity)
                except json.JSONDecodeError:
                    continue
        
        if entities:
            print("\n解析出的实体:")
            for entity in entities:
                print(f"- 文本: {entity.get('entity_text', '')}, 类型: {entity.get('entity_label', '')}")
        elif result.strip() == "没有找到任何实体":
            print("\n未找到任何实体")
        else:
            print("\n无法解析结果，请检查输出格式")
    except Exception as e:
        print(f"\n解析结果时出错: {e}")


if __name__ == "__main__":
    main()