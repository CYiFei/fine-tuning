# inference_int4.py - Qwen2模型推理脚本

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import swanlab

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    # 添加attention_mask以解决警告并优化生成
    attention_mask = model_inputs.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(model_inputs["input_ids"])
    
    # 进一步减少生成的最大token数，添加更多参数以节省显存
    generated_ids = model.generate(
        model_inputs.input_ids, 
        attention_mask=attention_mask,  # 显式传递attention_mask
        max_new_tokens=16,  # 进一步减少生成token数
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # 添加以下参数进一步节省显存
        use_cache=True,  # 在推理时启用缓存
        # 进一步优化显存使用
        num_beams=1,  # 使用贪心解码而不是beam search
        repetition_penalty=1.1,  # 控制重复
        # 添加长度惩罚以避免生成过长序列
        max_length=min(model_inputs.input_ids.shape[1] + 16, 512),  # 限制总长度
        # 减少beam数量和batch大小
        num_return_sequences=1,
    )
    
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

    # 清理显存
    del model_inputs, generated_ids, attention_mask
    torch.cuda.empty_cache()
    
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
    base_model_path = "./qwen/Qwen2-1___5B-Instruct/"
    adapter_path = "./output/Qwen2"  # 微调后的模型路径
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
    
    # 使用4bit量化来大幅减少显存占用
    from transformers import BitsAndBytesConfig
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16  # 使用float16而非bfloat16
    )
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        device_map="auto",  # 使用auto以自动分配到GPU和CPU
        torch_dtype=torch.float16,
        quantization_config=nf4_config,  # 启用4bit量化
        trust_remote_code=True,
        use_cache=True  # 在推理时启用缓存
    )
    
    # 加载微调后的适配器
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to("cuda")
    model.eval()  # 设置为评估模式
    
    # 读取测试数据集 - 限制数量进一步节省显存
    test_df = pd.read_json("new_test.jsonl", lines=True)[:3]  # 减少到前3条做主观评测

    # SwanLab记录
    if swanlab_available:
        swanlab_callback = swanlab.init(
            project="Qwen2-fintune",
            experiment_name="Qwen2-1.5B-Instruct-Inference",
            description="使用微调后的Qwen2-1.5B-Instruct模型进行推理测试。",
            config={
                "model": "qwen/Qwen2-1.5B-Instruct",
                "dataset": "huangjintao/zh_cls_fudan-news",
            },
        )

    # ====== 推理测试 ===== #
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
        
        # 每次迭代后清理显存
        torch.cuda.empty_cache()

    if swanlab_available:
        swanlab.log({"Prediction": test_text_list})
        swanlab.finish()
    
    # 最后再次清理显存
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()