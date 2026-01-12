from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download("ZhipuAI/chatglm3-6b", cache_dir="./", revision="master")