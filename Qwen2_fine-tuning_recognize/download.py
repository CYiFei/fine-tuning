import os
from modelscope import snapshot_download

def download_model(model_id, local_dir="./models"):
    """
    下载指定的模型到本地目录
    """
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    print(f"开始下载模型 {model_id} 到 {local_dir}")
    model_dir = snapshot_download(
        model_id, 
        cache_dir=local_dir, 
        revision="master"
    )
    print(f"模型下载完成: {model_dir}")
    return model_dir

if __name__ == "__main__":
    # 示例：下载Qwen2-1.5B-Instruct模型
    model_id = "qwen/Qwen2-1.5B-Instruct"
    download_model(model_id)