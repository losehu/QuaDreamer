
from safetensors.torch import save_file
import torch

# 加载 ckpt 文件
ckpt_path = '/root/autodl-tmp/TrackDiffusion-SVD/CameraCtrl_svd.ckpt'
model = torch.load(ckpt_path, map_location='cpu')

# 提取模型的权重参数（确保每个值是 torch.Tensor）
model_dict = {}

# 遍历模型字典
for k, v in model.items():
    if isinstance(v, torch.Tensor):  # 只保留 Tensor 类型的参数
        model_dict[k] = v
    elif isinstance(v, dict):  # 如果值是一个字典（如 pose_encoder_state_dict），则递归处理
        for sub_k, sub_v in v.items():
            if isinstance(sub_v, torch.Tensor):
                model_dict[f"{k}.{sub_k}"] = sub_v

# 保存为 safetensors 格式
safetensors_path = '/root/autodl-tmp/TrackDiffusion-SVD/CameraCtrl_svd.safetensors'
save_file(model_dict, safetensors_path)
