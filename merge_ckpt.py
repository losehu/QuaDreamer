from safetensors.torch import load_file, save_file
import torch
from collections import OrderedDict

def ensure_tensor(value):
    """递归地确保值是 torch.Tensor 类型"""
    if isinstance(value, torch.Tensor):
        return value
    elif isinstance(value, dict):
        # 处理字典中的每个元素，递归地确保它们是 tensor
        return {k: ensure_tensor(v) for k, v in value.items()}
    elif isinstance(value, OrderedDict):
        # 处理 OrderedDict 类型的值
        return OrderedDict((k, ensure_tensor(v)) for k, v in value.items())
    else:
        raise ValueError(f"Unexpected type: {type(value)}")

def modify_and_merge_ckpt_with_safetensors(ckpt_path, safetensors_path, output_path):
    # 加载 .ckpt 文件
    ckpt = load_file(ckpt_path)  # 使用 torch.load 加载模型权重
    # 加载 .safetensors 文件
    safetensors = load_file(safetensors_path)
    
    # 创建一个新的字典用于保存合并后的权重
    merged_checkpoint = {}

    # 1. 修改 .ckpt 中的参数名称
    for name, param in ckpt.items():

        if 'pose_encoder_state_dict' in name:
            new_name = name.replace('pose_encoder_state_dict', 'pose_encoder')
            merged_checkpoint[new_name] = param
        elif 'attention_processor_state_dict' in name:
            new_name = name.replace('attention_processor_state_dict.', '')
            merged_checkpoint[new_name] = param
        else:
            merged_checkpoint[name] = param

    # 2. 将 .safetensors 中的参数添加到 merged_checkpoint 中
    for name, param in safetensors.items():

        # 如果该参数名不在 merged_checkpoint 中，直接加入
        if name not in merged_checkpoint:
            # if 'pose_encoder' not in name and 'attn_pose' not in name and 'attn2_pose' not in name:
            merged_checkpoint[name] = param


    # 4. 保存合并后的字典为新的 .safetensors 文件
    save_file(merged_checkpoint, output_path)
    print(f"Modified and merged weights saved to {output_path}")

# 示例使用
ckpt_path = './CameraCtrl_svd.safetensors'  # 替换为你的 .ckpt 文件路径
safetensors_path = '/root/autodl-tmp/TrackDiffusion-SVD/TrackDiffusion_Pretrain/stable-video-diffusion-img2vid/unet/diffusion_pytorch_model.safetensors'  # 替换为你的 safetensors 文件路径
#/root/autodl-tmp/TrackDiffusion-SVD/trackdiffusion_ytvis/modelscope_ft/unet/diffusion_pytorch_model.safetensors
output_path = '/root/autodl-tmp/TrackDiffusion-SVD/TrackDiffusion_Pretrain/presvd+precam/diffusion_pytorch_model.safetensors'  # 保存文件的路径

# 调用函数
modify_and_merge_ckpt_with_safetensors(ckpt_path, safetensors_path, output_path)
