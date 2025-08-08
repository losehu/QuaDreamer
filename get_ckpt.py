from safetensors.torch import load_file
import torch

def print_pose_weights(state_dict, prefix=''):
    for name, param in state_dict.items():
        # 如果当前的权重名称包含 'pose'，则打印该权重的名称和形状
        # if 'pose' in name.lower() or True:  # 忽略大小写
        # if "attention_processor_state_dict"  in name.lower():
        print(f"{prefix + name}: {param.size()}")  # 打印权重名称和其形状

        # # 如果当前的参数是字典类型，则递归遍历子字典
        # elif isinstance(param, dict):
        #     print_pose_weights(param, prefix + name + '.')
# ckpt_path = './pose_weights.safetensors'
# 加载 .safetensors 文件/root/autodl-tmp/TrackDiffusion-SVD/TrackDiffusion_Pretrain/stable-video-diffusion-img2vid/unet/diffusion_pytorch_model.fp16.safetensors
# ckpt_path = './diffusion_pytorch_model.safetensors'  # 替换为你的文件路径/
# ckpt_path='/root/autodl-tmp/TrackDiffusion-SVD/TrackDiffusion_Pretrain/stable-video-diffusion-img2vid/unet/diffusion_pytorch_model.fp16.safetensors'
# ckpt_path="/root/autodl-tmp/TrackDiffusion-SVD/outputs/2025-03-26-18-10-49/unet/diffusion_pytorch_model.safetensors"
ckpt_path="/root/autodl-tmp/TrackDiffusion-SVD/TrackDiffusion_Pretrain/svd_cam/diffusion_pytorch_model.safetensors"
checkpoint = load_file(ckpt_path)  # 加载 safetensors 文件

# 打印所有包含 'pose' 的权重名称和形状
print_pose_weights(checkpoint)

