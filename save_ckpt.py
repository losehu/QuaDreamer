from safetensors.torch import load_file, save_file

def filter_pose_weights(input_path, output_path):
    # 加载 safetensors 文件
    checkpoint = load_file(input_path)
    
    # 用于保存符合条件的权重
    filtered_weights = {}

    # 遍历权重
    for name, param in checkpoint.items():

        # 如果参数名中不包含 'pose'，直接保存
        if 'pose' not in name:
            filtered_weights[name] = param
        else:
            # 如果参数名中包含 'pose'，且同时包含 'qkv_merge' 或 'pose_encoder'，保存该权重
            if 'qkv_merge' in name or 'pose_encoder' in name:
                if 'attn2_pose' not in name:
                    if 'attn_pose'  in name:
                        name=name.replace('attn_pose','attn1')
                        filtered_weights[name] = param
                        print(name)
                    else:
                        filtered_weights[name] = param
                else:
                    continue
            # 如果参数名包含 'pose' 但不包含 'qkv_merge' 或 'pose_encoder'，不保存
            else:
                continue
    
    # 保存符合条件的权重到新的 safetensors 文件
    save_file(filtered_weights, output_path)
    print(f"Filtered weights saved to {output_path}")

# 示例使用
input_path = '/root/autodl-tmp/TrackDiffusion-SVD/outputs/2025-03-25-01-18-08_cam2/unet/diffusion_pytorch_model.safetensors'  # 替换为你的 safetensors 文件路径
output_path = './diffusion_pytorch_model.safetensors'  # 保存文件的路径

# 调用函数
filter_pose_weights(input_path, output_path)



#  down_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, mid_block.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, up_blocks.1.attentions.2.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, up_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, down_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, up_blocks.3.attentions.2.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, up_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, up_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, mid_block.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, down_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, down_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, up_blocks.2.attentions.2.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, up_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, up_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, up_blocks.2.attentions.2.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, down_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, up_blocks.1.attentions.2.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, up_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, up_blocks.3.attentions.2.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, up_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, down_blocks.0.attentions.1.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, down_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, down_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.weight, down_blocks.0.attentions.1.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, down_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias, up_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor.qkv_merge.bias. 