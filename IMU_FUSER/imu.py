import torch
import numpy as np
from einops import rearrange
from IMU_FUSER.embedder import get_embedder
from einops import repeat, rearrange
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union, List

# 你的 BEVControlNetModel
class BEVControlNetModel(nn.Module):
    def __init__(self):
        super().__init__()  # 重要！初始化 nn.Module
        uncond_imu_in_dim=(3,7)
        self.uncond_imu_num = uncond_imu_in_dim[1]
        camera_in_dim=72
        camera_out_dim=1024
        param = {
            'input_dims': 1,
            'num_freqs': 4,
            'log_sampling': True,
            'include_input': True
        }
        self.imu_embedder = get_embedder(**param)
        self.imu2token = nn.Linear(camera_in_dim, camera_out_dim)
        self.uncond_imu = nn.Embedding( 1,uncond_imu_in_dim[0] * uncond_imu_in_dim[1])
    def _embed_camera(self, camera_param):
        """
        Args:
            camera_param (torch.Tensor): [N, 6, 3, 7], 7 for 3 + 4
        """
        (bs, N_cam, C_param, emb_num) = camera_param.shape
        # assert C_param == 3
        # assert emb_num == self.uncond_imu_num[1] or self.uncond_imu_num is None, (
        #     f"You assigned `uncond_imu_in_dim[1]={self.uncond_imu_num[1]}`, "
        #     f"but your data actually has {emb_num} to embed. Please change your config."
        # )
        camera_emb = self.imu_embedder(
            rearrange(camera_param, "b n d c -> (b n c) d")
        )
        camera_emb = rearrange(
            camera_emb, "(b n c) d -> b n (c d)", n=N_cam, b=bs
        )
        return camera_emb
    def uncond_imu_param(self, repeat_size: Union[List[int], int] = 1):
        if isinstance(repeat_size, int):
            repeat_size = [1, repeat_size]
        repeat_size_sum = int(np.prod(repeat_size))
        # we only have one uncond cam, embedding input is always 0
        param = self.uncond_imu(torch.LongTensor(
            [0] * repeat_size_sum).to(device=self.device))
        param = param.reshape(*repeat_size, -1, self.uncond_imu_num)
        return param
    def add_cam_states(self, encoder_hidden_states, camera_emb=None):
        """
        Args:
            encoder_hidden_states (torch.Tensor): b, len, 768
            camera_emb (torch.Tensor): b, n_cam, dim. if None, use uncond cam.
        """
        bs = encoder_hidden_states.shape[0]
        if camera_emb is None:
            # B, 1, 768
            cam_hidden_states = self.imu2token(self._embed_camera(
                self.uncond_imu_param(bs)))
        else:
            camera_emb = camera_emb.to(self.imu2token.weight.dtype)  # 确保 dtype 一致
            cam_hidden_states = self.imu2token(camera_emb)  # B, 1, dim
        # N_cam = cam_hidden_states.shape[1]

        return cam_hidden_states
    
    def forward(self, hidden,camera_param):
        camera_emb = self._embed_camera(camera_param)
        # print(f"Input shape: {camera_param.shape} -> Output shape: {camera_emb.shape}")
        
        return self.add_cam_states(hidden,camera_emb)
    
# # 创建 BEVControlNetModel 实例
# model = BEVControlNetModel()

# # 生成符合 [3, 6, 3, 7] 形状的 camera_param（随机模拟数据）
# camera_param = torch.randn(2, 1,1, 8)  # Batch=3, IMU=1, Channels=1, Embedding_size=8
# hidden = torch.randn(2, 1,1024)  # Batch=3, IMU=1, Channels=1, Embedding_size=8
# # 调用 forward
# output = model.forward(camera_param)

# output=model.add_cam_states(hidden,output)

# # 打印输出形状
# print("Final embedded camera shape:", output.shape)


# /root/micromamba/envs/svd/lib/python3.10/site-packages/diffusers/models/unet_spatio_temporal_condition.py
#         self.imu_fuser = BEVControlNetModel()

#         # Repeat the embeddings num_video_frames times
#         # emb: [batch, channels] -> [batch * frames, channels]
#         emb = emb.repeat_interleave(num_frames, dim=0)
#         encoder_hidden_states=self.imu_fuser(encoder_hidden_states,imu_data.unsqueeze(1).unsqueeze(1))
#         # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
#         encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)