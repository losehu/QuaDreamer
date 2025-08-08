import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
    
from transformer import BlockAxial,my_Block_2,EdgeLineGPTConfig



# 模型构建
class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.TSR_Blocks = []
        for _ in range(config.n_layer // 2):
            self.TSR_Blocks.append(BlockAxial(config))
            self.TSR_Blocks.append(my_Block_2(config))
        self.TSR_Blocks = nn.Sequential(*self.TSR_Blocks)

    def forward(self, x):
        return self.TSR_Blocks(x)
    #torch.Size([16, 1024, 320]) 16,1024,16,20
n_embd=1024
model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=n_embd, block_size=32,
                                attn_pdrop=0.0, n_layer=8, n_head=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(model_config).to(device)
print( model_config.n_embd)
input_tensor = torch.randn(8, n_embd, 1,20).to(device)
# 前向传播
output_tensor = model(input_tensor)

# 输出维度
print("输入维度:", input_tensor.shape)
print("输出维度:", output_tensor.shape)