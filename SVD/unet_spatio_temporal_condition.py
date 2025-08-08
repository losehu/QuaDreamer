from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from .embeddings import TimestepEmbedding, Timesteps, PositionNet
from diffusers.models.modeling_utils import ModelMixin
from .unet_3d_blocks import UNetMidBlockSpatioTemporal, get_down_block, get_up_block
from IMU_FUSER.imu import BEVControlNetModel
from objctrl_2_5d.utils.ui_utils import process_image, get_camera_pose, get_subject_points, get_points, undo_points, mask_image, traj2cam, get_mid_params
from cameractrl.pipelines.pipeline_animation import StableVideoDiffusionPipelinePoseCond
from cameractrl.models.unet import UNetSpatioTemporalConditionModelPoseCond
from cameractrl.models.pose_adaptor import CameraPoseEncoder
from cameractrl.utils.util import save_videos_grid

from objctrl_2_5d.utils.objmask_util import RT2Plucker, Unprojected, roll_with_ignore_multidim, dilate_mask_pytorch
from objctrl_2_5d.utils.filter_utils import get_freq_filter, freq_mix_3d

from cameractrl.models.unet_3d_blocks import (
    # get_down_block,
    # get_up_block,
    UNetMidBlockSpatioTemporalPoseCond
)
from cameractrl.models.attention_processor import XFormersAttnProcessor as CustomizedXFormerAttnProcessor
from cameractrl.models.attention_processor import PoseAdaptorXFormersAttnProcessor
import torch
import torch.nn as nn


from FTR.src.models.LaMa import *
from FTR.src.models.TSR_model import *
from FTR.src.models.FTR_model import *

from FTR.src.config import Config

# if hasattr(F, "scaled_dot_product_attention"):
#     from cameractrl.models.attention_processor import PoseAdaptorAttnProcessor2_0 as PoseAdaptorAttnProcessor
#     from cameractrl.models.attention_processor import AttnProcessor2_0 as CustomizedAttnProcessor
# else:
from cameractrl.models.attention_processor import PoseAdaptorAttnProcessor
from cameractrl.models.attention_processor import AttnProcessor as CustomizedAttnProcessor

from omegaconf import OmegaConf
import numpy as np
import os
add_imu = os.getenv('ADD_IMU')
if add_imu == 'true' or add_imu == 'True':
    add_imu = True
else:
    add_imu = False 

add_obj = os.getenv('ADD_OBJ')
if add_obj == 'true' or add_obj == 'True':
    add_obj = True
else:
    add_obj = False 
add_ftr = os.getenv('ADD_FTR')
if add_ftr == 'true' or add_ftr == 'True':
    add_ftr = True
else:
    add_ftr = False 

# import sys,os
# sys.path.append("/root/autodl-tmp/TrackDiffusion-SVD/")  # 替换成实际路径
# add_imu = os.getenv('ADD_IMU')
# if add_imu == 'true' or add_imu == 'True':
#     add_imu = True
# else:
#     add_imu = False 
logger = logging.getLogger(__name__)


@dataclass
class UNetSpatioTemporalConditionOutput(BaseOutput):
    """
    The output of [`UNetSpatioTemporalConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None


class UNetSpatioTemporalConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    A conditional Spatio-Temporal UNet model that takes a noisy video frames, conditional state, and a timestep and returns a sample
    shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 8): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        addition_time_embed_dim: (`int`, defaults to 256):
            Dimension to to encode the additional time ids.
        projection_class_embeddings_input_dim (`int`, defaults to 768):
            The dimension of the projection of encoded `added_time_ids`.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple]` , *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_3d_blocks.CrossAttnDownBlockSpatioTemporal`], [`~models.unet_3d_blocks.CrossAttnUpBlockSpatioTemporal`],
            [`~models.unet_3d_blocks.UNetMidBlockSpatioTemporal`].
        num_attention_heads (`int`, `Tuple[int]`, defaults to `(5, 10, 10, 20)`):
            The number of attention heads.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 10, 20),
        num_frames: int = 25,
    ):
        super().__init__()

        self.sample_size = sample_size

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.add_time_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        # if add_imu:
        #     self.imu_fuser = BEVControlNetModel()
        if add_obj:
            config = "configs/svd_320_576_cameractrl.yaml"
            model_config = OmegaConf.load(config)
            self.pose_encoder= CameraPoseEncoder(**model_config['pose_encoder_kwargs'])

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-5,
                resolution_idx=i,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()

        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )
        
        positive_len = 768
        if isinstance(cross_attention_dim, int):
            positive_len = cross_attention_dim
        elif isinstance(cross_attention_dim, tuple) or isinstance(cross_attention_dim, list):
            positive_len = cross_attention_dim[0]
        self.position_net = PositionNet(positive_len=positive_len, out_dim=cross_attention_dim)
        if add_obj:
            self.set_pose_cond_attn_processor()
        if add_ftr:
            config = Config("/root/autodl-tmp/TrackDiffusion-SVD/FTR/config_list/config_ZITS_places2.yml")
            self.ftr=DefaultInpaintingTrainingModule(config).to(dtype=torch.float16)
            # self.ftr_imu = IMU2ImageNet().to(dtype=torch.float16)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def set_pose_cond_attn_processor(self,
                                     add_spatial=False,
                                     add_temporal=True,
                                     enable_xformers=True,
                                     attn_processor_name='attn1',
                                     pose_feature_dimensions=[320, 640, 1280, 1280],
                                     ):
        attention_processor_kwargs={'query_condition': True, 'key_value_condition': True, 'scale': 1.0}
        all_attn_processors = {}
        set_processor_names = attn_processor_name.split(',')
        if add_spatial:
            for processor_key in self.attn_processors.keys():
                if 'temporal' in processor_key:
                    continue
                processor_name = processor_key.split('.')[-2]
                cross_attention_dim = None if processor_name == 'attn1' else self.config.cross_attention_dim
                if processor_key.startswith("mid_block"):
                    hidden_size = self.config.block_out_channels[-1]
                    block_id = -1
                    add_pose_adaptor = processor_name in set_processor_names
                    pose_feature_dim = pose_feature_dimensions[block_id] if add_pose_adaptor else None
                elif processor_key.startswith("up_blocks"):
                    block_id = int(processor_key[len("up_blocks.")])
                    hidden_size = list(reversed(self.config.block_out_channels))[block_id]
                    add_pose_adaptor = processor_name in set_processor_names
                    pose_feature_dim = list(reversed(pose_feature_dimensions))[block_id] if add_pose_adaptor else None
                else:
                    block_id = int(processor_key[len("down_blocks.")])
                    hidden_size = self.config.block_out_channels[block_id]
                    add_pose_adaptor = processor_name in set_processor_names
                    pose_feature_dim = pose_feature_dimensions[block_id] if add_pose_adaptor else None
                if add_pose_adaptor and enable_xformers:
                    all_attn_processors[processor_key] = PoseAdaptorXFormersAttnProcessor(hidden_size=hidden_size,
                                                                                  pose_feature_dim=pose_feature_dim,
                                                                                  cross_attention_dim=cross_attention_dim,
                                                                                  **attention_processor_kwargs)
                elif add_pose_adaptor:
                    all_attn_processors[processor_key] = PoseAdaptorAttnProcessor(hidden_size=hidden_size,
                                                                                  pose_feature_dim=pose_feature_dim,
                                                                                  cross_attention_dim=cross_attention_dim,
                                                                                  **attention_processor_kwargs)
                elif enable_xformers:
                    all_attn_processors[processor_key] = CustomizedXFormerAttnProcessor()
                else:
                    all_attn_processors[processor_key] = CustomizedAttnProcessor()
        else:
            for processor_key in self.attn_processors.keys():
                if 'temporal' not in processor_key and enable_xformers:
                    all_attn_processors[processor_key] = CustomizedXFormerAttnProcessor()
                    # if 'pose' not in processor_key:
                    #      all_attn_processors[processor_key] = self.attn_processors[processor_key]
                elif 'temporal' not in processor_key:
                    all_attn_processors[processor_key] = CustomizedAttnProcessor()
                    # if 'pose' not in processor_key:
                    #      all_attn_processors[processor_key] = self.attn_processors[processor_key]

        if add_temporal:
            for processor_key in self.attn_processors.keys():
                if 'temporal' not in processor_key:
                    continue
                processor_name = processor_key.split('.')[-2]
                cross_attention_dim = None if processor_name == 'attn1' else self.config.cross_attention_dim
                if processor_key.startswith("mid_block"):
                    hidden_size = self.config.block_out_channels[-1]
                    block_id = -1
                    add_pose_adaptor = processor_name in set_processor_names
                    pose_feature_dim = pose_feature_dimensions[block_id] if add_pose_adaptor else None
                elif processor_key.startswith("up_blocks"):
                    block_id = int(processor_key[len("up_blocks.")])
                    hidden_size = list(reversed(self.config.block_out_channels))[block_id]
                    add_pose_adaptor = (processor_name in set_processor_names)
                    pose_feature_dim = list(reversed(pose_feature_dimensions))[block_id] if add_pose_adaptor else None
                else:
                    block_id = int(processor_key[len("down_blocks.")])
                    hidden_size = self.config.block_out_channels[block_id]
                    add_pose_adaptor = processor_name in set_processor_names
                    pose_feature_dim = pose_feature_dimensions[block_id] if add_pose_adaptor else None
                if add_pose_adaptor and enable_xformers:

                    all_attn_processors[processor_key] = PoseAdaptorAttnProcessor(hidden_size=hidden_size,
                                                                                          pose_feature_dim=pose_feature_dim,
                                                                                          cross_attention_dim=cross_attention_dim,
                                                                                          **attention_processor_kwargs)
                    # if 'pose' not in processor_key:
                    #     all_attn_processors[processor_key] = self.attn_processors[processor_key]
                elif add_pose_adaptor:
                    all_attn_processors[processor_key] = PoseAdaptorAttnProcessor(hidden_size=hidden_size,
                                                                                  pose_feature_dim=pose_feature_dim,
                                                                                  cross_attention_dim=cross_attention_dim,
                                                                                  **attention_processor_kwargs)
                    # if 'pose' not in processor_key:
                    #      all_attn_processors[processor_key] = self.attn_processors[processor_key]
                elif enable_xformers:
                    all_attn_processors[processor_key] = CustomizedXFormerAttnProcessor()
                    # if 'pose' not in processor_key:
                    #      all_attn_processors[processor_key] = self.attn_processors[processor_key]
                else:
                    all_attn_processors[processor_key] = CustomizedAttnProcessor()
                    # if 'pose' not in processor_key:
                    #      all_attn_processors[processor_key] = self.attn_processors[processor_key]
        else:
            for processor_key in self.attn_processors.keys():
                if 'temporal' in processor_key and enable_xformers:
                    all_attn_processors[processor_key] = CustomizedXFormerAttnProcessor()
                    # if 'pose' not in processor_key:
                    #      all_attn_processors[processor_key] = self.attn_processors[processor_key]
                elif 'temporal' in processor_key:
                    all_attn_processors[processor_key] = CustomizedAttnProcessor()
                    # if 'pose' not in processor_key:
                    #      all_attn_processors[processor_key] = self.attn_processors[processor_key]

        self.set_attn_processor(all_attn_processors)
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        imu_data:torch.Tensor=None,
        return_dict: bool = True,
        cross_attention_kwargs = None,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.FloatTensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead of a plain
                tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # if add_imu:
        #     encoder_hidden_states=self.imu_fuser(encoder_hidden_states,imu_data.unsqueeze(1).unsqueeze(1))
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)
        if add_obj:
            height,width=128,512
            bg_mode="Reverse"
            pose_features_all=[]
            
            for i in range( imu_data.shape[0]):
                traj =imu_data[i].squeeze(0)
                traj=torch.stack(( torch.ones_like(traj),traj), dim=1)
                traj = traj.to(dtype=sample.dtype).to(sample.device)
                traj[ :,0] =  width/2# torch.linspace(0, width, traj.size(0)).to(dtype=sample.dtype).to(sample.device)
                traj[ :,1] =  traj[ :,1]
                num_frames=traj.shape[0]
                intrinsics = np.array([[float(width), float(width), float(width) / 2, float(height) / 2]])
                intrinsics = np.repeat(intrinsics, num_frames, axis=0) # [n_frame, 4]
                fx = intrinsics[0, 0] / width
                fy = intrinsics[0, 1] / height
                cx = intrinsics[0, 2] / width
                cy = intrinsics[0, 3] / height


                RTs=traj2cam(traj, 1,intrinsics)

                cur_plucker_embedding, _, _ = RT2Plucker(RTs, RTs.shape[0], (height, width), fx, fy, cx, cy) # 6, V, H, W
                cur_plucker_embedding = cur_plucker_embedding.to(dtype=sample.dtype).to(sample.device)
                cur_plucker_embedding = cur_plucker_embedding[None, ...] # b 6 f h w
                cur_plucker_embedding = cur_plucker_embedding.permute(0, 2, 1, 3, 4) # b f 6 h w
                cur_plucker_embedding = cur_plucker_embedding[:, :num_frames, ...]#torch.Size([1, 14, 6, 320, 576])
                cur_pose_features = self.pose_encoder(cur_plucker_embedding)#torch.Size([1, 320, 14, 40, 72])
                if bg_mode == "Fixed":
                    fix_RTs = np.repeat(RTs[0][None, ...], num_frames, axis=0) # [n_frame, 4, 3](14, 3, 4)->(14, 3, 4)
                    fix_plucker_embedding, _, _ = RT2Plucker(fix_RTs, num_frames, (height, width), fx, fy, cx, cy) # 6, V, H, W
                    fix_plucker_embedding = fix_plucker_embedding.to(dtype=sample.dtype).to(sample.device)
                    fix_plucker_embedding = fix_plucker_embedding[None, ...] # b 6 f h w
                    fix_plucker_embedding = fix_plucker_embedding.permute(0, 2, 1, 3, 4) # b f 6 h w
                    fix_plucker_embedding = fix_plucker_embedding[:, :num_frames, ...]
                    fix_pose_features = self.pose_encoder(fix_plucker_embedding)#4* torch.Size([1, 320, 14, 40, 72])
                    
                elif bg_mode == "Reverse":
                    bg_plucker_embedding, _, _ = RT2Plucker(RTs[::-1], RTs.shape[0], (height, width), fx, fy, cx, cy) # 6, V, H, W
                    bg_plucker_embedding = bg_plucker_embedding.to(dtype=sample.dtype).to(sample.device)
                    bg_plucker_embedding = bg_plucker_embedding[None, ...] # b 6 f h w
                    bg_plucker_embedding = bg_plucker_embedding.permute(0, 2, 1, 3, 4) # b f 6 h w
                    bg_plucker_embedding = bg_plucker_embedding[:, :num_frames, ...]
                    fix_pose_features = self.pose_encoder(bg_plucker_embedding)
                    
                else:
                    fix_pose_features = None

                kernel_sizes = [5, 3, 3, 7]
                pose_features = []

                for i in range(0, len(cur_pose_features)):
                    kernel_size = kernel_sizes[i]
                    h, w = cur_pose_features[i].shape[-2:]
                    if fix_pose_features is None:
                        pose_features.append(torch.zeros_like(cur_pose_features[i]))
                    else:
                        pose_features.append(fix_pose_features[i])
                pose_embedding=cur_plucker_embedding
                if pose_features is None:
                    assert pose_embedding.ndim == 5                         # [b, f, c, h, w]
                    pose_features = self.pose_encoder(pose_embedding)       # list of [b, c, f, h, w]
                pose_features[-1] = cur_pose_features[i] 
                pose_features_all.append(pose_features)



            #cur_plucker_embedding and pose_features
            pose_features_in = []
            # 检查 pose_features_all 的长度
            if len(pose_features_all) == 0:
                # 如果长度为 0，直接跳过
                raise ValueError("pose_features_all 是空的，无法处理")
            elif len(pose_features_all) == 1:
                # 如果长度为 1，直接使用第一个列表中的特征
                for i in range(len(pose_features_all[0])):
                    pose_features_in.append(pose_features_all[0][i])
            else:
                # 如果长度大于 1，按顺序拼接每个列表中的特征
                for i in range(len(pose_features_all[0])):
                    # 检查每个列表的长度是否一致
                    if any(len(features) != len(pose_features_all[0]) for features in pose_features_all):
                        raise ValueError("pose_features_all 中的列表长度不一致，无法拼接")
                    # 拼接所有列表中的第 i 个特征
                    concatenated_features = torch.cat([features[i] for features in pose_features_all], dim=0)
                    pose_features_in.append(concatenated_features)
        if add_imu:
            cross_attention_kwargs["gligen"]["pose_features_in"]=pose_features_in
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            location_args = cross_attention_kwargs.pop("gligen")
            fixed_objs = cross_attention_kwargs.pop("fixed_objs")
            bs = sample.shape[0]
            objs = self.position_net(**location_args)
            cross_attention_kwargs["gligen"] = {"objs": objs.flatten(0, 1).view(bs, -1, 1024),
                                                "boxes": location_args["boxes"],
                                                "masks": location_args["masks"],
                                                }
            if add_imu:
                cross_attention_kwargs["gligen"] ["pose_features_in"]= location_args["pose_features_in"],
        down_block_res_samples = (sample,)
        for block_idx,downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention :
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    cross_attention_kwargs=cross_attention_kwargs,
                    pose_feature=pose_features_in[block_idx] if add_obj else None,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
            cross_attention_kwargs=cross_attention_kwargs,
            pose_feature=pose_features_in[-1] if add_obj else None
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    cross_attention_kwargs=cross_attention_kwargs,
                    pose_feature=pose_features_in[-(i + 1)] if add_obj else None
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)
