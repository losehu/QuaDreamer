from email.headerregistry import DateHeader
import random
import numpy as np
from mydata import CustomDataset
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torch
from collections import Counter
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor
import datetime
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import os  # 操作系统接口
import re  # 正则表达式
from PIL import Image, ImageDraw, ImageSequence, ImageFont  # 图像处理
from moviepy.editor import ImageSequenceClip  # 视频处理
import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
import gradio as gr  # 交互式Web应用框架
from SVD.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel  # 时空条件UNet模型
from pipelines.pipeline_stable_video_diffusion_image import StableVideoDiffusionPipeline  # 视频扩散管道
from common_metrics_on_video_quality.calculate_fvd import calculate_fvd
from common_metrics_on_video_quality.fvd.styleganv.fvd import load_i3d_pretrained
from common_metrics_on_video_quality.calculate_psnr import calculate_psnr
from common_metrics_on_video_quality.calculate_ssim import calculate_ssim
from common_metrics_on_video_quality.calculate_lpips import calculate_lpips
import os
import shutil
add_imu = os.getenv('ADD_IMU')
add_tsr = os.getenv('ADD_TSR')
add_obj = os.getenv('ADD_OBJ')

if add_imu == 'true' or add_imu == 'True':
    add_imu = True
else:
    add_imu = False 
if add_tsr == 'true' or add_tsr == 'True':
    add_tsr = True
else:
    add_tsr = False 
if add_obj == 'true' or add_obj == 'True':
    add_obj = True
else:
    add_obj = False 
# 加载预训练模型路径/rooautodl-tmp/TrackDiffusion-SVD/outputs/
pretrained_model_path = "./TrackDiffusion_Pretrain/stable-video-diffusion-img2vid/"
# unet_path="/root/autodl-tmp/TrackDiffusion-SVD/outputs/2025-04-26-06-30-10/unet"
unet_path="/root/autodl-tmp/TrackDiffusion-SVD/outputs/35/2025-04-16-12-15-11/unet"

# 加载UNet模型
unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16,)
# 创建稳定视频扩散管道
pipe = StableVideoDiffusionPipeline.from_pretrained(
    pretrained_model_path, 
    unet=unet,
    torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=True)
pipe = pipe.to('cuda:0')  # 将管道移动到GPU
def copy_folder(src_folder, dst_folder):
    """
    复制源文件夹到目标文件夹。如果目标文件夹已存在，则先删除之前的文件夹。
    
    :param src_folder: 源文件夹路径
    :param dst_folder: 目标文件夹路径
    """
    # 如果目标文件夹已经存在，先删除它
    if os.path.exists(dst_folder):
        try:
            shutil.rmtree(dst_folder)
            print(f"目标文件夹 {dst_folder} 已被删除。")
        except Exception as e:
            print(f"删除目标文件夹失败: {e}")
            return

    # 复制整个文件夹及其内容
    try:
        shutil.copytree(src_folder, dst_folder)
        print(f"文件夹 {src_folder} 已成功复制到 {dst_folder}")
    except Exception as e:
        print(f"复制失败: {e}")
def draw_bboxes_on_image(image, bboxes, colors):
    """在图像上绘制边界框"""
    draw = ImageDraw.Draw(image)
    for bbox, color in zip(bboxes, colors):
        # 画框，bbox 格式为 [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=tuple(map(int, color)), width=3)
    return image
def draw_bboxes_and_save(ori_pixel_values, bboxes_prompt, output_dir):
    """
    使用 bboxes_prompt 绘制目标框并保存图像。
    ori_pixel_values 的形状是 [1, 8, 3, 256, 512]
    bboxes_prompt 的形状是 [1, 8, 20, 7]，每个框包含 [x1, y1, x2, y2, R, G, B]
    """
    # 提取信息
    bboxes_prompt = bboxes_prompt.squeeze(0)  # 去掉 batch 维度, 形状变成 [8, 20, 7]
    ori_pixel_values = ori_pixel_values.squeeze(0)  # 去掉 batch 维度, 形状变成 [8, 3, 256, 512]
    ori_pixel_values = (ori_pixel_values + 1) * 127.5

    # 遍历每一张图像
    for i in range(ori_pixel_values.shape[0]):
        # 获取当前图像和 bounding boxes
        img_tensor = ori_pixel_values[i]  # 形状为 [3, 256, 512]
        bboxes = bboxes_prompt[i, :, :4]  # 获取坐标 [x1, y1, x2, y2]
        colors = bboxes_prompt[i, :, 4:]  # 获取颜色 [R, G, B]
        
        # 转换成 PIL 图片格式
        img = Image.fromarray(img_tensor.permute(1, 2, 0).byte().cpu().numpy())
                
        
        # 将 bboxes 转换成图片坐标系中的值
        width, height = img.size
        bboxes[:, 0] *= width  # x1
        bboxes[:, 1] *= height  # y1
        bboxes[:, 2] *= width  # x2
        bboxes[:, 3] *= height  # y2
        
        # 绘制 bounding boxes
        img = draw_bboxes_on_image(img, bboxes, colors)
        
        # 保存图片
        img.save(os.path.join(output_dir, f"frame_{i}.png"))
# 坐标归一化函数（将绝对坐标转换为相对坐标）
def normalize_coordinates(arr, width, height):
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]

    x1_normalized = x1 / width  # 归一化x1坐标
    y1_normalized = y1 / height  # 归一化y1坐标
    x2_normalized = x2 / width  # 归一化x2坐标
    y2_normalized = y2 / height  # 归一化y2坐标

    normalized_arr = np.stack((x1_normalized, y1_normalized, x2_normalized, y2_normalized), axis=1)
    return normalized_arr

# 导出视频函数（将帧序列转换为视频文件）
def export_to_video(frames, video_path):
    clip = ImageSequenceClip(list(frames), fps=7)  # 创建视频片段
    clip.write_videofile(video_path, codec='libx264')  # 写入视频文件


def save_tensor_as_image(image_tensor, output_path):
    """
    将 torch 格式的图片（[-1, 1] 范围）转换为 PIL 图片并保存到本地。

    Args:
        image_tensor (torch.Tensor): 输入的图像，范围为 [-1, 1]，形状为 (C, H, W)
        output_path (str): 保存图片的路径
    """
    # 将 tensor 图像从 [-1, 1] 转换到 [0, 255] 范围
    image_numpy = ((image_tensor + 1) / 2 * 255).clamp(0, 255).byte()

    # 转换为 HxWxC 形式的 NumPy 数组 (C, H, W) -> (H, W, C)
    image_numpy = image_numpy.permute(1, 2, 0).cpu().numpy()

    # 使用 PIL 从 NumPy 数组创建图像
    image_pil = Image.fromarray(image_numpy)

    # 保存图像到本地
    image_pil.save(output_path)
    # print(f"图像已保存到 {output_path}")


def validation_step(step,batch,i3d,fvd_all,ssim_all,psnr_all,lpips_all):
    # save_tensor_as_image(batch['ori_pixel'][0][0,:,:,:], 'output_image.jpg')
    tensor = batch['ori_pixel'].squeeze(0)  # 从 [1, 8, 3, H, W] 转为 [8, 3, H, W]
    # 恢复到 [0, 1] 范围
    to_tensor = transforms.ToTensor()

    tensor = (tensor + 1.0) / 2.0 #原图
    img = tensor[0]  # 获取第 0 张图像

    img_init = transforms.ToPILImage()(img)  # 使用 transforms.ToPILImage() 实例进行转换
    frames = []

    # for i in range(14):
    #     frames.append(np.array(transforms.ToPILImage()(tensor[i])))
    # export_to_video(np.array(frames), f"./outputs/val-train/{step}_ori.mp4")

    images = pipe(
        image=img_init,  # 调整输入图像尺寸
        image_prompt=img_init,
        width=512,
        height=128,
        fps=8,
        video_masks=batch['mask'].squeeze(0),
        bbox_prompt= batch['bboxes'].squeeze(0) ,
        imu_data= batch['imu'] ,
        num_frames=14,
        num_inference_steps=30,  # 推理步数
        noise_aug_strength=0.02,  # 噪声增强强度
        motion_bucket_id=127,  # 运动桶ID
        output_type='pil',
    ).frames[0]  # 获取生成的帧
    frames = []
    for fid in range(14):
        img = images[fid]
        # 注释掉的绘制边界框代码
        frames.append(np.array(img))  # 收集所有帧
    export_to_video(np.array(frames),f"./outputs/val-train/{step}.mp4")
    for i in range(14):
        img = images[i].resize((2048, 480))
        save_path=batch['path'][i][0]
        save_path=os.path.join( unet_path.rstrip('/unet'),save_path.replace('./mydata','.'))
        img.save(save_path)
    tensor_list = [to_tensor(img) for img in images]
    # 使用 torch.stack 将列表中的所有张量堆叠成一个批量张量，形状为 (n, 3, h, w)
    batch_tensor = torch.stack(tensor_list)
    video1=batch_tensor.unsqueeze(0).to("cuda")
    video2=tensor.unsqueeze(0).to("cuda")
    fvd = calculate_fvd(video1,video2 ,  torch.device("cuda"),i3d, method='styleganv', only_final=True)
    ssim= calculate_ssim(video1.to("cpu"), video2.to("cpu"), only_final=True)
    psnr=  calculate_psnr(video1.to("cpu"), video2.to("cpu"), only_final=True)
    lpips=calculate_lpips(video1, video2,  torch.device("cuda"), only_final=True)
    fvd_all.append(fvd['value'][0])
    ssim_all.append(ssim['value'][0])
    psnr_all.append(psnr['value'][0])
    lpips_all.append(lpips['value'][0])




def main(train_dataloader):

    i3d=load_i3d_pretrained(device=torch.device("cuda"))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    fvd_all=[]
    ssim_all=[]
    psnr_all=[]
    lpips_all=[]
    for step, batch in enumerate(train_dataloader):
    # for i in range(train_dataloader.__len__()):
        # step=i
        # batch=train_dataloader.__getitem__(i)
        print(f"{step+1}/{len(train_dataloader)}")
        validation_step(step,batch,i3d,fvd_all,ssim_all,psnr_all,lpips_all)
        with open(os.path.join(unet_path.rstrip('/unet') , f"eval_{timestamp}.txt"), 'a') as f:
            f.write(f"{step+1}/{len(train_dataloader)}: {fvd_all[-1]} {lpips_all[-1]} {psnr_all[-1]} {ssim_all[-1]}\n")
            print(f"{step+1}/{len(train_dataloader)}: {fvd_all[-1]} {lpips_all[-1]} {psnr_all[-1]} {ssim_all[-1]}")
            print(f"avg: {sum(fvd_all) / len(fvd_all)} {sum(lpips_all) / len(lpips_all)} {sum(psnr_all) / len(psnr_all)} {sum(ssim_all) / len(ssim_all)}")
    with open(os.path.join(unet_path.rstrip('/unet') , f"eval_{timestamp}.txt"), 'a') as f:
        f.write(f"avg: {sum(fvd_all) / len(fvd_all)} {sum(lpips_all) / len(lpips_all)} {sum(psnr_all) / len(psnr_all)} {sum(ssim_all) / len(ssim_all)}\n")

if __name__ == "__main__":
    data_dir = './mydata'
    copy_folder(os.path.join(data_dir,"test"),os.path.join( unet_path.rstrip('/unet'),'test') )
    num_workers = 14
    seed = 1234  # 设置随机种子
    random.seed(seed)  # 设置随机种子
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    train_dataset = CustomDataset(data_dir, frame_num=num_workers, train=False,track_length=50)  # 传递 num_workers 和 seed 参数
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
    )
    
    main(train_dataloader)


    # {'value': [581.6776189936741]}

