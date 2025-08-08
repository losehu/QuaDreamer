import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torch
from collections import Counter
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from torchvision import transforms
from PIL import Image
import numpy as np
import random
color_list = [ 
    (220, 20, 60),    # Bright red
    (0, 0, 255),      # Dark blue
    (106, 0, 228),    # Vivid purple
    (246, 0, 122),    # Bright pink
    (250, 170, 30),   # Bright orange
    (220, 220, 0),    # Yellow
    (165, 42, 42),    # Brown
    (0, 226, 252),    # Bright cyan
    (182, 182, 255),  # Light periwinkle
    (120, 166, 157),  # Soft teal
    (255, 179, 240),  # Pale pink
    (0, 125, 92),     # Sea green
    (0, 220, 176),    # Turquoise
    (255, 99, 164),   # Soft magenta
    (45, 89, 255),    # Bright periwinkle
    (134, 134, 103),  # Olive green
    (197, 226, 255),  # Light sky blue
    (207, 138, 255),  # Lavender
    (74, 65, 105),    # Dark lavender
    (255, 109, 65),   # Bright coral
    (0, 143, 149),    # Dark turquoise
    (209, 99, 106),   # Soft red
    (227, 255, 205),  # Very pale lime
    (163, 255, 0),    # Neon green
    (183, 130, 88),   # Tan
    (166, 74, 118),   # Mauve
    (65, 70, 15),     # Dark olive
    (191, 162, 208),  # Soft purple
    (142, 108, 45),   # Umber ########################33
    (255, 223, 186),  # Light peach
    (0, 255, 255),    # Cyan
    (255, 255, 255),  # White
    (0, 0, 0),        # Black
    (128, 0, 128),    # Purple
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 128),      # Navy
    (255, 255, 0),    # Yellow
    (128, 128, 0),    # Olive
    (0, 128, 128),    # Teal
    (255, 165, 0),    # Orange
    (255, 20, 147),   # Deep pink
    (144, 238, 144),  # Light green
    (32, 178, 170),   # Light sea green
    (255, 0, 255),    # Magenta
    (255, 105, 180),  # Hot pink
    (186, 85, 211),   # Medium orchid
    (135, 206, 235),  # Sky blue
    (255, 99, 71) ,    # Tomato
    (155, 99, 71)     # Tomato？？

]
def draw_bboxes_on_image1(image, bboxes, colors):
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
        img = draw_bboxes_on_image1(img, bboxes, colors)
        
        # 保存图片
        img.save(os.path.join(output_dir, f"frame_{i}.png"))
def draw_bboxes_on_image(images, bboxes, image_size):
    """
    :param images: Tensor of images, shape [batch_size, channels, height, width].
    :param bboxes: Tensor of normalized bbox coordinates, shape [batch_size, num_bboxes, 4].
    :param image_size: Tuple of (height, width) representing the original image size.
    """
    drawn_images = []
    for img, bbox in zip(images, bboxes):
        img = img.permute(1, 2, 0)
        img = img.byte().numpy()
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        idx = 0
        for box in bbox:
            x_min, y_min, x_max, y_max = box[0:4]
            x_min = x_min * image_size[1]
            y_min = y_min * image_size[0]
            x_max = x_max * image_size[1]
            y_max = y_max * image_size[0]
            color=box[4:7]
            draw.rectangle([x_min, y_min, x_max, y_max], outline=tuple(map(int, color.tolist())), width=4)
            idx += 1

        drawn_images.append(img)
    
    return drawn_images

def save_images_from_tensor(tensor, save_dir=".", prefix="image"):
    """
    将一个形状为 [1, 8, 3, H, W] 的 tensor 保存为 8 张 RGB 图像。
    
    参数：
    tensor (torch.Tensor): 形状为 [1, 8, 3, H, W] 的张量，包含 8 张图像。
    save_dir (str): 图像保存的目录，默认为当前工作目录。
    prefix (str): 图像文件名前缀，默认为 "image"。
    """
    # 去除批次维度，转换为 [8, 3, H, W]
    tensor = tensor.squeeze(0)  # 从 [1, 8, 3, H, W] 转为 [8, 3, H, W]

    # 恢复到 [0, 1] 范围
    tensor = (tensor + 1.0) / 2.0

    # 创建转换为 PIL 图像的变换
    to_pil = transforms.ToPILImage()

    # 保存每一张图像
    for i in range(tensor.shape[0]):  # tensor.shape[0] 应该是 8
        img = tensor[i]  # 获取第 i 张图像
        pil_img = to_pil(img)  # 转换为 PIL 图像
        
        # 构建文件路径
        file_path = f"{save_dir}/{prefix}_{i+1}.png"
        
        # 保存图像
        pil_img.save(file_path)
        print(f"Saved {file_path}")

def load_and_stack_images(img_list, target_height=256, target_width=512):
    """
    将 img_list 中的图片加载，并转换成一个形状为 [8, 3, 256, 512] 的张量，
    保持原始像素值范围（即 [0, 255]），不进行归一化。

    参数：
    img_list (list): 包含 8 个图片路径的列表。
    target_height (int): 图片的目标高度。
    target_width (int): 图片的目标宽度。
    
    返回：
    torch.Tensor: 形状为 [8, 3, 256, 512] 的张量。
    """
    # 用于存储所有图片的列表
    img_tensor_list = []
    
    # 遍历图片路径并加载图片
    for img_path in img_list:
        # 打开图片并确保是 RGB 格式
        img = Image.open(img_path).convert("RGB")
        # 调整大小
        img = img.resize((target_width, target_height))
        # 转换为 NumPy 数组 (H, W, C)，并转置为 (C, H, W) 适配 PyTorch 的要求
        img_array = np.array(img)
        img_tensor = torch.tensor(img_array).permute(2, 0, 1)  # 转换为 [3, height, width]
        img_tensor_list.append(img_tensor)
    
    # 将所有的图片堆叠成一个批次，形状为 [8, 3, 256, 512]
    img_tensor_batch = torch.stack(img_tensor_list)
    
    return img_tensor_batch
def shift_image_horizontally(img, shift_pixels):
    """
    将图像在水平方向环形平移，向右为正，向左为负。
    移出的部分拼接到另一边。
    """
    c, h, w = img.shape
    shift_pixels = shift_pixels % w  # 防止越界

    if shift_pixels == 0:
        return img

    # 向右平移
    if shift_pixels > 0:
        right = img[..., -shift_pixels: ]
        left =  img[..., :-shift_pixels]
        shifted = torch.cat((right, left), dim=-1)

    else:
        # 向左平移
        shift_pixels = abs(shift_pixels)
        left = img[..., shift_pixels:]
        right = img[..., :shift_pixels]
        shifted = torch.cat((left, right), dim=-1)


    return shifted
def shift_all(imgs, shift_nums):
    for i in range(len(shift_nums)):
        imgs[i] = shift_image_horizontally(imgs[i], shift_nums[i])
    return imgs
    
def get_subfolders(directory):
    subfolders = []
    for root, dirs, files in os.walk(directory):
        # 判断只获取当前文件夹的子文件夹
        if root == directory:
            subfolders = [d for d in dirs if not d.startswith('.')]
            break
    return subfolders

def get_train_and_test(base_directory):
    train_dir = os.path.join(base_directory, 'train')
    test_dir = os.path.join(base_directory, 'test')

    if os.path.exists(train_dir) and os.path.isdir(train_dir):
        train_subfolders = sorted(get_subfolders(train_dir))
    else:
        raise FileNotFoundError("Train folder does not exist.")
    
    if os.path.exists(test_dir) and os.path.isdir(test_dir):
        test_subfolders = sorted(get_subfolders(test_dir))  # 按名称排序

    else:
        raise FileNotFoundError("Test folder does not exist.")
    train_all=[]
    for train_filepath in train_subfolders:
        train_dir = os.path.join(base_directory, 'train')
        train_filepath=os.path.join(train_dir,train_filepath)
        train_all.append(train_filepath)
    test_all=[]
    for test_filepath in test_subfolders:
        test_dir = os.path.join(base_directory, 'test')
        test_filepath=os.path.join(test_dir,test_filepath)
        test_all.append(test_filepath)
    return train_all,test_all
def check_each_folder(train_list):
    for train_path in train_list:
        gt_filepath=os.path.join(train_path, 'gt/gt.txt')
        img_filepath=os.path.join(train_path, 'img1')
        #检查目录是否存在
        if os.path.exists(gt_filepath) and os.path.exists(img_filepath):
            pass
        else:
            raise FileNotFoundError(f"{gt_filepath} or {img_filepath} not exist.")
        #检查img_filepath文件夹内是否有600张jpg，从000000.jpg到000599.jpg
        img_list=os.listdir(img_filepath)
        img_list.sort()
        with open(gt_filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.count(',')!=8:
                    raise FileNotFoundError(f"{gt_filepath} have error format.")





class CustomDataset(Dataset):
    def __init__(self, data_dir, frame_num=8,track_length=50 ,train=True):
        self.data_dir = data_dir
        self.frame_num = frame_num  # 将 frame_num 赋值给 self.frame_num
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.labels = {f: int(f.split('_')[0]) for f in self.image_files}  # 假设文件名格式为 "label_xxx.jpg"
        self.height = 128
        self.width = 512
        self.track_length = track_length #最大追踪几个物体

        base_directory =data_dir  
        train_list,test_list=get_train_and_test(base_directory)
        check_each_folder(train_list)
        check_each_folder(test_list)
        self.train_img_list=[]
        self.gt_list=[]
        max_num2 = 0
        max_gt2 = 0
        if train==False:
            train_list=test_list
        for train_path in train_list:
            gt_filepath=os.path.join(train_path, 'gt/gt.txt')
            max_consecutive = 0  # 记录最大连续相同数字出现的次数
            max_gt = None  # 记录最大连续相同数字



            with open(gt_filepath, 'r') as f:
                lines = f.readlines()
                
                current_consecutive = 1  # 当前连续相同数字的计数，初始为 1，因为第一行已经算作一个连续数字
                previous_number = None  # 记录前一行的第一个数字
                max_num=None
                for line in lines:
                    numbers = line.split(',')
                    first_number = int(float(numbers[0]))  # 获取第一个数字

                    # 如果当前数字和前一个数字相同，增加当前连续计数
                    if first_number == previous_number:
                        current_consecutive += 1
                    else:
                        # 如果数字不同，重置连续计数为 1
                        current_consecutive = 1

                    # 更新最大连续次数
                    if current_consecutive > max_consecutive:
                        max_consecutive = current_consecutive
                        max_gt=gt_filepath
                        max_num=first_number
                    # 更新前一个数字
                    previous_number = first_number

                    now_num = int(float(numbers[1]))
                    if now_num > max_num2:
                        max_num2 = now_num
                        max_gt2=gt_filepath

            img_filepath=os.path.join(train_path, 'img1')
            #读取train_path文件夹下00张jpg，从000000.jpg到000599.jpg，存在img_list中,确保img_list是按照顺序排列的，且只有.jpg文件
            img_list=os.listdir(img_filepath)
            img_list.sort()
            img_list=[os.path.join(img_filepath,img) for img in img_list if img.endswith('.jpg')]
            self.train_img_list.append(img_list)
            self.gt_list.append(gt_filepath)
        # print(f"{max_gt}: {max_consecutive}:{max_num}")
        # print(f"{max_gt2}: {max_num2}")
        self.num_samples = len(self.image_files)  # 添加这一行

    def __len__(self):
        sum_data=0
        for i in self.train_img_list:
            sum_data+=len(i)//self.frame_num
        return sum_data  # 确保返回正确的样本数量

    def __getitem__(self, idx):



        # idx=7*42+29+43
        video_idx=(idx)*self.frame_num
        now_video_idx=0 #在第几个600frame
        now_video_num=len(self.train_img_list[now_video_idx]) #在第几个frame
        while video_idx>=0:
            if now_video_num>=video_idx+self.frame_num:
                now_video_num=len(self.train_img_list[now_video_idx])-now_video_num+video_idx
                break
            else:
                if now_video_idx==len(self.train_img_list)-1:
                    raise FileNotFoundError(f"{idx} out!")
                video_idx=video_idx-int(len(self.train_img_list[now_video_idx])/self.frame_num)*self.frame_num
                now_video_idx=now_video_idx+1
                now_video_num=len(self.train_img_list[now_video_idx])
        # print(now_video_idx,now_video_num)
        now_imglist=[]
        for i in range(self.frame_num):
            now_imglist.append(self.train_img_list[now_video_idx][now_video_num+i])
        # 存储每张图片的 (height, width)
        image_sizes = []

        # 遍历图片路径，获取每张图片的高度和宽度
        for img_path in now_imglist:
            with Image.open(img_path) as img:
                width, height = img.size  # 获取宽度和高度
                image_sizes.append((height, width))
        gt=[]
 #'/root/autodl-tmp/mydata/train/20241010152940420-2/gt/gt.txt'  14 120
        # print(self.gt_list[now_video_idx])
        #读取self.gt_list[now_video_idx]中的每行开头是now_video_num～now_video_num+self.frame_num-1的行
        with open(self.gt_list[now_video_idx], 'r') as f:
            lines = f.readlines()
            i=0
            a = -999 
            while a <= now_video_num + self.frame_num-1 and i < len(lines):
                line = lines[i].strip()  # 去掉每行末尾的\n'0,1,1411.48,207.70084033613443,100.49000000000001,259.49243697478994,1,1,1.0'
                numbers = line.split(',')  # 用逗号分割
                now_gt = [float(num) if '.' in num else int(num)  for num in numbers]
                #如果now_gt[1]不是-1，就加入gt中

                if now_gt[0] <= now_video_num + self.frame_num-1 and now_gt[0] >= now_video_num and now_gt[1] != -1:
                    if now_gt[1] <= 0:
                        raise FileNotFoundError(f"{self.gt_list[now_video_idx]} have error format.")
                    gt.append(now_gt)
                a = now_gt[0]
                i += 1

        # 提取每个子列表的第二个数值
        second_values = [sublist[1] for sublist in gt]

        # 统计每个第二个数值的出现次数
        count = Counter(second_values)
        sorted_count = sorted(count.items(), key=lambda x: (-x[1], x[0]))
        # print(sorted_count)
        # 获取出现次数最多的前self.track_length个数值
        if len(sorted_count) < self.track_length:
            top_n = [item[0] for item in sorted_count]
            for i in range(self.track_length - len(sorted_count)):
                top_n.append(0)#不存在填充0
        else:
            top_n = [item[0] for item in sorted_count[:self.track_length]]
        color_tensor = torch.tensor(color_list)
        color_tensor = color_tensor[torch.randperm(color_tensor.size(0))]
        # print(top_n)  # 输出：[2, 3, 4]     
        top_n_dict = {value: idx for idx, value in enumerate(top_n)}
        #如果top_n_dict中有的key是0，删除这个key
        if 0 in top_n_dict:
            del top_n_dict[0]
        cnt_n_dict = {key: 0 for key in top_n_dict}

        extracted_rows = []
        for row in gt:
            if row[1] in top_n:
                extracted_rows.append(row)
        # 按照 top_n 的顺序对 extracted_rows 中的行排序
        extracted_rows_sorted = sorted(extracted_rows, key=lambda row: (top_n.index(row[1]), row[0]))
        # for now_gt in extracted_rows_sorted:
        #     print(now_gt)
        #创建一个torch，维度是self.frame_num,self.track_length,7
        padding_bboxes_with_color = torch.zeros(self.frame_num, self.track_length, 7)
        for gt_useful in extracted_rows_sorted:
            if len(gt_useful) != 9:
                # print(self.train_img_list[now_video_idx])
                print("ssbsbbsbs")
                for i in gt_useful:
                    print(i)
                print(f"{self.gt_list[now_video_idx]} have error format.\n{gt_useful} \n {len(gt_useful)}")
                exit()
            now_num=gt_useful[1]#物体的id
            now_cnt= cnt_n_dict[now_num] 
            now_sort=top_n_dict.get(now_num)#物体的排序编号,第几个物体
            cnt_n_dict[now_num] +=1
            # padding_bboxes_with_color[now_cnt,now_sort,:4]=torch.tensor(gt_useful[2:6], dtype=torch.float32) 

            if now_cnt < padding_bboxes_with_color.shape[0] and now_sort < padding_bboxes_with_color.shape[1]:
                padding_bboxes_with_color[now_cnt, now_sort, :4] = torch.tensor(gt_useful[2:6], dtype=torch.float32)
            else:
                print(f"Index out of bounds: now_cnt={now_cnt}, now_sort={now_sort}")
        padding_bboxes_with_color[:,:,4:]= color_tensor[:self.track_length, :]
        # for i in range(self.track_length):
        #     print(padding_bboxes_with_color[:,i,:7])
        padding_bboxes_with_color[:,:,3]=padding_bboxes_with_color[:,:,1]+padding_bboxes_with_color[:,:,3]
        padding_bboxes_with_color[:,:,2]=padding_bboxes_with_color[:,:,0]+padding_bboxes_with_color[:,:,2]

        padding_bboxes_with_color[:,:,0]=padding_bboxes_with_color[:,:,0]/2048
        padding_bboxes_with_color[:,:,1]=padding_bboxes_with_color[:,:,1]/512
        padding_bboxes_with_color[:,:,2]=padding_bboxes_with_color[:,:,2]/2048
        padding_bboxes_with_color[:,:,3]=padding_bboxes_with_color[:,:,3]/512
        #把now_imglist里的图片变成torch.Size([ n, 3, height, width])
        ori_image=load_and_stack_images(now_imglist, self.height, self.width)
        ori_image_backup=ori_image/127.5-1
        ori_image = draw_bboxes_on_image(ori_image, padding_bboxes_with_color, (self.height, self.width))
        ori_image = [to_tensor(img) for img in ori_image]
        ori_image = torch.stack(ori_image) * 2.0 - 1.0
        text_prompt = 'A segment of multi-object tracking video.'
        # video_key = data['data_samples'].img_path[0].split('/')[-2]
        # if self.caption_data is not None:
        #     text_prompt = self.caption_data.get(video_key, "")
        #定义一个张量，维度为self.frame_num,self.track_length
        mask = torch.ones(self.frame_num, self.track_length)
        # 判断 padding_bboxes_with_color 每一行的前四个数字是否全部为零
        condition = torch.all(padding_bboxes_with_color[..., :4] == 0, dim=-1)
        # 如果满足前四个数字全为零的条件，就将对应位置的 mask 设置为 0，否则为 1
        mask[condition] = 0
        #####imu
        with open(self.gt_list[now_video_idx].replace("gt.txt","y_normal.txt"), 'r') as f:
            imu_list = [float(line.strip()) for line in f]
        imu_data = torch.tensor(imu_list, dtype=torch.float32)[now_video_num:now_video_num+self.frame_num]
        #######################旋转
        # random_list = [0  for _ in range(ori_image_backup.shape[0])] #
        # random_list[0]=384
        # sum_list = [0 for _ in range(ori_image_backup.shape[0])]
        # for i in range(len(random_list)-1):
        #     sum_list[i+1] = random_list[i] + sum_list[i]
        # sum_list[0]=384
        # ori_image_backup=shift_all(ori_image_backup,sum_list)
        # ori_image=shift_all(ori_image,sum_list)
        # for i in range(padding_bboxes_with_color.shape[0]):
        #     for j in range(padding_bboxes_with_color.shape[1]):
        #         if padding_bboxes_with_color[i, j, 0] == 0 and padding_bboxes_with_color[i, j, 1] == 0 and padding_bboxes_with_color[i, j, 2] == 0 and padding_bboxes_with_color[i, j, 3] == 0:
        #             continue
        #         padding_bboxes_with_color[i][j][0] = sum_list[i]/512+ padding_bboxes_with_color[i][j][0]  # x1
        #         if padding_bboxes_with_color[i][j][0]>=1:
        #             padding_bboxes_with_color[i][j][0] = padding_bboxes_with_color[i][j][0] - 1
        #         if padding_bboxes_with_color[i][j][0]<0:
        #             padding_bboxes_with_color[i][j][0] = padding_bboxes_with_color[i][j][0] + 1
        #         padding_bboxes_with_color[i][j][2] = sum_list[i]/512+ padding_bboxes_with_color[i][j][2]  # x2
        #         if padding_bboxes_with_color[i][j][2]>=1:
        #             padding_bboxes_with_color[i][j][2] = padding_bboxes_with_color[i][j][2] - 1
        #         if padding_bboxes_with_color[i][j][2]<0:
        #             padding_bboxes_with_color[i][j][2] = padding_bboxes_with_color[i][j][2] + 1
        #         if padding_bboxes_with_color[i][j][0]>padding_bboxes_with_color[i][j][2]:
        #             if 1-padding_bboxes_with_color[i][j][0]> padding_bboxes_with_color[i][j][2]:
        #                 padding_bboxes_with_color[i][j][2]=511/512
        #             else:
        #                 padding_bboxes_with_color[i][j][0]=0
        #######################旋转

        return {
            "ori_pixel": ori_image_backup,
            "pixel_values": ori_image,
            "text_prompt": text_prompt,
            'bboxes': padding_bboxes_with_color,
            'mask': mask,
            "imu":imu_data,
            "path":now_imglist,
        }


def save_images(tensor, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    tensor = (tensor + 1) * 127.5
    tensor = tensor.clamp(0, 255).byte()
    for i in range(tensor.shape[1]):
        img = tensor[0, i].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img)
        img.save(os.path.join(output_dir, f"{prefix}_frame_{i}.png"))
        
def main():

    # 自定义数据集和数据加载器
    data_dir = './mydata'
    num_workers = 14
    seed = 42  # 设置随机种子
    random.seed(seed)  # 设置随机种子
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    train_dataset = CustomDataset(data_dir, frame_num=num_workers,train=True,track_length=50)  # 传递 num_workers 和 seed 参数
    # numm=525+52+74+75+75*6+7
    # train_dataset.__getitem__(numm)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        
        num_workers=num_workers,
    )
    print(len(train_dataloader))

    for step, batch in enumerate(train_dataloader):
        save_images(batch['ori_pixel'], "/root/autodl-tmp/TrackDiffusion-main/TrackDiffusion-SVD/outputs", f"step_{step}")
        save_images_from_tensor(batch['pixel_values'], save_dir="/root/autodl-tmp/TrackDiffusion-main/TrackDiffusion-SVD/outputs", prefix="output_image")
        ori_pixel_values = batch["ori_pixel"]
        bboxes_prompt = batch["bboxes"]

        draw_bboxes_and_save(ori_pixel_values, bboxes_prompt, "/root/autodl-tmp/TrackDiffusion-main/TrackDiffusion-SVD/outputs")

        print(step)
        break
        pass
 

    # 其他训练代码...
if __name__ == "__main__":
    main()

