# my_datasets.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import common
import torch
import numpy as np

class mydatasets(Dataset):
    def __init__(self, root_dir):
        super(mydatasets, self).__init__()
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 添加数据增强
        self.transforms = transforms.Compose([
            transforms.Resize((60, 160)),
            transforms.Grayscale(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        # self.transforms = transforms.Compose([
        #     transforms.Resize((60, 160)),
        #     transforms.RandomRotation(10),  # 随机旋转
        #     transforms.RandomAffine(0, shear=10),  # 随机仿射变换
        #     transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # 透视变换
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        #     transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),  # 随机擦除
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5], std=[0.5])
        # ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        try:
            # 加载图像
            img = Image.open(image_path)
            img = self.transforms(img)


            # 从文件名获取标签（不含扩展名）
            filename = os.path.basename(image_path)
            image_name = image_path.split("\\")[-1]
            label_str=image_name.split("_")[0]
            labels_str = filename[:4]  # 假设文件名的前四位是正确的答案
            # 验证标签长度
            if len(label_str) != common.captcha_size:
                raise ValueError(f"Label length must be {common.captcha_size}, got {len(label_str)}")
            
            # 将标签转换为索引序列
            label_indices = []
            for char in label_str:
                if char not in common.captcha_array:
                    raise ValueError(f"Character '{char}' not in captcha_array")
                label_indices.append(common.captcha_array.index(char))
            
            # 返回图像和标签索引
            return img, torch.tensor(label_indices, dtype=torch.long),labels_str
        
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # 返回一个空样本（在训练中应该跳过）
            return torch.zeros(1, 60, 160), torch.zeros(common.captcha_size, dtype=torch.long),labels_str
if __name__ == '__main__':

    d=mydatasets("./dataset/train")
    img,label=d[0]
    writer=SummaryWriter("logs")
    writer.add_image("img",img,1)
    print(img.shape)
    writer.close()
