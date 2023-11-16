from torchvision.transforms.transforms import ColorJitter, RandomRotation, RandomVerticalFlip
from utils import *
from config import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as F
import pathlib
from torchvision.io import read_image
import numpy as np 
import cv2

# create dataset class
class knifeDataset(Dataset):
    def __init__(self, images_df, mode="train"):
        valid_indices = []
        for idx, row in images_df.iterrows():
            filename = str(row.Id)
            im = cv2.imread(filename)
            if im is not None:
                valid_indices.append(idx)

        self.images_df = images_df.iloc[valid_indices].copy()
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):
        X, fname = self.read_images(index)
        if self.mode != "test":
            labels = self.images_df.iloc[index].Label
        else:
            labels = None  # 或者是您想要的默认标签值
            # 注意：y = str(self.images_df.iloc[index].Id.absolute()) 这行代码未被使用，您可能需要调整这部分逻辑

        if self.mode == "train":
            # 训练模式的转换
            X = T.Compose([T.ToPILImage(),
                    T.Resize((config.img_weight, config.img_height)),
                    T.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
                    T.RandomRotation(degrees=(0, 180)),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)
        elif self.mode == "val":
            # 验证模式的转换
            X = T.Compose([T.ToPILImage(),
                    T.Resize((config.img_weight, config.img_height)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)
        return X.float(), labels, fname

    def read_images(self, index):
        row = self.images_df.iloc[index]
        filename = str(row.Id)
        im = cv2.imread(filename)[:, :, ::-1]
        return im, filename



