import os
import pandas as pd
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class MFpicDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B') 
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))   
        self.B_size = len(self.B_paths) 

        btoA = self.opt.direction == 'BtoA'
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc     
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        index_B = index % self.B_size
        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path).convert('RGB')
        B = self.transform_B(B_img)

        return {'B': B, 'B_paths': B_path}

    def __len__(self):
        return self.B_size
