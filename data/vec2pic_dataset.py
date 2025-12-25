import os
import pandas as pd
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class Vec2PicDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A.csv')  
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  

        self.A = self._load_data_from_excel(self.dir_A)
        self.A_size = len(self.A)  

        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))   
        self.B_size = len(self.B_paths) 
        btoA = self.opt.direction == 'BtoA'
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.data_shape = self._get_vector_size()

    def _load_data_from_excel(self, excel_file):
        df = pd.read_csv(excel_file, header=None)
        data = df.values.tolist()
        data = torch.tensor(data)
        return data

    def _get_vector_size(self):
        return (list(self.A[0].shape),list(self.__getitem__(0)['B'].shape)) 

    def __getitem__(self, index):
        index_A = index % self.A_size  
        if self.opt.serial_batches:   
            index_B = index % self.B_size
        else:  
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path).convert('RGB')

        A = self.A[index_A]
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
