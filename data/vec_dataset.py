import os
import torch
import pandas as pd
import random
from data.base_dataset import BaseDataset


class VecDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A.csv')  
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B.csv')  

        self.A = self._load_data_from_excel(self.dir_A)
        self.B = self._load_data_from_excel(self.dir_B)

        self.A_size = len(self.A) 
        self.B_size = len(self.B)  
        self.data_shape = self._get_vector_size()

    def _load_data_from_excel(self, excel_file):
        df = pd.read_csv(excel_file, header=None)
        data = df.values.tolist()
        data = torch.tensor(data)
        return data

    def _get_vector_size(self):
        return (list(self.A[0].shape), list(self.B[0].shape))
    
    def __getitem__(self, index):
        index_A = index % self.A_size  
        if self.opt.serial_batches:   
            index_B = index % self.B_size
        else:  
            index_B = random.randint(0, self.B_size - 1)
        
        A = self.A[index_A]
        B = self.B[index_B]

        return {'A': A, 'B': B}

    def __len__(self):
        return max(self.A_size, self.B_size)
