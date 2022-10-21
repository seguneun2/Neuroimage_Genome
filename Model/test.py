import pandas as pd
import torch
import nibabel as nib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
import torchio as tio
import os
import torch.nn as nn


df = '/home/cps_lab/seungeun/MRI/Label_File_path_df.csv'
df = pd.read_csv(df, index_col=0)
df = df[(df['LABEL']=='pMCI')|(df['LABEL']=='sMCI')]
df['LABEL'] = df['LABEL'].replace({'pMCI':1, 'sMCI':0})


from dataset import MRIdatamodule
from Resnet3D import MedicalNet 



weights_path = '/home/cps_lab/seungeun/MRI/MedicalNet/pretrain/resnet_34.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device} device')

model = MedicalNet(path_to_weights=weights_path, 
                    device=device, 
                    sample_input_D=128, 
                    sample_input_H=128, 
                    sample_input_W=128)

MRI_Dataset = MRIdatamodule(df, train_ratio=0.8)
MRI_Dataset.prepare_data()
MRI_Dataset.setup()

train(model=model, Epochs=300,train_loader=MRI_Dataset.train_loader, test_loader = MRI_Dataset.test_loader)
