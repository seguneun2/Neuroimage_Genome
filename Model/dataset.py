import pandas as pd
import torch
import matplotlib.pyplot as plt 
import torchio as tio

# df = '/home/cps_lab/seungeun/MRI/Label_File_path_df.csv'
# df = pd.read_csv(df, index_col=0)
# df = df[(df['LABEL']=='pMCI')|(df['LABEL']=='sMCI')]


class MRIdatamodule(torch.utils.data.Dataset):
    """

    init :: dataframe 
    1. prepare_data()
    2. setup()
            
    input dataframe

    -----------------------------
             LABEL | Image_path | cropped(path)
    sub_id1       
    sub_id2 

    """

    def __init__(self, df, train_ratio, batch_size = 4):
        super().__init__()
        self.image_paths = df['cropped'].values
        self.labels =df['LABEL'].values

        self.train_ratio = train_ratio
        self.subjects = None
        
        self.preprocess = None

        self.transform = None

        self.train_set = None
        self.test_set = None

        self.batch_size = batch_size
    
    def get_max_shape(self, subjects):
        print("get_max_shape ")
        import numpy as np
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        print(shapes.max(axis=0)) 
        return shapes.max(axis=0)


    def prepare_data(self):
        """
        subject별 image, label 취합, dataset 으로 
        """
        image_paths, labels  = self.image_paths, self.labels

        self.subjects = []
        for (image_path, pt_label) in zip(image_paths, labels):
            subject = tio.Subject(
                mri=tio.ScalarImage(image_path),
                label=torch.FloatTensor([pt_label]),
            )
            self.subjects.append(subject)
            
    def get_preprocessing_transform(self):
        ### apply both train, test 
        preprocess = tio.Compose([
            tio.RescaleIntensity((0, 1)),  ############# INTENSIRT RESCAKING ::::  DO MUST !!!!!!!!!!!!!!!!!!!!!!!
           # tio.CropOrPad(128),
        ])
        return preprocess
    
    def get_augmentation_transform(self):
        ### apply to train  ---> 필요한가??? 
        augment = tio.Compose([
            tio.RandomAffine(),
            tio.RandomGamma(p=0.5),
            tio.RandomNoise(p=0.5),
            tio.RandomMotion(p=0.1),
            tio.RandomBiasField(p=0.25),
        ])
        return augment

    def setup(self, stage=None):
        from torch.utils.data import random_split
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_ratio))
        num_test_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_test_subjects
        
        train_subjects, test_subjects = random_split(self.subjects, splits)
        
        preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()

        transform = tio.Compose([preprocess,  augment])  ######## augment 어떻게,,????????


        self.transform = tio.Compose([preprocess, augment])
        self.train_set = tio.SubjectsDataset(train_subjects, transform=transform)  ### train_set-> augument 추가하게 하기
        self.test_set = tio.SubjectsDataset(test_subjects, transform=preprocess)
        print("create dataset")
        print('TrainDataset size:', len(self.train_set), 'subjects')
        print('TestDataset size:', len(self.test_set), 'subjects')

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size= self.batch_size)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size)
        print("create dataloader")
