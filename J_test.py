
output = []

## Simple test
a = 2+2
b = 3*a

output.append(a)
output.append(b)

## Data test
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import scipy
import elasticdeform


class LoadData(Dataset):

    def __init__(self, img_dir, heatmap_dir, msk_dir, transform=None):
        self.img_dir = img_dir
        self.heatmap_dir = heatmap_dir
        self.msk_dir = msk_dir
        self.images = os.listdir(img_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.images[index])
        heatmap_path = os.path.join(self.heatmap_dir,self.images[index].replace('img.npy','heatmap.npy'))
        msk_path = os.path.join(self.msk_dir,self.images[index].replace('img.npy','msk.npy'))

        #INPUTS
        #Load image
        img = np.load(img_path)
        #Load heatmap
        heatmap = np.load(heatmap_path)

        #TARGET
        #Load mask
        msk = np.load(msk_path)
        msk = msk.astype(np.float64)
                
        #Augmentation
        if self.transform is not None:
            self.elastic = 0
            self.rotation = 0
            #Test cases
            if self.transform == 'elastic':
                #Probability for elastic deformation
                p_elastic = 0.5
                self.elastic = np.random.choice(np.arange(2), p=[ 1-p_elastic , p_elastic])    
            
            if self.transform == 'rotation':
                #Probability of doing rotiation    
                p_rotation = 0.5
                self.rotation = np.random.choice(np.arange(2), p=[ 1-p_rotation , p_rotation])
            
            if self.transform == 'both':
                #Probability for elastic deformation
                p_elastic = 0.5
                self.elastic = np.random.choice(np.arange(2), p=[ 1-p_elastic , p_elastic])
                #Probability of doing rotiation    
                p_rotation = 0.5 
                self.rotation = np.random.choice(np.arange(2), p=[ 1-p_rotation , p_rotation])
            
            #Perform
            if self.elastic:
                # print("Elastic deformation is performed")
                [img, msk] = elasticdeform.deform_random_grid([img, msk], sigma=2, points=6, cval=-1, order=[3, 0])
            if self.rotation:
                # print("Rotation is performed")
                #Probability for rotation
                angle1 = random.uniform(-15,15)
                angle2 = random.uniform(-15,15)
                angle3 = random.uniform(-15,15)
                img = scipy.ndimage.rotate(img, angle1, order=3, axes=(1,2), cval=-1, reshape=False)
                msk = scipy.ndimage.rotate(msk, angle1, order=0, axes=(1,2), cval=0, reshape=False)
                heatmap = scipy.ndimage.rotate(heatmap, angle1, order=3, axes=(1,2), cval=0, reshape=False)
                
                img = scipy.ndimage.rotate(img, angle2, order=3, axes=(0,2), cval=-1, reshape=False)
                msk = scipy.ndimage.rotate(msk, angle2, order=0, axes=(0,2), cval=0, reshape=False)
                heatmap = scipy.ndimage.rotate(heatmap, angle2, order=3, axes=(0,2), cval=0, reshape=False)

                img = scipy.ndimage.rotate(img, angle3, order=3, axes=(0,1), cval=-1, reshape=False)
                msk = scipy.ndimage.rotate(msk, angle3, order=0, axes=(0,1), cval=0, reshape=False)
                heatmap = scipy.ndimage.rotate(heatmap, angle3, order=3, axes=(0,1), cval=0, reshape=False)

        #Input tensor
        inputs = np.stack((img, heatmap), axis=3)
        #Reshape
        inputs = np.moveaxis(inputs, -1, 0)
        inputs = inputs.astype(np.float64)
                
        #Add one dimension to mask to set #channels = 1
        msk = np.expand_dims(msk, axis=0)

        #Convert to tensor
        inputs = torch.from_numpy(inputs)
        msk = torch.from_numpy(msk)

        subject = self.images[index].split("_")[0]

        return inputs, msk, subject

img_dir_training = 'Data/Verse20/VertebraeSegmentation/Verse20_test_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir_training = 'Data/Verse20/VertebraeSegmentation/Verse20_test_prep/heatmaps'
msk_dir_training = 'Data/Verse20/VertebraeSegmentation/Verse20_test_prep/msk'

#Define paramters
parameters_dict = {
    'epochs': 3000,
    'learning_rate': 1e-5, #1e-5, # 1e-8
    'weight_decay': 5e-4,
    'batch_size': 1,
    'dropout': 0.0,
    'transform': None
}

transform = parameters_dict['transform']
VerSe_train = LoadData(img_dir=img_dir_training, heatmap_dir=heatmap_dir_training, msk_dir = msk_dir_training,transform=transform)

c = "Data loaded"
output.append(c)

