import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np




class LoadData(Dataset):
    def __init__(self, img_dir, msk_dir, distfield_dir):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.distfield_dir = distfield_dir
        self.images = os.listdir(img_dir)

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.images[index])
        msk_path = os.path.join(self.msk_dir,self.images[index].replace('img.npy','msk.npy'))
        distfield_path = os.path.join(self.heatmap_dir,self.images[index].replace('img.npy','distfield.npy'))

        #INPUTS
        #Load image
        img = np.load(img_path)
        #Load distfield
        distfield = np.load(distfield_path)

        #Load mask
        msk = np.load(msk_path)
        msk = msk.astype(np.float64)
    
        #Input tensor
        inputs = np.stack((img, distfield), axis=3)
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



class VAE(nn.Module):
    def __init__(self, dropout):
        super(VAE, self).__init__()

        #Average pooling
        self.avgpool = nn.AvgPool3d(kernel_size=2,stride=2)


    
    def forward(self, image):
        layer1 = xxx
        x = xxx
        output = None
        return output
    


if __name__ == "__main__":
    image = torch.rand((1,3,96,96,128)) #1 batch, 3 inputs, 96x96, depth 128
    model = VAE(0.0)
    print(model)
    #Call model
    print(model(image).shape)
