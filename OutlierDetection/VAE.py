import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import os




class LoadData(Dataset):
    def __init__(self, img_dir, msk_dir, distfield_dir):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.distfield_dir = distfield_dir
        self.images = os.listdir(img_dir)

        
    def __len__(self):
        return len(self.images)
    



class VAE(nn.Module):
    def __init__(self, dropout):
        super(VAE, self).__init__()


    
    def forward(self, image):
        output = None
        return output
    


if __name__ == "__main__":
    image = torch.rand((1,3,96,96,128)) #1 batch, 3 inputs, 96x96, depth 128
    model = VAE(0.0)
    print(model)
    #Call model
    print(model(image).shape)
