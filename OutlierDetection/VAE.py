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
    # def __init__(self, dropout):
    def __init__(self, dim, device=device): # dim is a list with the dimensions of input, hidden and latent space
        super(VAE, self).__init__()

        # #Average pooling
        # self.avgpool = nn.AvgPool3d(kernel_size=2,stride=2)

# Inspo fra https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
    
        # Define dimensions
        input_dim = dim[0]
        hidden_dim = dim[1]
        latent_dim = dim[2]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        # Latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.var_layer = nn.Linear(latent_dim, 2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encoder(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        var = self.var_layer(x)
        return mean, var
    
    def reparameterize(self, mean, var):
        eps = torch.randn_like(var).to(device)
        z = mean + var*eps
        return z
    
    def decoder(self, z):
        return self.decoder(z)



    def forward(self, image):
        mean, var = self.encoder(image)
        z = self.reparameterize(mean, var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mean, var
    
## Define model and optimizer

if __name__ == "__main__":
    image = torch.rand((1,3,96,96,128)) #1 batch, 3 inputs, 96x96, depth 128
    model = VAE(0.0)
    print(model)
    #Call model
    print(model(image).shape)
