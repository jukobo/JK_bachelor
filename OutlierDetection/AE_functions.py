import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


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
        distfield_path = os.path.join(self.distfield_dir,self.images[index].replace('img.npy','heatmap.npy'))

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
        inputs = inputs.astype(np.float32)
                
        #Add one dimension to mask to set #channels = 1
        msk = np.expand_dims(msk, axis=0)

        #Convert to tensor
        inputs = torch.from_numpy(inputs)
        msk = torch.from_numpy(msk)

        subject = self.images[index].split("_")[0]

        return inputs, msk, subject


class conv_AE_UNet(nn.Module):
    # def __init__(self, dropout):
    def __init__(self, dim, device=device): # dim is a list with the dimensions of input, hidden and latent space
        super(conv_AE_UNet, self).__init__()
    
        # Define dimensions
        input_dim = dim[0]
        hidden_dim_1 = dim[1]
        hidden_dim_2 = dim[2]
        hidden_dim_3 = dim[3]
        latent_dim = dim[4]

        kernel_size = 3
        stride = 1
        padding = 1


        # Encoder
        self.encoder = nn.Sequential(
            # input: 128x96x1
            nn.Conv2d(input_dim, hidden_dim_1, kernel_size = kernel_size, stride = stride, padding = padding), # output: 128x96x16
            nn.ReLU(inplace=True), 
            nn.Conv2d(hidden_dim_1, hidden_dim_1, kernel_size = kernel_size, stride = stride, padding = padding), # output: 128x96x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: 64x48x16
            # nn.AvgPool2d(kernel_size=2, stride=2), # output: 64x48x16

            # input: 64x48x16
            nn.Conv2d(hidden_dim_1, hidden_dim_2, kernel_size = kernel_size, stride = stride, padding = padding), # output: 64x48x32
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim_2, hidden_dim_2, kernel_size = kernel_size, stride = stride, padding = padding), # output: 64x48x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: 32x24x32
            # nn.AvgPool2d(kernel_size=2, stride=2), # output: 32x24x32


            # input: 32x24x32
            nn.Conv2d(hidden_dim_2, hidden_dim_3, kernel_size = kernel_size, stride = stride, padding = padding), # output: 32x24x64
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim_3, hidden_dim_3, kernel_size = kernel_size, stride = stride, padding = padding), # output: 32x24x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: 16x12x64
            # nn.AvgPool2d(kernel_size=2, stride=2), # output: 16x12x64


            # input: 16x12x64
            nn.Conv2d(hidden_dim_3, latent_dim, kernel_size = kernel_size, stride = stride, padding = padding), # output: 16x12x128

            # --- NOTE er i tvivl med om vis skal have dette med
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size = kernel_size, stride = stride, padding = padding), # output: 16x12x128
        )


        # Decoder
        self.decoder = nn.Sequential(

            nn.Conv2d(latent_dim, hidden_dim_3, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim_3, hidden_dim_3, kernel_size=2, stride=2), # output: 32x24x64
            # nn.ConvTranspose2d(kernel_size=2, stride=2), # output: 32x24x64
            nn.Conv2d(hidden_dim_3, hidden_dim_3, kernel_size = kernel_size, stride = stride, padding = padding), # output: 32x24x64
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim_3, hidden_dim_2, kernel_size = kernel_size, stride = stride, padding = padding), # output: 32x24x32
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim_2, hidden_dim_2, kernel_size=2, stride=2), # output: 64x48x32
            # nn.ConvTranspose2d(kernel_size=2, stride=2), # output: 64x48x32
            nn.Conv2d(hidden_dim_2, hidden_dim_2, kernel_size = kernel_size, stride = stride, padding = padding), # output: 64x48x32
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim_2, hidden_dim_1, kernel_size = kernel_size, stride = stride, padding = padding), # output: 64x48x16
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim_1, hidden_dim_1, kernel_size=2, stride=2), # output: 128x96x16
            # nn.ConvTranspose2d(kernel_size=2, stride=2), # output: 128x96x16
            nn.Conv2d(hidden_dim_1, hidden_dim_1, kernel_size = kernel_size, stride = stride, padding = padding), # output: 128x96x16
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim_1, input_dim, kernel_size = kernel_size, stride = stride, padding = padding), # output: 128x96x1
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


    def forward(self, image):
        z = self.encode(image)
        x_reconstructed = self.decode(z)

        return x_reconstructed



class conv_AE_UNet2(nn.Module):
    # def __init__(self, dropout):
    def __init__(self, dim, device=device): # dim is a list with the dimensions of input, hidden and latent space
        super(conv_AE_UNet2, self).__init__()
    
        # Define dimensions
        input_dim = dim[0]
        hidden_dim_1 = dim[1]
        hidden_dim_2 = dim[2]
        hidden_dim_3 = dim[3]
        # hidden_dim_4 = 32
        latent_dim = dim[4]

        kernel_size = 3
        stride = 1
        padding = 1


        # Encoder
        self.encoder = nn.Sequential(
            # input: 128x96x1
            nn.Conv2d(input_dim, hidden_dim_1, kernel_size = kernel_size, stride = stride, padding = padding), # output: 128x96x16
            nn.ReLU(inplace=True), 
            nn.Conv2d(hidden_dim_1, hidden_dim_1, kernel_size = kernel_size, stride = stride, padding = padding), # output: 128x96x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: 64x48x16


            # input: 64x48x16
            nn.Conv2d(hidden_dim_1, hidden_dim_2, kernel_size = kernel_size, stride = stride, padding = padding), # output: 64x48x32
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim_2, hidden_dim_2, kernel_size = kernel_size, stride = stride, padding = padding), # output: 64x48x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: 32x24x32


            # input: 32x24x32
            nn.Conv2d(hidden_dim_2, hidden_dim_3, kernel_size = kernel_size, stride = stride, padding = padding), # output: 32x24x64
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim_3, hidden_dim_3, kernel_size = kernel_size, stride = stride, padding = padding), # output: 32x24x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: 16x12x64

            nn.Dropout(p=0.2), 

            # input: 16x12x64 # NOTE LATENT SPACE
            nn.Conv2d(hidden_dim_3, latent_dim, kernel_size = kernel_size, stride = stride, padding = padding), # output: 16x12x128
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.ReLU(inplace=True) # output: 16x12x128
        )


        # Decoder
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(latent_dim, hidden_dim_3, kernel_size=2, stride=2), # output: 32x24x64
            # nn.ConvTranspose2d(kernel_size=2, stride=2), # output: 32x24x64
            nn.Conv2d(hidden_dim_3, hidden_dim_3, kernel_size = kernel_size, stride = stride, padding = padding), # output: 32x24x64
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim_3, hidden_dim_2, kernel_size = kernel_size, stride = stride, padding = padding), # output: 32x24x32
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim_2, hidden_dim_2, kernel_size=2, stride=2), # output: 64x48x32
            # nn.ConvTranspose2d(kernel_size=2, stride=2), # output: 64x48x32
            nn.Conv2d(hidden_dim_2, hidden_dim_2, kernel_size = kernel_size, stride = stride, padding = padding), # output: 64x48x32
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim_2, hidden_dim_1, kernel_size = kernel_size, stride = stride, padding = padding), # output: 64x48x16
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_dim_1, hidden_dim_1, kernel_size=2, stride=2), # output: 128x96x16
            # nn.ConvTranspose2d(kernel_size=2, stride=2), # output: 128x96x16
            nn.Conv2d(hidden_dim_1, hidden_dim_1, kernel_size = kernel_size, stride = stride, padding = padding), # output: 128x96x16
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim_1, input_dim, kernel_size = kernel_size, stride = stride, padding = padding), # output: 128x96x1
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


    def forward(self, image):
        z = self.encode(image)
        x_reconstructed = self.decode(z)

        return x_reconstructed



def loss_function(x_reconstructed, x):
    loss = nn.MSELoss()
    mse = loss(x_reconstructed, x)
    return mse



## For simple AE NN

class AE(nn.Module): # Bruges til simpel AE
    # def __init__(self, dropout):
    def __init__(self, dim, device=device): # dim is a list with the dimensions of input, hidden and latent space
        super(AE, self).__init__()
    
        # Define dimensions
        input_dim = dim[0]
        hidden_dim_1 = dim[1]
        hidden_dim_2 = dim[2]
        # hidden_dim_3 = dim[3]
        # latent_dim = dim[4]
        latent_dim = dim[3]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(), 
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, latent_dim)
            # nn.Linear(hidden_dim_2, hidden_dim_3),
            # nn.ReLU(),
            # nn.Linear(hidden_dim_3, latent_dim)
        )


        # Decoder
        self.decoder = nn.Sequential(
            # nn.Linear(latent_dim, hidden_dim_3),
            # nn.ReLU(),
            # nn.Linear(hidden_dim_3, hidden_dim_2),
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, input_dim),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


    def forward(self, image):
        z = self.encode(image)
        x_reconstructed = self.decode(z)

        return x_reconstructed



