import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



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
    

def loss_function(x, x_reconstructed, mean, var):
    loss = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction='sum'),

    # Kullback–Leibler (KL) divergence
    kl_divergence = 0.5 * torch.sum(1 + var - mean**2 - var.exp())

    return loss + kl_divergence


class AE(nn.Module):
    # def __init__(self, dropout):
    def __init__(self, dim, device=device): # dim is a list with the dimensions of input, hidden and latent space
        super(AE, self).__init__()
    
        # Define dimensions
        input_dim = dim[0]
        hidden_dim_1 = dim[1]
        hidden_dim_2 = dim[2]
        latent_dim = dim[3]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(), #NOTE True??
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, latent_dim)
        )


        # Decoder (reconstruction)
        self.decoder_re = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, input_dim),
            nn.Sigmoid(), #NOTE  Tanh??
        )

        # Decoder (classification)
        self.decoder_cla = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 150),
            nn.ReLU(),
            nn.Linear(150, 2),
            nn.Sigmoid(), #NOTE  Tanh??
        )


    def encode(self, x):
        return self.encoder(x)
    
    def decode_re(self, z):
        return self.decoder_re(z)
    
    def decode_cla(self, z):
        out_temp = self.decoder_cla(z)
        out = torch.mean(out_temp, dim=0, keepdim=True)

        return out


    def forward(self, image):
        z = self.encode(image)
        x_reconstructed = self.decode_re(z)
        x_classified = self.decode_cla(z)

        return x_reconstructed, x_classified


class AE2(nn.Module):
    # def __init__(self, dropout):
    def __init__(self, dim, device=device): # dim is a list with the dimensions of input, hidden and latent space
        super(AE2, self).__init__()
    
        # Define dimensions
        input_dim = dim[0]
        hidden_dim_1 = dim[1]
        hidden_dim_2 = dim[2]
        # hidden_dim_3 = dim[3]
        latent_dim = dim[3]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1), #3D?
            nn.ReLU(), 
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            # nn.Linear(hidden_dim_2, hidden_dim_3),
            # nn.ReLU(),
            nn.Linear(hidden_dim_2, latent_dim)
        )


        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
            # nn.Linear(hidden_dim_3, hidden_dim_2),
            # nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


    def forward(self, image):
        z = self.encode(image)
        x_reconstructed = self.decode(z)

        return x_reconstructed


class AE2D(nn.Module):
    # def __init__(self, dropout):
    def __init__(self, dim, device=device): # dim is a list with the dimensions of input, hidden and latent space
        super(AE2D, self).__init__()
    
        # Define dimensions
        input_dim = dim[0]
        hidden_dim_1 = dim[1]
        hidden_dim_2 = dim[2]
        # hidden_dim_3 = dim[3]
        latent_dim = dim[3]

        do = 0.2

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1), #3D?
            nn.Dropout2d(p=do),
            nn.ReLU(), 
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.Dropout2d(p=do),
            nn.ReLU(),
            # nn.Linear(hidden_dim_2, hidden_dim_3),
            # nn.ReLU(),
            nn.Linear(hidden_dim_2, latent_dim)
        )


        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.Dropout2d(p=do),
            nn.ReLU(),
            # nn.Linear(hidden_dim_3, hidden_dim_2),
            # nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.Dropout2d(p=do),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


    def forward(self, image):
        z = self.encode(image)
        x_reconstructed = self.decode(z)

        return x_reconstructed



def loss_function_re(x, x_reconstructed):
    criterion = nn.MSELoss() #reduction='sum'
    loss = criterion(x_reconstructed, x)

    return loss

def loss_function_cla(x, x_classified):
    loss_function_cla = nn.functional.binary_cross_entropy
    loss = loss_function_cla(x_classified, x, reduction='sum')

    return loss



def load_split_data(folder_healthy, folder_outlier):
    folder_path_healthy = folder_healthy
    file_names_healthy = os.listdir(folder_path_healthy)

    folder_path_outlier= folder_outlier
    file_names_outlier = os.listdir(folder_path_outlier)

    healthy_A, outlier_A, healthy_B, outlier_B = train_test_split(file_names_healthy, file_names_outlier, test_size=0.2, random_state=42)


    return healthy_A, outlier_B


## Function to display tensor as image
def display_tensor_as_image(tensor):
    # Detach tensor from computation graph and convert to numpy array
    tensor = tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    # Assuming tensor is a numpy array
    plt.imshow(tensor, cmap='gray')  # You can change the colormap as needed
    plt.axis('off')  # Turn off axis
    plt.show()
