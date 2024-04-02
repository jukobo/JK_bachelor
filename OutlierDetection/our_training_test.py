import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

from VAE import *


img = Image.open("crop3.png")
img_1 = img.convert('L')
img_1 = np.array(img_1)
img_1 = np.resize(img_1, (256, 256))

transform = transforms.ToTensor()

# Apply the transformation to the image
img_gray = transform(img_1)

clas = torch.tensor([[1.0,0.0]]) # 1 for normal, 0 for outlier
# print(clas.size())

#Define paramters
parameters_dict = {
    'epochs': 10,
    'learning_rate': 1e-5,
    'weight_decay': 5e-4,
    'batch_size': 1,
    'dropout': 0.0,
    'transform': None
}


## Unpack parameters
num_epochs = parameters_dict['epochs']
lr = parameters_dict['learning_rate']
wd = parameters_dict['weight_decay']
batch_size = parameters_dict['batch_size']
dropout = parameters_dict['dropout']
transform = parameters_dict['transform']



## Define model
# model = VAE(dropout).double()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = AE([img_1.shape[0], 800, 300, 200]).to(device) #NOTE insert dimensions here
print(model)

optimizer = optim.Adam(model.parameters(), lr=lr)

train_loss = []
val_loss = []
step = -1


## Train model
def train(model, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):

        overall_loss = 0

        for batch_idx, x in enumerate(img_gray): #NOTE insert data loader here
            x = x.to(device)

            optimizer.zero_grad()

            x_reconstructed, x_classified = model(x)
            loss_re = loss_function_re(x_reconstructed, x)
            loss_cla = loss_function_cla(x_classified, clas)
            print(f"shapes = {x_classified.shape}, {clas.shape}")
            print(f"Reconstruction loss = {loss_re},", f"Classification loss = {loss_cla}")

            loss = 0.5*loss_re + 0.5*loss_cla
            print(loss)

            overall_loss += loss.item()

            loss. backward()
            optimizer.step()
    
            # Update step
            step+=1

            # # Do evaluation every 50 step
            # if step%1 == 0:
            #     print("EVALUATION!")
            #     model.eval() #Set to evaluation

            #     #Training evaluation
            #     train_loss_eval = []
            #     with torch.no_grad():
            #         for i in range(5):
            #             inputs, _ , _  = next(iter(train_loader_EVAL))
            #             inputs = inputs.to(device)
            #             output, mu, var = model(inputs)
            #             loss = loss_function(x, output, mu, var)
                        
            #             # Save loss
            #             train_loss_eval.append(loss.item())
            #     avg_loss_train = np.mean(train_loss_eval)
            #     print("Train loss: "+str(avg_loss_train))
            #     train_loss.append(avg_loss_train)

            #     #Training evaluation
            #     val_loss_eval = []
            #     with torch.no_grad():
            #         for i in range(5): #10 random batches
            #             inputs, _ , _  = next(iter(val_loader))
            #             inputs = inputs.to(device)
            #             output, mu, var = model(inputs)
            #             loss = loss_function(x, x_reconst, mu, var)
            #             # Save loss
            #             val_loss_eval.append(loss.item())
            #     avg_loss_val = np.mean(val_loss_eval)
            #     print("Validation loss: "+str(avg_loss_val))
            #     val_loss.append(avg_loss_val)

            #     #Save checkpoint
            #     checkpoint = {
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'epoch': epoch,
            #         'train_loss': train_loss,
            #         'val_loss': val_loss,
            #         'parameters_dict': parameters_dict,
            #         'run_name': run_name,
            #         'transform': transform
            #     }
            #     torch.save(checkpoint, os.path.join(checkpoint_dir,str(run_name)+'_step'+str(step)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))




        # print(f'Epoch {epoch+1}, Average loss: {overall_loss/len(train_loader)}')
        print(f'Epoch {epoch+1}, Average loss: {overall_loss/batch_idx*batch_size}')
    
        
    return overall_loss

train(model, optimizer, num_epochs, device=device)



