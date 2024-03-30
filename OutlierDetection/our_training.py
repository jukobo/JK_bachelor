import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from VAE import *



#Define paramters
parameters_dict = {
    'epochs': 3000,
    'learning_rate': 1e-5,
    'weight_decay': 5e-4,
    'batch_size': 1,
    'dropout': 0.0,
    'transform': None
}

#Personaized path, change to right study ID
# path = 'scratch/s214704'
path = 'scratch/s214725'

#gpu-cluster
#Training
img_dir_training = os.path.join(path, 'Data/Verse20/VertebraeSegmentation/Verse20_test_prep/img')
heatmap_dir_training = os.path.join(path, 'Data/Verse20/VertebraeSegmentation/Verse20_test_prep/heatmaps')
msk_dir_training = os.path.join(path, 'Data/Verse20/VertebraeSegmentation/Verse20_test_prep/msk')
 #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'

#Validation
img_dir_validation = os.path.join(path, 'Data/Verse20/VertebraeSegmentation/Verse20_validation_prep/img')
heatmap_dir_validation = os.path.join(path, 'Data/Verse20/VertebraeSegmentation/Verse20_validation_prep/heatmaps')
msk_dir_validation = os.path.join(path, 'Data/Verse20/VertebraeSegmentation/Verse20_validation_prep/msk')
run_name = 'outlierdec' 

#Checkpoint
checkpoint_dir = os.path.join(path, 'Data/Checkpoints/Outlierdetection')


## Create checkpoint parent folder if it does not exist
os.makedirs(checkpoint_dir, exist_ok=True)


## Unpack parameters
num_epochs = parameters_dict['epochs']
lr = parameters_dict['learning_rate']
wd = parameters_dict['weight_decay']
batch_size = parameters_dict['batch_size']
dropout = parameters_dict['dropout']
transform = parameters_dict['transform']


## Loading data
VerSe_train = LoadData(img_dir=img_dir_training, heatmap_dir=heatmap_dir_training, msk_dir = msk_dir_training,transform=transform)
train_loader = DataLoader(VerSe_train, batch_size=batch_size,
                        shuffle=True, num_workers=0) #SET TO True!

VerSe_train_EVAL = LoadData(img_dir=img_dir_training, heatmap_dir=heatmap_dir_training, msk_dir = msk_dir_training)
train_loader_EVAL = DataLoader(VerSe_train_EVAL, batch_size=batch_size,
                        shuffle=True, num_workers=0) #SET TO True! - Random evaluation
VerSe_val = LoadData(img_dir=img_dir_validation, heatmap_dir=heatmap_dir_validation, msk_dir = msk_dir_validation)
val_loader = DataLoader(VerSe_val, batch_size=batch_size,
                        shuffle=True, num_workers=0) #SET TO True! - Random evaluation



## Define model
# model = VAE(dropout).double()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = VAE.to(device)
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

        for batch_idx, (x, _, _) in enumerate(train_loader):
            # x = x.view(batch_size, x_dim).to(device)
            x = x.to(device)

            optimizer.zero_grad()

            x_reconst, mu, var = model(x)
            loss = loss_function(x, x_reconst, mu, var)

            overall_loss += loss.item()

            loss. backward()
            optimizer.step()
    
            # Update step
            step+=1

            # Do evaluation every 50 step
            if step%50 == 0:
                print("EVALUATION!")
                model.eval() #Set to evaluation

                #Training evaluation
                train_loss_eval = []
                with torch.no_grad():
                    for i in range(5):
                        inputs, _ , _  = next(iter(train_loader_EVAL))
                        inputs = inputs.to(device)
                        output, mu, var = model(inputs)
                        loss = loss_function(x, output, mu, var)
                        
                        # Save loss
                        train_loss_eval.append(loss.item())
                avg_loss_train = np.mean(train_loss_eval)
                print("Train loss: "+str(avg_loss_train))
                train_loss.append(avg_loss_train)

                #Training evaluation
                val_loss_eval = []
                with torch.no_grad():
                    for i in range(5): #10 random batches
                        inputs, _ , _  = next(iter(val_loader))
                        inputs = inputs.to(device)
                        output, mu, var = model(inputs)
                        loss = loss_function(x, x_reconst, mu, var)
                        # Save loss
                        val_loss_eval.append(loss.item())
                avg_loss_val = np.mean(val_loss_eval)
                print("Validation loss: "+str(avg_loss_val))
                val_loss.append(avg_loss_val)

                #Save checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'parameters_dict': parameters_dict,
                    'run_name': run_name,
                    'transform': transform
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir,str(run_name)+'_step'+str(step)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))




        print(f'Epoch {epoch+1}, Average loss: {overall_loss/len(train_loader)}')
        print(f'Epoch {epoch+1}, Average loss: {overall_loss/batch_idx*batch_size}')
    
        

    return overall_loss

train(model, optimizer, num_epochs, device=device)



