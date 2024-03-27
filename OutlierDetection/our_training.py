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



#gpu-cluster
#Training
img_dir_training = 'scratch/s214725/Data/Verse20/VertebraeSegmentation/Verse20_test_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir_training = 'scratch/s214725/Data/Verse20/VertebraeSegmentation/Verse20_test_prep/heatmaps'
msk_dir_training = 'scratch/s214725/Data/Verse20/VertebraeSegmentation/Verse20_test_prep/msk'

#Validation
img_dir_validation ='scratch/s214725/Data/Verse20/VertebraeSegmentation/Verse20_validation_prep/img'
heatmap_dir_validation = 'scratch/s214725/Data/Verse20/VertebraeSegmentation/Verse20_validation_prep/heatmaps'
msk_dir_validation = 'scratch/s214725/Data/Verse20/VertebraeSegmentation/Verse20_validation_prep/msk'
run_name = 'outlierdec' 

#Checkpoint
checkpoint_dir = 'scratch/s214725/Data/Checkpoints/Outlierdetection'


#Create checkpoint parent folder if it does not exist
os.makedirs(checkpoint_dir, exist_ok=True)


#Unpack parameters
num_epochs = parameters_dict['epochs']
lr = parameters_dict['learning_rate']
wd = parameters_dict['weight_decay']
batch_size = parameters_dict['batch_size']
dropout = parameters_dict['dropout']
transform = parameters_dict['transform']


#Load data
VerSe_train = LoadData(img_dir=img_dir_training, heatmap_dir=heatmap_dir_training, msk_dir = msk_dir_training,transform=transform)
train_loader = DataLoader(VerSe_train, batch_size=batch_size,
                        shuffle=True, num_workers=0) #SET TO True!
#Evaluation
VerSe_train_EVAL = LoadData(img_dir=img_dir_training, heatmap_dir=heatmap_dir_training, msk_dir = msk_dir_training)
train_loader_EVAL = DataLoader(VerSe_train_EVAL, batch_size=batch_size,
                        shuffle=True, num_workers=0) #SET TO True! - Random evaluation
VerSe_val = LoadData(img_dir=img_dir_validation, heatmap_dir=heatmap_dir_validation, msk_dir = msk_dir_validation)
val_loader = DataLoader(VerSe_val, batch_size=batch_size,
                        shuffle=True, num_workers=0) #SET TO True! - Random evaluation




#Define model
model = VAE(dropout).double()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model.to(device)

#Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss() #????
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=wd) 





train_loss = []
val_loss = []

step = -1

#Train loop
for epoch in range(num_epochs):

    print("Epoch nr.: "+str(epoch))
    train_loss_batches = []
    
    #Train model
    model.train()

    #Batching through data
    for inputs, targets, subject in tqdm(train_loader): #tqdm(train_loader)

        #Send to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass, compute gradients, perform one training step.
        output = model(inputs) #Output
        loss = criterion(output, targets)
        optimizer.zero_grad() #Clean up gradients
        loss.backward() #Compute gradients
        optimizer.step () #Clean up gradients

        #Update step
        step+=1

        #Do evaluation every 50 step
        if step%50 == 0:
            print("EVALUATION!")
            model.eval() #Set to evaluation

            #Training evaluation
            train_loss_eval = []
            with torch.no_grad():
                for i in range(5): #10 random batches
                    inputs, targets, _  = next(iter(train_loader_EVAL))
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    loss = criterion(output, targets)
                    
                    # Save loss
                    train_loss_eval.append(loss.item())
            avg_loss_train = np.mean(train_loss_eval)
            print("Train loss: "+str(avg_loss_train))
            train_loss.append(avg_loss_train)

            #Training evaluation
            val_loss_eval = []
            with torch.no_grad():
                for i in range(5): #10 random batches
                    inputs, targets, _  = next(iter(val_loader))
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    loss = criterion(output, targets)
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
           

