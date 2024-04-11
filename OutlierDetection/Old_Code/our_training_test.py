import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from VAE import *


clas = torch.tensor([[1.0,0.0]], dtype=torch.float64) # 1 for normal, 0 for outlier

## Loading data
img_dir_training = "C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_training_prep/img"
heatmap_dir_training = "C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_training_prep/heatmaps"
msk_dir_training = "C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_training_prep/msk"

VerSe_train = LoadData(img_dir=img_dir_training, msk_dir = msk_dir_training, distfield_dir=heatmap_dir_training) #, transform=transform
train_loader = DataLoader(VerSe_train, batch_size=1, shuffle=True, num_workers=0)

x, y, z = train_loader.dataset[10]
# plt.imshow(x[0][75, :, :], cmap='gray')
# plt.show()


#Validation
img_dir_validation = "C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_validation_prep/img"
heatmap_dir_validation = "C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_validation_prep/heatmaps"
msk_dir_validation = "C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_validation_prep/msk"

VerSe_train_EVAL = LoadData(img_dir=img_dir_training, msk_dir = msk_dir_training, distfield_dir=heatmap_dir_training)
train_loader_EVAL = DataLoader(VerSe_train_EVAL, batch_size=1, shuffle=True, num_workers=0) 
VerSe_val = LoadData(img_dir=img_dir_validation, msk_dir = msk_dir_validation, distfield_dir=heatmap_dir_validation)
val_loader = DataLoader(VerSe_val, batch_size=1, shuffle=True, num_workers=0) 



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
model = AE([96, 800, 300, 200]).double() #NOTE insert dimensions here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=lr)

train_loss = []
train_loss_re = []
train_loss_cla = []
val_loss = []

## Train model
def train(model, optimizer, epochs, device):
    model.train()
    step = -1

    for epoch in range(epochs):

        overall_loss = 0

        for batch_idx, (x, _, _) in enumerate(train_loader): #NOTE insert data loader here
            x = x.to(device)
            x = x[:, 0, 64, :, :].squeeze() # Dim is now 128x96

            optimizer.zero_grad()

            x_reconstructed, x_classified = model(x)
            loss_re = loss_function_re(x_reconstructed, x)
            loss_cla = loss_function_cla(x_classified, clas)
            # print(" ")
            # print(f"Classification = {x_classified},", f"True = {clas}")
            # print(f"Reconstruction loss = {loss_re},", f"Classification loss = {loss_cla}")

            loss = 0.5*loss_re + 0.5*loss_cla
            # print("")
            # print(f"Total loss = {loss}")
            # print("")

            overall_loss += loss.item()

            loss. backward()
            optimizer.step()
    
            # Update step
            step+=1

            # Do evaluation every 50 step
            if step%1 == 0:
                print("EVALUATION!")
                model.eval() #Set to evaluation

                #Training evaluation
                train_loss_eval = []
                with torch.no_grad():
                    for i in range(5):
                        inputs, _ , _  = next(iter(train_loader_EVAL))
                        inputs = inputs.to(device)
                        inputs = inputs[:, 0, 64, :, :].squeeze() # Dim is now 128x96

                        x_reconstructed, x_classified = model(x)

                        loss_re = loss_function_re(x_reconstructed, x)
                        loss_cla = loss_function_cla(x_classified, clas)
                        
                        loss = 0.5*loss_re + 0.5*loss_cla

                        # Save loss
                        train_loss_re.append(loss_re.item())
                        train_loss_cla.append(loss_cla.item())
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
                        inputs = inputs[:, 0, 64, :, :].squeeze() # Dim is now 128x96

                        x_reconstructed, x_classified = model(x)
                        loss_re = loss_function_re(x_reconstructed, x)
                        loss_cla = loss_function_cla(x_classified, clas)
                        
                        loss = 0.5*loss_re + 0.5*loss_cla

                        # Save loss
                        val_loss_eval.append(loss.item())
                avg_loss_val = np.mean(val_loss_eval)
                print("Validation loss: "+str(avg_loss_val))
                val_loss.append(avg_loss_val)

                # #Save checkpoint
                # checkpoint = {
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'epoch': epoch,
                #     'train_loss': train_loss,
                #     'val_loss': val_loss,
                #     'parameters_dict': parameters_dict,
                #     'run_name': run_name,
                #     'transform': transform
                # }
                # torch.save(checkpoint, os.path.join(checkpoint_dir,str(run_name)+'_step'+str(step)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))




        print(f'Epoch {epoch+1}, Average loss: {overall_loss/len(train_loader)}')
        print(f'Epoch {epoch+1}, Average loss: {overall_loss/batch_idx*batch_size}')
    
        
    return overall_loss

train(model, optimizer, num_epochs, device=device)

