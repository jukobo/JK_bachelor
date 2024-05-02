import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
import numpy as np


from our_VAE import *

n_1 = 0
n_2 = 19

#Define paramters
parameters_dict = {
    'epochs': 4000,
    'learning_rate': 1e-3,
    'batch_size': 1, #Noget galt når batch size ændres til mere end 1
    'weight_decay': 5e-4 #1e-6
}

## Unpack parameters
num_epochs = parameters_dict['epochs']
lr = parameters_dict['learning_rate']
batch_size = parameters_dict['batch_size']
wd = parameters_dict['weight_decay']


## Loading data
study_no_data = 's214704'
study_no_save = 's214704'

# img_dir_training = "C:/Users/julie/Bachelor_data/crops_training_prep/img"
# heatmap_dir_training = "C:/Users/julie/Bachelor_data/crops_training_prep/heatmaps"
# msk_dir_training = "C:/Users/julie/Bachelor_data/crops_training_prep/msk"
img_dir_training = f"/scratch/{study_no_data}/Data/crops_training_prep/img"
heatmap_dir_training = f"/scratch/{study_no_data}/Data/crops_training_prep/heatmaps"
msk_dir_training = f"/scratch/{study_no_data}/Data/crops_training_prep/msk"

VerSe_train = LoadData(img_dir=img_dir_training, msk_dir = msk_dir_training, distfield_dir=heatmap_dir_training)
train_loader = DataLoader(VerSe_train, batch_size=batch_size, shuffle=False, num_workers=0)
    # 39 elements (images) in train_loader
    # Each element is a tuple of 3 elements: (img, heatmap, msk)
    # img: torch.Size([2, 128, 128, 96])

# input_train, y, z = train_loader.dataset[n]
# print(torch.min(input_train[0][64,:,:]), torch.max(input_train[0][64,:,:])) # min = -0.9077, max = 0.6916
# plt.imshow(input_train[0][64, :, :], cmap='gray')
# plt.show()
# exit()


## Define model
# For simple AE
model = conv_AE_UNet([1, 32, 16, 8]).double() #NOTE insert dimensions here


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model.to(device)
print(model)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

o_loss = []
train_loss = []
val_loss = []

## Train model
def train2D_conv(model, optimizer, epochs, device):
    model.train()
    step = -1

    for epoch in range(epochs):

        overall_loss = 0

        for idx, data in enumerate(train_loader):
            if idx >= n_1 and idx <= n_2:
                input_train, _, _ = data

                x = input_train[0][0,64,:,:].unsqueeze(dim=0)
                x = x.to(device)

                x_reconstructed = model(x)

                #-- Loss function
                squared_diff = (x_reconstructed - x) ** 2
                loss_temp = torch.mean(squared_diff, dim=1)
                loss = torch.mean(loss_temp, dim=1).squeeze()
                # print(type(loss), loss.shape, loss)
                

                # loss = loss_function(x_reconstructed, x)
                overall_loss += loss.item()

                optimizer.zero_grad()
                loss. backward()
                optimizer.step()
        
                # Update step
                step+=1
                print(step)

                # Do evaluation every 50 step
                if step%1000 == 0:
                    print()
                    print("EVALUATION!")
                    model.eval() #Set to evaluation

                    #Training evaluation
                    val_loss_eval = []
                    with torch.no_grad():
                        inputs, _, _ = train_loader.dataset[n_1]
                        inputs = input_train[0][0,64,:,:].unsqueeze(dim=0)

                        #-- Plotting the original image
                        #plt.imshow(inputs.squeeze(), cmap='gray')
                        #plt.title('Original')
                        #plt.show()
                        #exit()

                        inputs = inputs.to(device)

                        inputs_reconstructed = model(inputs)
                        
                        #-- Loss function
                        squared_diff = (inputs_reconstructed - inputs) ** 2
                        loss_temp = torch.mean(squared_diff, dim=1)
                        v_loss = torch.mean(loss_temp, dim=1).squeeze()
                        print(type(v_loss), v_loss.shape, v_loss)

                        
                        # Save reconstructed images
                        # numpy_array = inputs_reconstructed.cpu().numpy()
                        # np.save(f'OutlierDetection/rec_data3/reconstruction{epoch}.npy', numpy_array)
                        # np.save(f'/scratch/{study_no_save}/Data/rec_data/reconstruction{epoch}.npy', numpy_array)


                        # Save loss
                        val_loss_eval.append(v_loss.item())
                    avg_loss_val = np.mean(val_loss_eval)
                    print("Validation loss: "+str(avg_loss_val))
                    val_loss.append(avg_loss_val)

                   
            if idx == n_2:
                break

        o_loss.append(overall_loss/(n_2-n_1+1))
        if epoch%100 == 0:
            print(f'Epoch {epoch+1}, Average loss: {overall_loss/(n_2-n_1+1)}')    

        ## Save model
        if epoch == 0:
            torch.save(model.state_dict(), f'/scratch/{study_no_save}/Data/model_conv_{epoch}.pth')
            print('Model saved')
        elif epoch == epochs-1:
            torch.save(model.state_dict(), f'/scratch/{study_no_save}/Data/model_conv_{epoch}.pth')
            print('Model saved')

    # # np.save('OutlierDetection/o_loss3.npy', o_loss)
    # # np.save('OutlierDetection/val_loss3.npy', val_loss)
    np.save(f'/scratch/{study_no_save}/Data/o_loss.npy', o_loss)
    np.save(f'/scratch/{study_no_save}/Data/Val_loss.npy', val_loss)

    

train2D_conv(model, optimizer, num_epochs, device=device)

