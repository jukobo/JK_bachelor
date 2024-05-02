# import time
# import os 
print('Script started')
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

# run_name = 'Test_AE2'
# run_name2 = 'rec_img'

# checkpoint_dir = '/scratch/s214704/Data/Checkpoints/VertebraeSegmentation/Test_AE2'
# checkpoint_dir2 = '/scratch/s214704/Data/Checkpoints/VertebraeSegmentation/rec_img'
# #Create checkpoint parent folder if it does not exist
# os.makedirs(checkpoint_dir, exist_ok=True)
# os.makedirs(checkpoint_dir2, exist_ok=True)


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

def train2D_conv_simple(model, optimizer, epochs, device):
    model.train()
    step = -1

    for epoch in range(epochs):

        overall_loss = 0

        # for batch_idx, (x, _) in enumerate(train_loader.dataset[n]): #NOTE insert data loader here
        for batch_idx in range(1):
            x = input_train[0][64,:,:].unsqueeze(dim=0)
            x = x.to(device)
            # plt.imshow(x, cmap='gray')
            # plt.title('Original')
            # plt.show()
            # print(x.shape)
            # exit()

            x_reconstructed = model(x)

            loss = loss_function_re(x_reconstructed, x)
            overall_loss += loss.item()
            o_loss.append(overall_loss)

            optimizer.zero_grad()
            loss. backward()
            optimizer.step()
    
            # Update step
            step+=1

            # Do evaluation every 50 step
            if step%500 == 0:
                print("EVALUATION!")
                model.eval() #Set to evaluation

                #Training evaluation
                # train_loss_eval = []
                # with torch.no_grad():
                    # for i in range(5):
                    # for i in range(1):
                    #     # inputs, _   = next(iter(train_loader_EVAL.dataset[n]))
                    #     inputs = input_train_EVAL[0]
                    #     inputs = inputs[64, :, :].squeeze(dim=0) # Dim is now 128x96
                    #     inputs = inputs.to(device)
                    #     # inputs = inputs.to(device)
                    #     # inputs = inputs[:, 0, :, :, :].squeeze(dim=0) # Dim is now 128x96

                    #     x_reconstructed = model(inputs)
                    #     # print(x_reconstructed.shape)

                    #     loss = loss_function_re(x_reconstructed, inputs)

                    #     # Save loss
                    #     train_loss_eval.append(loss.item())

                # avg_loss_train = np.mean(train_loss_eval)
                # print("Train loss: "+str(avg_loss_train))
                # train_loss.append(avg_loss_train)

                #Training evaluation
                val_loss_eval = []
                with torch.no_grad():
                    # for i in range(5): #10 random batches
                    for i in range(1):
                        inputs = input_train[0]
                        # inputs, _  = next(iter(val_loader.dataset[n]))
                        inputs = inputs[64, :, :].unsqueeze(dim=0) # Dim is now 128x96
                        inputs = inputs.to(device)

                        x_reconstructed = model(inputs)
                        loss = loss_function_re(x_reconstructed, inputs)
                        
                        # Save reconstructed images
                        numpy_array = x_reconstructed.cpu().numpy()
                        np.save(f'/scratch/s214725/Data/rec_data/reconstruction{epoch}.npy', numpy_array)


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
                # }
                # torch.save(checkpoint, os.path.join(checkpoint_dir,str(run_name)+'_step'+str(step)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))


        # print(f'Epoch {epoch+1}, Average loss: {overall_loss/len(train_loader)}')    



    # Plotting the loss
    fig, ax = plt.subplots()
    ax.set_title(f'Model Loss, batch_size={batch_size}, lr={lr}, wd={wd}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avg. loss')
    ax.set_xticks(np.arange(0, num_epochs, step= 1000))

    ax.plot(list(range(1, num_epochs+1, 1)), o_loss, label='Training loss', color='b')  # Update the plot with the current loss
    ax.plot(list(range(500, num_epochs+1, 500)), val_loss, label='Validation loss', color='r')
    
    ax.legend()
    # plt.show()
    fig.savefig('model_loss_conv.png')  # Save the plot as a PNG file


    # Plotting the loss
    fig2, ax2 = plt.subplots()
    ax2.set_title(f'Model Loss, batch_size={batch_size}, lr={lr}, wd={wd}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Avg. loss')
    ax2.set_xticks(np.arange(0, num_epochs, step=1000))

    ax2.plot(list(range(601, num_epochs+1, 1)), o_loss[600:], label='Training loss', color='b')  # Update the plot with the current loss
    ax2.plot(list(range(1000, num_epochs+1, 500)), val_loss[1:], label='Validation loss', color='r') ## NOTE ikke helt rigtig ved x/y-akse
    
    ax2.legend()
    # plt.show()
    fig2.savefig('model_loss_conv2.png')  # Save the plot as a PNG file


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
                loss = torch.mean(squared_diff)
                # print(type(loss), loss.shape, loss)

                # loss = loss_function(x_reconstructed, x)
                overall_loss += loss.item()

                optimizer.zero_grad()
                loss. backward()
                optimizer.step()
        
                # Update step
                step+=1

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
                        exit()

                        
                        # Save reconstructed images
                        numpy_array = inputs_reconstructed.cpu().numpy()
                        # np.save(f'OutlierDetection/rec_data3/reconstruction{epoch}.npy', numpy_array)
                        np.save(f'/scratch/{study_no_save}/Data/rec_data/reconstruction{epoch}.npy', numpy_array)


                        # Save loss
                        val_loss_eval.append(v_loss.item())
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
                    # }
                    # torch.save(checkpoint, os.path.join(checkpoint_dir,str(run_name)+'_step'+str(step)+'_batchsize'+str(batch_size)+'_lr'+str(lr)+'_wd'+str(wd)+'.pth'))
        
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

    # np.save('OutlierDetection/o_loss3.npy', o_loss)
    # np.save('OutlierDetection/val_loss3.npy', val_loss)
    np.save(f'/scratch/{study_no_save}/Data/o_loss.npy', o_loss)
    np.save(f'/scratch/{study_no_save}/Data/Val_loss.npy', val_loss)

    

train2D_conv(model, optimizer, num_epochs, device=device)





