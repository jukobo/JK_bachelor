import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from VAE import *

n = 10

#Define paramters
parameters_dict = {
    'epochs': 20000,
    'learning_rate': 1e-5,
    'batch_size': 1, #Noget galt når batch size ændres til mere end 1
    'weight_decay': 5e-4 #1e-6
}

## Unpack parameters
num_epochs = parameters_dict['epochs']
lr = parameters_dict['learning_rate']
batch_size = parameters_dict['batch_size']
wd = parameters_dict['weight_decay']


## Loading data
img_dir_training = "C:/Users/julie/Bachelor_data/crops_training_prep/img"
heatmap_dir_training = "C:/Users/julie/Bachelor_data/crops_training_prep/heatmaps"
msk_dir_training = "C:/Users/julie/Bachelor_data/crops_training_prep/msk"
# img_dir_training = '/scratch/s214725/Data/Verse20/VertebraeSegmentation/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_training_prep/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
# heatmap_dir_training = '/scratch/s214725/Data/Verse20/VertebraeSegmentation/Verse20_training_prep/heatmaps'
# msk_dir_training = '/scratch/s214725/Data/Verse20/VertebraeSegmentation/Verse20_training_prep/msk'


VerSe_train = LoadData(img_dir=img_dir_training, msk_dir = msk_dir_training, distfield_dir=heatmap_dir_training)
train_loader = DataLoader(VerSe_train, batch_size=batch_size, shuffle=True, num_workers=0)
    # 39 elements (images) in train_loader
    # Each element is a tuple of 3 elements: (img, heatmap, msk)
    # img: torch.Size([2, 128, 128, 96])

input_train, y, z = train_loader.dataset[n]
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
model = AE2D([96, 64, 32, 16]).double() #NOTE insert dimensions here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model.to(device)
print(model)
# exit()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
# optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

o_loss = []
train_loss = []
val_loss = []

## Train model
def train(model, optimizer, epochs, device):
    model.train()
    step = -1

    # Plotting the loss
    fig, ax = plt.subplots()
    ax.set_title(f'Model Loss, batch_size={batch_size}, lr={lr}, wd={wd}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.ion()

    for epoch in range(epochs):

        overall_loss = 0

        for batch_idx, (x, _, _) in enumerate(train_loader): #NOTE insert data loader here
            x = x.to(device)
            x = x[:, 0, 64, :, :].squeeze().reshape(batch_size, 128*96) # Dim is now 128x96

            x_reconstructed = model(x)

            loss = loss_function_re(x_reconstructed, x)
            overall_loss += loss.item()

            optimizer.zero_grad()
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
                        inputs = inputs[:, 0, 64, :, :].squeeze().reshape(batch_size, 128*96) # Dim is now 128x96

                        x_reconstructed = model(inputs)

                        loss = loss_function_re(x_reconstructed, inputs)

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
                        inputs = inputs[:, 0, 64, :, :].squeeze().reshape(batch_size, 128*96) # Dim is now 128x96

                        x_reconstructed = model(inputs)
                        loss = loss_function_re(x_reconstructed, inputs)
                        
                        # Save reconstructed images
                        numpy_array = x_reconstructed.cpu().numpy()
                        numpy_array = numpy_array.reshape(128, 96)
                        np.save(f'OutlierDetection/rec_data/reconstruction{epoch}{step}.npy', numpy_array)

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


        ax.plot(epoch + 1, overall_loss/len(train_loader), marker='o', color='b')  # Update the plot with the current loss
        plt.pause(0.1)


        print(f'Epoch {epoch+1}, Average loss: {overall_loss/len(train_loader)}')    


    plt.ioff() 
    plt.show()
    fig.savefig('model_loss_plot.png')  # Save the plot as a PNG file

    fig2, ax2 = plt.subplots()
    ax2.plot(train_loss, label='Training loss')
    ax2.plot(val_loss, label='Validation loss')
    ax2.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss')
    fig2.show()
    fig2.savefig('train_val_loss_plot.png')


def train2D(model, optimizer, epochs, device):
    model.train()
    step = -1

    for epoch in range(epochs):

        overall_loss = 0

        # for batch_idx, (x, _) in enumerate(train_loader.dataset[n]): #NOTE insert data loader here
        for batch_idx in range(1):
            x = input_train[0][64,:,:]
            x = x.to(device)
            # plt.imshow(x, cmap='gray')
            # plt.title('Original')
            # plt.show()
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
                        inputs = inputs[64, :, :] # Dim is now 128x96
                        inputs = inputs.to(device)

                        x_reconstructed = model(inputs)
                        loss = loss_function_re(x_reconstructed, inputs)
                        
                        # Save reconstructed images
                        numpy_array = x_reconstructed.cpu().numpy()
                        np.save(f'OutlierDetection/rec_data2/reconstruction{epoch}.npy', numpy_array)


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
    ax.set_xticks(np.arange(0, num_epochs, step=1000))

    ax.plot(list(range(1, num_epochs+1, 1)), o_loss, label='Training loss', color='b')  # Update the plot with the current loss
    ax.plot(list(range(500, num_epochs+1, 500)), val_loss, label='Validation loss', color='r')
    
    ax.legend()
    # plt.show()
    fig.savefig('model_loss_plot.png')  # Save the plot as a PNG file



train2D(model, optimizer, num_epochs, device=device)


