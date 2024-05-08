import torch 
import torch.optim as optim
from torch.utils.data import DataLoader 
import numpy as np

from AE_functions import *
from make_dataset import *


#Define paramters
parameters_dict = {
    'epochs': 1000,
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
study_no_save = 's214725'

# img_dir_training = "C:/Users/julie/Bachelor_data/crops_training_prep/img"
# heatmap_dir_training = "C:/Users/julie/Bachelor_data/crops_training_prep/heatmaps"
# msk_dir_training = "C:/Users/julie/Bachelor_data/crops_training_prep/msk"
img_dir_training = f"/scratch/{study_no_data}/Data/crops_training_prep/img"
heatmap_dir_training = f"/scratch/{study_no_data}/Data/crops_training_prep/heatmaps"
msk_dir_training = f"/scratch/{study_no_data}/Data/crops_training_prep/msk"

VerSe_train = LoadData(img_dir=img_dir_training, msk_dir = msk_dir_training, distfield_dir=heatmap_dir_training)
train_loader = DataLoader(VerSe_train, batch_size=batch_size, shuffle=True, num_workers=0)
    # 39 elements (images) in train_loader
    # Each element is a tuple of 3 elements: (img, heatmap, msk)
    # img: torch.Size([2, 128, 128, 96])

# input_train, y, z = train_loader.dataset[-1]
# plt.imshow(input_train[0][64, :, :], cmap='gray')
# plt.title('Original')
# plt.show()
# exit()

## Generere dataset med angivet antal 2D images
n = 100
dataset = generate_dataset(train_loader, n)


# run_name = 'Test_AE2'
# run_name2 = 'rec_img'

# checkpoint_dir = '/scratch/s214704/Data/Checkpoints/VertebraeSegmentation/Test_AE2'
# checkpoint_dir2 = '/scratch/s214704/Data/Checkpoints/VertebraeSegmentation/rec_img'
# #Create checkpoint parent folder if it does not exist
# os.makedirs(checkpoint_dir, exist_ok=True)
# os.makedirs(checkpoint_dir2, exist_ok=True)


## Define model
# For simple AE
# model = conv_AE_UNet([1, 16, 32, 64, 128]) #NOTE insert dimensions here
model = conv_AE_UNet([1, 8, 16, 32, 64]) #NOTE insert dimensions here


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

        # for idx, data in enumerate(train_loader):
        #     input_train, _, _ = data

        #     x = input_train[0][0,64,:,:].unsqueeze(dim=0)
        for idx, data in enumerate(dataset):
            x = data.to(device)

            x_reconstructed = model(x)

            #-- Loss function
            squared_diff = (x_reconstructed - x) ** 2
            loss = torch.mean(squared_diff)

            overall_loss += loss.item()

            optimizer.zero_grad()
            loss. backward()
            optimizer.step()
    
            # Update step
            step+=1

            # Do evaluation every 50 epoch
            if step%2500 == 0:
                print()
                print("EVALUATION!")
                model.eval() #Set to evaluation

                #Training evaluation
                val_loss_eval = []
                with torch.no_grad():
                    # inputs, _, _ = train_loader.dataset[0]
                    # inputs = input_train[0][0,64,:,:].unsqueeze(dim=0)
                    inputs = dataset[0]

                    org_img = inputs.cpu().numpy()
                    np.save(f'/scratch/{study_no_save}/Data/rec_data/original.npy', org_img)

                    #-- Plotting the original image
                    # plt.imshow(inputs.squeeze(), cmap='gray')
                    # plt.title('Original')
                    # plt.show()
                    # exit()

                    inputs = inputs.to(device)

                    inputs_reconstructed = model(inputs)
                    
                    #-- Loss function
                    squared_diff = (inputs_reconstructed - inputs) ** 2
                    loss_temp = torch.mean(squared_diff, dim=1)
                    v_loss = torch.mean(loss_temp, dim=1).squeeze()
                    
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

        o_loss.append(overall_loss/n)
        if epoch%100 == 0:
            print(f'Epoch {epoch+1}, Average loss: {overall_loss/n}')    

        ## Save model
        if epoch == 0:
            torch.save(model.state_dict(), f'/scratch/{study_no_save}/Data/model_conv_{epoch}.pth')
            print('Model saved')
        elif epoch == epochs-1:
            torch.save(model.state_dict(), f'/scratch/{study_no_save}/Data/model_conv_{epoch}.pth')
            print('Model saved')

    # np.save('OutlierDetection/o_loss.npy', o_loss)
    # np.save('OutlierDetection/val_loss.npy', val_loss)
    np.save(f'/scratch/{study_no_save}/Data/o_loss.npy', o_loss)
    np.save(f'/scratch/{study_no_save}/Data/Val_loss.npy', val_loss)

    

train2D_conv(model, optimizer, num_epochs, device=device)





