import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from VAE import *

#Define paramters
parameters_dict = {
    'epochs': 50,
    'learning_rate': 1e-4,
    'batch_size': 1, #Noget galt når batch size ændres til mere end 1
    'weight_decay': 5e-4,
}

## Unpack parameters
num_epochs = parameters_dict['epochs']
lr = parameters_dict['learning_rate']
batch_size = parameters_dict['batch_size']
wd = parameters_dict['weight_decay']


## Loading data
img_dir_training = "C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_training_prep/img"
heatmap_dir_training = "C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_training_prep/heatmaps"
msk_dir_training = "C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_training_prep/msk"

VerSe_train = LoadData(img_dir=img_dir_training, msk_dir = msk_dir_training, distfield_dir=heatmap_dir_training)
train_loader = DataLoader(VerSe_train, batch_size=batch_size, shuffle=True, num_workers=0)

x, y, z = train_loader.dataset[10]
# plt.imshow(x[0][75, :, :], cmap='gray')
# plt.show()


#Validation
img_dir_validation = "C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_validation_prep/img"
heatmap_dir_validation = "C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_validation_prep/heatmaps"
msk_dir_validation = "C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_validation_prep/msk"

VerSe_train_EVAL = LoadData(img_dir=img_dir_training, msk_dir = msk_dir_training, distfield_dir=heatmap_dir_training)
train_loader_EVAL = DataLoader(VerSe_train_EVAL, batch_size=batch_size, shuffle=True, num_workers=0) 
VerSe_val = LoadData(img_dir=img_dir_validation, msk_dir = msk_dir_validation, distfield_dir=heatmap_dir_validation)
val_loader = DataLoader(VerSe_val, batch_size=batch_size, shuffle=True, num_workers=0) 


## Define model
model = AE2([96*128, 512, 256, 128]).double() #NOTE insert dimensions here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
# optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


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

train(model, optimizer, num_epochs, device=device)


