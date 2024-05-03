import numpy as np
import matplotlib.pyplot as plt
import torch
from our_VAE import *

# no = 1900
# file = f'OutlierDetection/rec_data3/reconstruction{no}.npy'
# file2 = f'OutlierDetection/rec_data3/reconstruction{no-100}.npy'
# # file = f'OutlierDetection/rec_data2/reconstruction0.npy'
# x = np.load(file)
# x2 = np.load(file2)
# print(x.shape)
# # print(x[:,64,:].shape)

# plt.imshow(x[0,:,:], cmap='gray')
# plt.title(f'Reconstructed Image, {no} epochs')
# plt.show()


# x_tensor = torch.from_numpy(x)
# print(x_tensor.shape)
# x2_tensor = torch.from_numpy(x2)
# print(x2_tensor.shape)

# print(loss_function(x_tensor, x2_tensor))


## Plotting
file_loss = 'C:/Users/julie/OneDrive/Skrivebord/Bachelor/JK_bachelor/o_loss.npy'
file_val_loss = 'C:/Users/julie/OneDrive/Skrivebord/Bachelor/JK_bachelor/Val_loss.npy'

o_loss = np.load(file_loss)
val_loss = np.load(file_val_loss)
# print(o_loss.shape, val_loss.shape)
print(f"Converged towards {np.mean(o_loss[500:])}")

fig, ax = plt.subplots()
ax.set_title(f'Model Loss, batch_size=20, lr=0.001, wd=0.0005')
ax.set_xlabel('Epoch')
ax.set_ylabel('Avg. loss')
ax.set_xticks(np.arange(0, len(o_loss), step= 500))

ax.plot(list(range(1, len(o_loss)+1, 1)), o_loss, label='Training loss', color='b')  # Update the plot with the current loss
# ax.plot(list(range(51, len(o_loss)+1, 48)), val_loss[:-1], label='Validation loss', color='r')
ax.plot(list(range(49, len(o_loss), 50)), val_loss, label='Validation loss', color='r')

ax.legend()
plt.show()
fig.savefig('model_loss_conv_bs20_L.png')  # Save the plot as a PNG file

