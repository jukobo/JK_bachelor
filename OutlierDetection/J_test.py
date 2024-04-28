import numpy as np
import matplotlib.pyplot as plt
from VAE import *

no = 4952
file = f'OutlierDetection/rec_data2/reconstruction{no}.npy'
# file = f'OutlierDetection/rec_data2/reconstruction0.npy'
x = np.load(file)
print(x.shape)
# print(x[:,64,:].shape)

plt.imshow(x[0,:,:], cmap='gray')
plt.title(f'Reconstructed Image, {5000} epochs')
plt.show()

# ## Plotting
# file_loss = 'OutlierDetection/o_loss.npy'
# file_val_loss = 'OutlierDetection/val_loss.npy'

# o_loss = np.load(file_loss)
# val_loss = np.load(file_val_loss)
# print(f"Converged towards {np.mean(o_loss[2000:])}")

# fig, ax = plt.subplots()
# ax.set_title(f'Model Loss, batch_size=10, lr=0.001, wd=0.0005')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Avg. loss')
# ax.set_xticks(np.arange(0, len(o_loss), step= 500))

# ax.plot(list(range(1, len(o_loss)+1, 1)), o_loss, label='Training loss', color='b')  # Update the plot with the current loss
# # ax.plot(list(range(51, len(o_loss)+1, 48)), val_loss[:-1], label='Validation loss', color='r')
# ax.plot(list(range(51, len(o_loss)+1, 45)), val_loss, label='Validation loss', color='r')

# ax.legend()
# plt.show()
# fig.savefig('model_loss_conv_bs10.png')  # Save the plot as a PNG file

