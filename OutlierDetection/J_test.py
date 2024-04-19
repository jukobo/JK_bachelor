import numpy as np
import matplotlib.pyplot as plt
from VAE import *

no = 9500
file = f'OutlierDetection/rec_data2/reconstruction{no}.npy'
# file = f'OutlierDetection/rec_data2/reconstruction0.npy'
x = np.load(file)
print(x.shape)
# print(x[:,64,:].shape)

plt.imshow(x[0,:,:], cmap='gray')
plt.title(f'Reconstructed Image, {no} epochs')
plt.show()

