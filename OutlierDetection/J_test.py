import numpy as np
import matplotlib.pyplot as plt
from VAE import *

no = 69500
file = f'OutlierDetection/rec_data2/reconstruction{no}.npy'
x = np.load(file)
print(x.shape)
# print(x[:,64,:].shape)

plt.imshow(x, cmap='gray')
plt.title(f'Reconstructed Image, {no} epochs')
plt.show()

