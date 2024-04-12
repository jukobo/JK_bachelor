import numpy as np
import matplotlib.pyplot as plt
from VAE import *

no = 4900
file = f'OutlierDetection/rec_data2/reconstruction{no}{no}.npy'
x = np.load(file)
print(x.shape)
# print(x[:,64,:].shape)

plt.imshow(x, cmap='gray')
plt.show()

