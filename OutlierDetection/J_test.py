import numpy as np
import matplotlib.pyplot as plt
from VAE import *

no = 2400
file = f'C:/Users/julie/OneDrive/Skrivebord/Bachelor/JK_bachelor/OutlierDetection/rec_data/reconstruction{no}.npy'
x = np.load(file)
print(x[:,64,:].shape)

plt.imshow(x[:,64,:], cmap='gray')
plt.show()

