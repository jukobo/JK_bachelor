import numpy as np
import matplotlib.pyplot as plt
from VAE import *

no = 900
file = f'C:/Users/julie/OneDrive/Skrivebord/Bachelor/JK_bachelor/OutlierDetection/rec_data/rec_img_step900_batchsize1_lr1e-05_wd0.0005.npy'
x = np.load(file)
print(x.shape)
# print(x[:,64,:].shape)

# plt.imshow(x[:,64,:], cmap='gray')
# plt.show()

