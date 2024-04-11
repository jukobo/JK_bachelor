import numpy as np
import matplotlib.pyplot as plt
from VAE import *

<<<<<<< HEAD
no = 750
# file = f'C:/Users/julie/OneDrive/Skrivebord/Bachelor/JK_bachelor/OutlierDetection/rec_data/reconstruction{no}.npy'
file = f'/scratch/s214704/Data/Checkpoints/VertebraeSegmentation/rec_img/rec_img_step{no}_batchsize1_lr1e-05_wd0.0005.npy'
=======
no = 900
file = f'C:/Users/julie/OneDrive/Skrivebord/Bachelor/JK_bachelor/OutlierDetection/rec_data/rec_img_step900_batchsize1_lr1e-05_wd0.0005.npy'
>>>>>>> e2487ac355987bdb1e70b1b5fd6a0e16fad84e18
x = np.load(file)
print(x.shape)
# print(x[:,64,:].shape)

# plt.imshow(x[:,64,:], cmap='gray')
# plt.show()

