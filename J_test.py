
output = []

## Simple test
a = 2+2
b = 3*a

output.append(a)
print(output)
output.append(b)
print(output)

import numpy as np
data = np.load("C:/Users/julie/Bachelor_data/Verse20/VertebraeSegmentation/Verse20_training_prep/img/sub-verse500-18_img.npy")
print(data.shape)
