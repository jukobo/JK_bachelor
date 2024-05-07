import torch
import numpy as np
import matplotlib.pyplot as plt

from our_VAE import conv_AE_UNet

## Planen er at starte på evaluerigen af billederne, altså hvor vi indlæser modellerne og laver histogrammer

def load_model(model_path, dim):
    model = conv_AE_UNet(dim)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    return model

def evaluate_model(model, img):
    # Load model
    rec_img = model(img)
    rec_img = rec_img.detach().numpy()
    print(rec_img.shape)

    # I = np.int8(feat_img)
    # max_val = np.int8(np.max(I.flatten()))
    max_val = np.max(rec_img.flatten())

    hist_in, bins = np.histogram(rec_img, bins=100, range = [-0.5, max_val+0.5])

    plt.bar(bins[:-1], hist_in, width = 1)
    plt.show()

    return




