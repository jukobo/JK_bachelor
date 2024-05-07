import torch
import numpy as np
import matplotlib.pyplot as plt

from our_VAE import conv_AE_UNet, loss_function

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
    
    # Calculate error
    mse = loss_function(rec_img, img)

    return mse.item()



def plot_histograms(loss_healthy, loss_outlier, no_bins):

    # Plotting the histogram of one loss
    # plt.hist(losses, bins=no_bins, color='blue', edgecolor='black')

    # Plotting the histogram of two losses
        # Plotting the histogram for the first set of losses (blue)
    plt.hist(loss_healthy, bins=no_bins, color='blue', edgecolor='black', alpha=0.5, label='Losses healthy')

    # Plotting the histogram for the second set of losses (red)
    plt.hist(loss_outlier, bins=no_bins, color='red', edgecolor='black', alpha=0.5, label='Losses outliers')


    # Adding labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram over errors in reconstruction of images')

    # Displaying the plot
    plt.show()

    return 

def plot_histogrgrams2(losses, no_bins):
    hist_values, bin_edges = np.histogram(losses, bins=no_bins)

    # Plotting the histogram
    plt.bar(bin_edges[:-1], hist_values, width=np.diff(bin_edges), color='blue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram over errors in reconstruction of images')

    # Displaying the plot
    plt.show()

    return hist_values, bin_edges
