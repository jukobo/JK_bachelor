import torch
import numpy as np
import matplotlib.pyplot as plt

from AE_functions import conv_AE_UNet, loss_function,conv_AE2D

## Planen er at starte på evaluerigen af billederne, altså hvor vi indlæser modellerne og laver histogrammer

def load_model(model_path, dim):
    # model = conv_AE_UNet(dim)
    model = conv_AE2D(dim)
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

    # Calculate the range for bins based on each set of losses
    max_value1 = max(loss_healthy)
    min_value1 = min(loss_healthy)
    bin_range1 = (max_value1 - min_value1) / no_bins
    bins1 = [min_value1 + i * bin_range1 for i in range(no_bins+1)]

    max_value2 = max(loss_outlier)
    min_value2 = min(loss_outlier)
    bin_range2 = (max_value2 - min_value2) / no_bins
    bins2 = [min_value2 + i * bin_range2 for i in range(no_bins+1)]

    # Saving the histogram values for both sets of losses
    hist_values1, _ = np.histogram(loss_healthy, bins=bins1)
    hist_values2, _ = np.histogram(loss_outlier, bins=bins2)


    # Plotting the histogram of two losses
    # Plotting the histogram for the first set of losses (blue)
    plt.hist(loss_healthy, bins=bins1, color='blue', edgecolor='black', alpha=0.5, label='Losses healthy')

    # Plotting the histogram for the second set of losses (red)
    plt.hist(loss_outlier, bins=bins2, color='red', edgecolor='black', alpha=0.5, label='Losses outliers')


    # Adding labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram over errors in reconstruction of images')
    plt.legend()

    # Displaying the plot
    plt.show()

    return hist_values1, hist_values2, bins1, bins2



def plot_histogrgrams2(losses, no_bins):
    hist_values, bin_edges = np.histogram(losses, bins=no_bins, range=(0, 0.1))

    # Plotting the histogram
    plt.bar(bin_edges[:-1], hist_values, width=np.diff(bin_edges), color='blue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram over errors in reconstruction of images')

    # Displaying the plot
    plt.show()

    return hist_values, bin_edges

