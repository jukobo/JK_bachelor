import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


from AE_functions import conv_AE_UNet, loss_function

## Planen er at starte på evaluerigen af billederne, altså hvor vi indlæser modellerne og laver histogrammer

def load_model(model_path, dim):
    model = conv_AE_UNet(dim)
    # model = conv_AE2D(dim)
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


def evaluate_model_SSIM(model, img):
    # Load model
    rec_img = model(img)

    img_np = img.cpu().squeeze().detach().numpy()
    rec_img_np = rec_img.squeeze().detach().numpy().astype(np.float64)
  

    # Calculate error
    mssim, _  = ssim(img_np, rec_img_np, full=True) # full=True returns the full structural similarity image

    return mssim


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
    plt.title('Histogram over MSE in reconstruction of images')
    plt.legend()

    # Displaying the plot
    plt.show()

    return hist_values1, hist_values2, bins1, bins2


def plot_histograms_SSIM(ssim_healthy, ssim_outlier, no_bins):

    mean_healthy = np.mean(ssim_healthy)
    std_healthy = np.std(ssim_healthy)

    mean_outlier = np.mean(ssim_outlier)
    std_outlier = np.std(ssim_outlier)


    # Calculate the range for bins based on each set of losses
    max_value1 = max(ssim_healthy)
    min_value1 = min(ssim_healthy)
    bin_range1 = (max_value1 - min_value1) / no_bins
    bins1 = [min_value1 + i * bin_range1 for i in range(no_bins+1)]

    max_value2 = max(ssim_outlier)
    min_value2 = min(ssim_outlier)
    bin_range2 = (max_value2 - min_value2) / no_bins
    bins2 = [min_value2 + i * bin_range2 for i in range(no_bins+1)]

    # Saving the histogram values for both sets of losses
    hist_values1, _ = np.histogram(ssim_healthy, bins=bins1)
    hist_values2, _ = np.histogram(ssim_outlier, bins=bins2)


    # Plotting the histogram of two losses
    # Plotting the histogram for the first set of losses (blue)
    plt.hist(ssim_healthy, bins=bins1, color='blue', edgecolor='black', alpha=0.5, label='SSIM healthy') #, density=True

    # y1 = ((1 / (np.sqrt(2 * np.pi) * std_healthy)) * np.exp(-0.5 * (1 / std_healthy * (bins1 - mean_healthy))**2))
    # plt.plot(bins1, y1, '--', color='blue', label='Normal distribution healthy')


    # Plotting the histogram for the second set of losses (red)
    plt.hist(ssim_outlier, bins=bins2, color='red', edgecolor='black', alpha=0.5, label='SSIM outliers') #, density=True)
    
    # y2 = ((1 / (np.sqrt(2 * np.pi) * std_outlier)) * np.exp(-0.5 * (1 / std_outlier * (bins2 - mean_outlier))**2))
    # plt.plot(bins2, y2, '--', color='red', label='Normal distribution outliers')


    # Adding labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram over SSIM in reconstruction of images')
    plt.legend()

    # Displaying the plot
    plt.show()

    return hist_values1, hist_values2, bins1, bins2


#for latent space

def get_latent_representation(model, input_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    input_data = input_data.to(device)

    # Pass the input data through the encoder
    with torch.no_grad():
        latent_representation = model.encode(input_data)

    return latent_representation


def collect_latent_representations(model, data):
    latent_representations = []
    for item in data:
        input_data = item[0].unsqueeze(dim=0)
        latent_representation = get_latent_representation(model, input_data)
        latent_representations.append(latent_representation.cpu().numpy()) 
    return latent_representations