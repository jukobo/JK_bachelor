import numpy as np
import torch
import random
def generate_dataset(trainloader, no):
    
    ## Generate a dataset from a trainloader

    if no > 3*len(trainloader): 
        return print(f'Not enough data')
        

    dataset = []
    for i, (x, y, z) in enumerate(trainloader):

        dataset.append(x[0][0,64,:,:].unsqueeze(dim=0))

        if len(dataset) == no:
            return dataset


    for j in range(len(trainloader)):
        dataset.append(x[0][0,50,:,:].unsqueeze(dim=0))

        if len(dataset) == no:
            return dataset
        

    for k in range(len(trainloader)):
        dataset.append(x[0][0,80,:,:].unsqueeze(dim=0))

        if len(dataset) == no:
            return dataset

    return dataset



def create_outlier(i, image, radius):
    if i == 1:
      #Sphere  
      # Create a meshgrid of indices
        h, w = image.shape[0], image.shape[1]
        y_indices, x_indices = torch.meshgrid(torch.arange(h), torch.arange(w))
        
        # Calculate distance from the center
        center_x, center_y = w // 2, h // 2
        dist = torch.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        
        # Add an outlier as a sphere
        outlier_mask = dist < radius
        image[outlier_mask] = torch.mean(image)
    elif i == 2:
      # Square 
        h, w = image.shape[0], image.shape[1]
        center = [ h // 2, w // 2]

        start = [max(0, center[0] - radius), max(0, center[1] - radius)]
        end = [min(image.shape[0], center[0] + radius), min(image.shape[1], center[1] + radius)]
        image[start[0]:end[0], start[1]:end[1]] = torch.mean(image)
    return image


def generate_dataset_outlier(trainloader, no, radius):
    ## Generate a dataset with outliers from a trainloader
    dataset = []
    
    for i, (x, y, z) in enumerate(trainloader):
        image = x[0][0, 64, :, :]
        Type = random.randint(1, 2)
        image_out = create_outlier(Type, image, radius)
        
        dataset.append(image_out.unsqueeze(dim=0))

        if len(dataset) == no:
            return dataset

    for j in range(len(trainloader)):
        image = x[0][0, 50, :, :]
        Type = random.randint(1, 2)
        image_out = create_outlier(Type, image, radius)
        
        dataset.append(image_out.unsqueeze(dim=0))

        if len(dataset) == no:
            return dataset

    for k in range(len(trainloader)):
        image = x[0][0, 80, :, :]
        Type = random.randint(1, 2)
        image_out = create_outlier(Type, image, radius)
        
        dataset.append(image_out.unsqueeze(dim=0))

        if len(dataset) == no:
            return dataset

    return dataset

