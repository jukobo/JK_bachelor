import numpy as np
import torch
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




def generate_dataset_outlier(trainloader, no):
    ## Generate a dataset with outliers from a trainloader
    radius = 30
    dataset = []
    
    for i, (x, y, z) in enumerate(trainloader):
        image = x[0][0, 64, :, :]
        # Create a meshgrid of indices
        h, w = image.shape[0], image.shape[1]
        y_indices, x_indices = torch.meshgrid(torch.arange(h), torch.arange(w))
        
        # Calculate distance from the center
        center_x, center_y = w // 2, h // 2
        dist = torch.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        
        # Add an outlier as a sphere
        outlier_mask = dist < radius
        image[outlier_mask] = torch.mean(image)
        
        dataset.append(image.unsqueeze(dim=0))

    for j in range(len(trainloader)):
        image = x[0][0, 50, :, :]
        # Create a meshgrid of indices
        h, w = image.shape[0], image.shape[1]
        y_indices, x_indices = torch.meshgrid(torch.arange(h), torch.arange(w))
        
        # Calculate distance from the center
        center_x, center_y = w // 2, h // 2
        dist = torch.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        
        # Add an outlier as a sphere
        outlier_mask = dist < radius
        image[outlier_mask] = torch.mean(image)
        
        dataset.append(image.unsqueeze(dim=0))

        if len(dataset) == no:
            return dataset

    for k in range(len(trainloader)):
        image = x[0][0, 80, :, :]
        # Create a meshgrid of indices
        h, w = image.shape[0], image.shape[1]
        y_indices, x_indices = torch.meshgrid(torch.arange(h), torch.arange(w))
        
        # Calculate distance from the center
        center_x, center_y = w // 2, h // 2
        dist = torch.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        
        # Add an outlier as a sphere
        outlier_mask = dist < radius
        image[outlier_mask] = torch.mean(image)
        
        dataset.append(image.unsqueeze(dim=0))

        if len(dataset) == no:
            return dataset

    return dataset
