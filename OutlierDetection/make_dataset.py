import numpy as np
import torch
import random


def healthy_outlier(trainloader):
    dataset_healthy = []
    dataset_outlier = []

    for i, (x, y, z) in enumerate(trainloader):

        split = np.ceil(len(trainloader)/2)

        if i >= split:
            dataset_healthy.append(x[0])
        elif i < split:
            dataset_outlier.append(x[0])

    return dataset_healthy, dataset_outlier


def generate_dataset_training(trainloader, no):
    ## Generate a dataset from a trainloader

    if no > 20*len(trainloader): 
        return print(f'Not enough data')
        
    # slices = [46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84]
    slices = [50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80] #16 slices
    dataset = []
    for i, (x, y, z) in enumerate(trainloader):
        for sli in slices:
            dataset.append(x[0][0,sli,:,:].unsqueeze(dim=0))

            if len(dataset) == no:
                return dataset


    return dataset


def generate_dataset(dataset, no):
    
    ## Generate a dataset from a trainloader

    if no > 5*len(dataset): 
        return print(f'Not enough data')
        

    dataset_new = []
    for i, x in enumerate(dataset):

        dataset_new.append(x[0,64,:,:].unsqueeze(dim=0))

        if len(dataset_new) == no:
            return dataset_new


    for j, x in enumerate(dataset):
        dataset_new.append(x[0,55,:,:].unsqueeze(dim=0))

        if len(dataset_new) == no:
            return dataset_new
        

    for l, x in enumerate(dataset):
        dataset_new.append(x[0,75,:,:].unsqueeze(dim=0))

        if len(dataset_new) == no:
            return dataset_new

      
    for m, x in enumerate(dataset):
        dataset_new.append(x[0,45,:,:].unsqueeze(dim=0))

        if len(dataset_new) == no:
            return dataset_new


    for k, x in enumerate(dataset):
        dataset_new.append(x[0,85,:,:].unsqueeze(dim=0))

        if len(dataset_new) == no:
            return dataset_new


    return dataset_new



def create_outlier(i, image, radius, mode):
    if mode.lower() == "black":
        value = -1
    else:
        value = torch.mean(image)


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
        image[outlier_mask] = value
    elif i == 2:
      # Square 
        h, w = image.shape[0], image.shape[1]
        center = [ h // 2, w // 2]

        start = [max(0, center[0] - radius), max(0, center[1] - radius)]
        end = [min(image.shape[0], center[0] + radius), min(image.shape[1], center[1] + radius)]
        image[start[0]:end[0], start[1]:end[1]] = value
    return image


def generate_dataset_outlier(dataset, no, radius, mode=""):
    ## Generate a dataset with outliers from a trainloader
    dataset_new = []
    
    for i, x in enumerate(dataset):
        image = x[0, 64, :, :]
        Type = random.randint(1, 2)
        image_out = create_outlier(Type, image, radius, mode)
        
        dataset_new.append(image_out.unsqueeze(dim=0))

        if len(dataset_new) == no:
            return dataset_new

    for j, x in enumerate(dataset):
        image = x[0, 55, :, :]
        Type = random.randint(1, 2)
        image_out = create_outlier(Type, image, radius, mode)
        
        dataset_new.append(image_out.unsqueeze(dim=0))

        if len(dataset_new) == no:
            return dataset_new

    for k, x in enumerate(dataset):
        image = x[0, 75, :, :]
        Type = random.randint(1, 2)
        image_out = create_outlier(Type, image, radius,mode)
        
        dataset_new.append(image_out.unsqueeze(dim=0))

        if len(dataset_new) == no:
            return dataset_new

    for l, x in enumerate(dataset):
        image = x[0, 45, :, :]
        Type = random.randint(1, 2)
        image_out = create_outlier(Type, image, radius,mode)
        
        dataset_new.append(image_out.unsqueeze(dim=0))

        if len(dataset_new) == no:
            return dataset_new

    for m, x in enumerate(dataset):
        image = x[0, 85, :, :]
        Type = random.randint(1, 2)
        image_out = create_outlier(Type, image, radius, mode)
        
        dataset_new.append(image_out.unsqueeze(dim=0))

        if len(dataset_new) == no:
            return dataset_new


    return dataset_new




