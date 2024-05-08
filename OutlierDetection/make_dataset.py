import numpy as np

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

    if no > 3*len(trainloader): 
        return print(f'Not enough data')
        
    radius = 30
    dataset = []
    dataset_temp =[]
    for i, (x, y, z) in enumerate(trainloader):

        dataset_temp = x[0][0,64,:,:].unsqueeze(dim=0)
        mean = np.mean(dataset_temp)
        center = [dataset_temp.shape[0]//2,dataset_temp.shape[1]//2]
        x_1, y_1 = np.ogrid[0:dataset_temp.shape[0], 0:dataset_temp.shape[1]]
        dist = np.sqrt((x_1 - center[0]) ** 2 + (y_1 - center[1]) ** 2)

        dataset_temp[dist < radius] = mean

        dataset.append(dataset_temp)

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