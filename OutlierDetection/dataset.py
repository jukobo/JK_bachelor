

def generate_dataset(trainloader, no):
    
    ## Generate a dataset from a trainloader

    if no > 3*len(trainloader): 
        return print(f'Not enough data')
        

    dataset = []
    for i, (x, y, z) in enumerate(trainloader):

        dataset.append(x[0][0,64,:,:].unsqueeze(dim=0))

        if i+1 == no:
            break


    for j in range(len(trainloader)):
        dataset.append(x[0][0,50,:,:].unsqueeze(dim=0))

        if i+j+2 == no:
            break


    for k in range(len(trainloader)):
        dataset.append(x[0][0,80,:,:].unsqueeze(dim=0))

        if i+j+k+3 == no:
            break

    return dataset

