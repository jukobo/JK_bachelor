#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 00:42:50 2023

@author: andreasaspe
"""

#General imports
import os
import sys
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import nibabel as nib
from functools import reduce 
#My own documents
from my_plotting_functions import *
#from new_VertebraeLocalisationNet import *
from Create_dataset import *
from my_data_utils import *
from new_VertebraeLocalisationNet_batchnormdropout import *
#from new_VertebraeLocalisationNet_batchnormdropout import *


#GPU-cluster
img_dir = '/scratch/s214725/Data/Verse20/VertebraeLocalisation/Verse20_test_prep/img' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_training_prep2/img' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_prep_alldata/img' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_prep2/img' #'/scratch/s174197/data/Verse20/Verse20_training_prep/img' #'/Users/andreasaspe/Documents/Data/Verse20_training_prep/img' #r'C:\Users\PC\Documents\Andreas_s174197\Preprocessed_data\img'
heatmap_dir = '/scratch/s214725/Data/Verse20/VertebraeLocalisation/Verse20_test_heatmaps' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_training_heatmaps2' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_heatmaps_alldata' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_heatmaps2'
output_pred_dir = '/scratch/s214725/Data/Verse20/VertebraeLocalisation/Verse20_test_predictions' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/for_making_figure/output' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions_alldata' #'/scratch/s174197/data/Verse20/VertebraeLocalisation2/Verse20_validation_predictions2' #Prediction directory
#Checkpoint
checkpoint_parent_dir = '/scratch/s214725/Data/Checkpoints/VertebraeLocalisation' #/VertebraeLocalisation2/alldata' #'/scratch/s174197/data/Checkpoints'
checkpoint_filename = 'FIXED_DATAAUG_both_epoch1500_batchsize1_lr1e-05_wd0.0001.pth' #'Batchnorm_dropout_batchsize1_lr1e-05_wd0.0001.pth' #'Second_try_No_dropout_newinitialisation_batchsize1_lr1e-05_wd0.0001.pth' #'Third_try_No_dropout_newinitialisation142.pth' #'Second_try_No_dropout_newinitialisation_batchsize1_lr1e-05_wd0.0001.pth' #'NEWNetworkWithNewInitialisation_GPU_batchsize1_lr0.001_wd0.pth' #'FIXEDVAL_dropout3_higherdecay_batchsize1_lr1e-05_wd0.0001.pth' #'FIXEDVAL_dropout3_higherdecay_batchsize1_lr1e-05_wd0.0001.pth'

checkpoint_dir = os.path.join(checkpoint_parent_dir,checkpoint_filename)


#Load data
Data = LoadFullData(img_dir=img_dir,heatmap_dir = heatmap_dir)
loader = DataLoader(Data, batch_size=1,
                        shuffle=False, num_workers=0)
#Load model
model = VertebraeLocalisationNet(0.0)


    
if not os.path.exists(output_pred_dir):
            os.makedirs(output_pred_dir)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
checkpoint = torch.load(checkpoint_dir,map_location=device)


#Send to GPU!
model.to(device)
# Load the saved weights into the model
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval() 

import os
import torch
from torch.utils.data import DataLoader
from functools import reduce
import nibabel as nib

# Your existing code goes here...

with torch.no_grad():
    for i, (img, _, inputs_list, targets_list, start_end_voxels, subject) in enumerate(tqdm(loader)):
        assert len(subject) == 1  # Make sure we are only predicting one batch
        _, dim1, dim2, dim3 = img.shape
        outputs_list = []

        for j in range(len(inputs_list)):
            # Unpack targets and inputs and get predictions
            inputs = inputs_list[j]
            # Send to device
            inputs = inputs.to(device)
            # Forward pass
            output, _, _ = model(inputs)  # output, local, spatial = model(inputs)

            # Get start and end voxel
            start_voxel = start_end_voxels[j][0].item()
            end_voxel = start_end_voxels[j][1].item()

            # Put into the output_tensor
            output_nifti = nib.Nifti1Image(output.cpu().numpy()[0, 0, :, :, start_voxel:end_voxel + 1], affine=None)
            nib.save(output_nifti, os.path.join(output_pred_dir, f"{subject[0]}_heatmap_pred.nii.gz"))

