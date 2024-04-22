#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 00:42:50 2023

@author: andreasaspe
"""

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








