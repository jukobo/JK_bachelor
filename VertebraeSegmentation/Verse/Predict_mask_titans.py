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
import importlib
import pickle
from tqdm import tqdm
#My own documents
from my_plotting_functions import *
from my_data_utils import *
from data_utilities import *
#Import networks
from SpineLocalisationNet import SpineLocalisationNet
# from new_VertebraeLocalisationNet import Vert[[[ebraeLocalisationNet
from new_VertebraeLocalisationNet_batchnormdropout import VertebraeLocalisationNet
from predict_VertebraeSegmentationNet import VertebraeSegmentationNet
# from VertebraeSegmentationNet_batchnormdropout import VertebraeSegmentationNet

#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 1
list_of_subjects = ['sub-verse708']


# VERTEBRAE_FRACTURE_0285_SERIES0015 går den her lidt galt i SpineLocalisation?

do_batchnorm = 1
data_type = 'test'


#Define directories
dir_data_original = '/scratch/s214725/Data/Verse20/Verse20_'+data_type+'_unpacked'
dir_data_stage3 = '/scratch/s214725/Data/Verse20/VertebraeSegmentation/Verse20_'+data_type+'_prep_NOPADDING' #'/zhome/bb/f/127616/Documents/Thesis/Rawdata_training'
checkpoint_dir_stage3 = '/scratch/s214725/Data/Verse20/VertebraeSegmentation/NO_DATAAUG_step22000_batchsize1_lr1e-05_wd0.0005.pth' #FIXED_DATAAUG_rotation_step13300_batchsize1_lr1e-05_wd0.0005.pth' #only_elastic2_again_step33650_batchsize1_lr1e-05_wd0.0005.pth' #FIXED_DATAAUG_rotation_step11950_batchsize1_lr1e-05_wd0.0005.pth' #only_elastic2_again_step33650_batchsize1_lr1e-05_wd0.0005.pth' #'/scratch/s174197/data/Checkpoints/VertebraeSegmentation/Only_rotation_batchsize1_lr1e-05_wd0.0005.pth'
predictions_folder = '/scratch/s214725/Data/Verse20/Predictions_from_titans'


# ctd
dir_data_ctd = '/scratch/s214725/Data/Verse20/Verse20_test_unpacked'


#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 8 # [mm]


#######################################################
#######################################################
#######################################################


#Define rest of data folders
dir_stage3_img = os.path.join(dir_data_stage3,'img')
dir_stage3_msk = os.path.join(dir_data_stage3,'msk')
dir_stage3_heatmaps = os.path.join(dir_data_stage3,'heatmaps')



#Create outputfolders if they dont exist
if do_batchnorm == 0:
    dir_pred_stage3 = os.path.join(predictions_folder,'VertebraeSegmentation',data_type)
    dir_FULL_SEGMENTATIONS_GT = os.path.join(predictions_folder,'FULL_SEGMENTATIONS_GT_onlyelastic',data_type)
    dir_FULL_SEGMENTATIONS_before = os.path.join(predictions_folder,'FULL_SEGMENTATIONS_beforeCCA_onlyelastic',data_type)
    dir_FULL_SEGMENTATIONS_after = os.path.join(predictions_folder,'FULL_SEGMENTATIONS_afterCCA_onlyelastic',data_type)
else:
    dir_pred_stage3 = os.path.join(predictions_folder,'VertebraeSegmentation_batchnorm',data_type)
    dir_FULL_SEGMENTATIONS_GT = os.path.join(predictions_folder,'FULL_SEGMENTATIONS_batchnorm_GT_evenbetterrotation',data_type)
    dir_FULL_SEGMENTATIONS_before = os.path.join(predictions_folder,'FULL_SEGMENTATIONS_batchnorm_beforeCCA_evenbetterrotation',data_type)
    dir_FULL_SEGMENTATIONS_after = os.path.join(predictions_folder,'FULL_SEGMENTATIONS_batchnorm_afterCCA_evenbetterrotation',data_type)


if not os.path.exists(dir_pred_stage3):
    os.makedirs(dir_pred_stage3)
if not os.path.exists(dir_FULL_SEGMENTATIONS_GT):
    os.makedirs(dir_FULL_SEGMENTATIONS_GT)
if not os.path.exists(dir_FULL_SEGMENTATIONS_before):
    os.makedirs(dir_FULL_SEGMENTATIONS_before)
if not os.path.exists(dir_FULL_SEGMENTATIONS_after):
    os.makedirs(dir_FULL_SEGMENTATIONS_after)


#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_stage3_img):
        subject = filename.split("_")[0]
        #if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    #Sorterer fil '.DS' fra
    all_subjects = all_subjects[all_subjects != '.DS']
else:
    all_subjects = list_of_subjects


goon = False
for subject in tqdm(all_subjects):  
    print(subject)


    ################ Load original image ################
    filename_img = [f for f in listdir(dir_data_original) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    img_nib_original = nib.load(os.path.join(dir_data_original,filename_img))
    img_nib_original = reorient_to(img_nib_original, axcodes_to=New_orientation) #REORIENT


    #LOAD CENTROIDS
    filename_ctd = [f for f in listdir(dir_data_ctd) if (f.startswith(subject) and f.endswith('json'))][0]
    ctd_list = load_centroids(os.path.join(os.path.join(dir_data_ctd,filename_ctd)))

    #Define zooms and shape
    original_zooms = img_nib_original.header.get_zooms()
    original_shape = img_nib_original.header.get_data_shape()
    
    

    #############################################################
    ########################## STAGE 3 ##########################
    #############################################################
    #Load data
    filename_img = subject + "_img.nii.gz"
    img_nib = nib.load(os.path.join(dir_stage3_img,filename_img))
    data_img = np.asanyarray(img_nib.dataobj, dtype=np.float32)
    dim1,dim2,dim3 = data_img.shape

    ########################## LOAD GT ##########################
    # LOAD FILES
    filename_msk = subject + "_msk.nii.gz"
    msk_nib = nib.load(os.path.join(dir_stage3_msk,filename_msk))
    data_msk = np.asanyarray(msk_nib.dataobj, dtype=np.float32)
    
    #Rescale centroids HVIS NOGET FUCKER OP, SÅ ER DET MÅSKE FORDI JEG SKAL SORTERE CENTROIDS I RESCALECENTROIDS_VERSE. Tror dog at jeg allerede har gjort det tidligere i pipeline.
    new_zooms = original_zooms
    old_zooms = (2,2,2)
    ctd_list = RescaleCentroids_verse(new_zooms, old_zooms, ctd_list, restrictions)
    #Rescale to (1,1,1)
    old_zooms = original_zooms
    new_zooms = (1,1,1)
    ctd_list = RescaleCentroids_verse(new_zooms, old_zooms, ctd_list)
    
  
    
    #Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_dir_stage3,map_location=device)

    #Define model
    model = VertebraeSegmentationNet(0.0)
    #Send to GPU
    model.to(device)
    # Load the saved weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    vs = original_zooms
    

    idx = 1
    full_output_image = np.zeros((dim1,dim2,dim3))
    full_output_list = []
    v_number_list = []
    for ctd in ctd_list[1:]:
        v_number = int(ctd[0])
        v_name = number_to_name[v_number]
        centroid = np.array([ctd[1],ctd[2],ctd[3]])
        full_output_temp = np.zeros((dim1,dim2,dim3))
        data_img_cropped, restrictions = center_and_pad(data=data_img, new_dim=(128,128,96), pad_value=-1,centroid=tuple(centroid))
        #Change centroid coordinate based on above cropping and padding
        x_min_restrict, _, y_min_restrict, _, z_min_restrict, _ = restrictions #OBS. x_max, y_max og z_max kan kun bruges ved bounding box!! Man skal tage trække padding og cropping fra først og SÅ kan man lægge dimensionen til. Der bliver en parantes fejl ellers.
        centroid = centroid + np.array([x_min_restrict,y_min_restrict,z_min_restrict]) #PLUS, because we are applying changes. Not reverting.

        #Create heatmap
        heatmap = gaussian_kernel_3d_new(origins=tuple(centroid), meshgrid_dim=(128,128,96), gamma=1, sigma=5)
        #Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        #Thresholding
        heatmap[heatmap < 0.001] = 0
        


        #Input tensor
        inputs = np.stack((data_img_cropped, heatmap), axis=3)
        #Reshape
        inputs = np.moveaxis(inputs, -1, 0)
        inputs = inputs.astype(np.float32)
        
               
        inputs = torch.from_numpy(inputs)
        inputs = inputs.unsqueeze(0) 
        
        #Send to device
        inputs = inputs.to(device)
        
        #Define sigmoid function
        sigmoid = nn.Sigmoid()
        
        # Set the model to evaluation mode
        model.eval() 
        with torch.no_grad():
            output = model(inputs)
            #Apply sigmoid
            predictions = sigmoid(output)
           
        #Change formatting
        predictions = predictions.squeeze()
        predictions = predictions.cpu().detach().numpy()

        #SAVE PREDICTIONS
        #Convert to nifti-file
        img_predictions = nib.Nifti1Image(predictions, img_nib.affine) #new_affine)
        nib.save(img_predictions, os.path.join(dir_pred_stage3, subject+'_prediction_'+v_name+'.nii.gz'))

        idx += 1

        ranges = [-x_min_restrict,-x_min_restrict+127,-y_min_restrict,-y_min_restrict+127,-z_min_restrict,-z_min_restrict+95] #MINUS TO EVERYTHING because we are going backwards. Not applying padding or cropping but reverting!

        
        limX1 = 0
        limX2 = 127
        limY1 = 0
        limY2 = 127
        limZ1 = 0
        limZ2 = 95
        
        if ranges[0] < 0:
            #Define offset for cropping mask
            limX1 = abs(ranges[0])
            #Set indices for full image
            ranges[0] = 0
                        
        if ranges[1] > dim1-1:
            #Define offset for cropping mask
            cut_off = ranges[1]-(dim1-1)
            limX2 = 127 - cut_off
            #Set indices for full image
            ranges[1] = dim1-1

        if ranges[2] < 0:
            #Define offset for cropping mask
            limY1 = abs(ranges[2])
            #Set indices for full image
            ranges[2] = 0
 
        if ranges[3] > dim2-1:
            #Define offset for cropping mask
            cut_off = ranges[3]-(dim2-1)
            limY2 = 127 - cut_off
            #Set indices for full image
            ranges[3] = dim2-1
            
        if ranges[4] < 0:
            #Define offset for cropping mask
            limZ1 = abs(ranges[4])
            #Set indices for full image
            ranges[4] = 0
            
        if ranges[5] > dim3-1:
            #Define offset for cropping mask
            cut_off = ranges[5]-(dim3-1)
            limZ2 = 95 - cut_off
            #Set indices for full image
            ranges[5] = dim3-1

        #Crop prediction
        predictions = predictions[limX1:limX2+1,limY1:limY2+1,limZ1:limZ2+1]
        full_output_temp[ranges[0]:ranges[1]+1,ranges[2]:ranges[3]+1,ranges[4]:ranges[5]+1] = predictions #Minus because we are transfering back!
        
        # show_mask_dim1(full_output_temp, str(idx))

        
        full_output_list.append(full_output_temp)
        v_number_list.append(v_number)

            
        idx+=1
    
    # Convert the list of arrays into a single NumPy array
    full_output_list = np.array(full_output_list)
    
    # Find the maximum values along the specified axis (axis=0 for elementwise comparison)
    max_values = np.max(full_output_list, axis=0)
    
    # Create a mask indicating where the maximum values are above 0.5
    mask = max_values > 0.5
    
    # Initialize an output array with zeros
    final_prediction = np.zeros_like(max_values)
    
    # Assign values based on the conditions
    for i in range(len(full_output_list)):
        final_prediction[np.logical_and(mask, max_values == full_output_list[i])] = v_number_list[i] #i + 1
    
    # # show_mask_dim1(final_prediction, str(idx))


    #Save segmentation
    msk_nib_pred = nib.Nifti1Image(final_prediction, img_nib.affine) #new_affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    nib.save(msk_nib_pred, os.path.join(dir_FULL_SEGMENTATIONS_before,subject+'_PREDICTIONbefore.nii.gz'))


    no_vertebrae = len(ctd_list)-1 #Fordi vi også har ('L','A','S') her

    print(subject)
    for i in range(no_vertebrae):
        v_number = int(ctd_list[i+1][0])
        final_prediction_temp = deepcopy(final_prediction)
        final_prediction_temp = np.where(final_prediction_temp==v_number,1,0)

        blobs = cc3d.connected_components(final_prediction_temp,connectivity=6) #26, 18, and 6 (3D) are allowed
        no_unique = len(np.unique(blobs))

        for label in range(no_unique):
            area = np.count_nonzero(blobs == label)
            if area < 10000:
                print(area)
                final_prediction[blobs==label] = 0

    
    data_msk_filtered = deepcopy(data_msk) #GT
    data_msk_filtered[(data_msk_filtered < 17) | (data_msk_filtered > 24)] = 0
    
    #Save segmentation
    msk_nib_pred = nib.Nifti1Image(final_prediction, img_nib.affine) #new_affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    nib.save(msk_nib_pred, os.path.join(dir_FULL_SEGMENTATIONS_after,subject+'_PREDICTIONafter.nii.gz'))
    msk_nib_GT = nib.Nifti1Image(data_msk_filtered, img_nib.affine) #new_affine) #random_affine) #img_nib.affine) #DEFINE ANOTHER AFFINE!
    nib.save(msk_nib_GT, os.path.join(dir_FULL_SEGMENTATIONS_GT,subject+'_GT.nii.gz'))

