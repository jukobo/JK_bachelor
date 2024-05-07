
import os
import nibabel as nib
import nibabel.orientations as nio
from data_utilities_combined import *
import pickle
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
import pickle
from copy import deepcopy
from tqdm import tqdm
import SimpleITK as sitk
from scipy import ndimage
from scipy.ndimage import distance_transform_edt


#######################################################
#################### CONTROL PANEL ####################
#######################################################
#Define scans
all_scans = 1 #Set to 1 if you want to preprocess all scans
# list_of_subjects = ['sub-verse813'] #List of subjects 521, 820
sigma = 5 #Parameter for heatmaps

new_dim = (128,128,96)
#######################################################
#######################################################
#######################################################

#Define directories
student_id = 's214725'
Type = 'test'

### GPU CLUSTER ###
dir_data = f'/scratch/{student_id}/Data/Verse20/Verse20_{Type}_unpacked' #Overall folder. Defining specific below #'/scratch/s174197/data/Verse20/Verse20_test_unpacked' #'/Users/andreasaspe/Documents/Data/Verse20/Verse20_test_unpacked'
#ONLY FOR SAVING TIME AND GETTING THE RIGHT SUBJECTS
dir_data_localisation = f'/scratch/{student_id}/Data/Verse20/Verse20_{Type}_unpacked' #list of images, rember to remove files, that spinetools haven't used
#Outputs
Output_folder = f'/scratch/{student_id}/Data/Verse20/Outlier_detection/crops_{Type}_prep'
#Padding
Padding_output_directory = f'/scratch/{student_id}/Verse20/Outlier_detection/Padding_specifications'
Padding_output_filename = f'pad_{Type}' 



#Define rescale and reorientation parameters
New_orientation = ('L', 'A', 'S')
New_voxel_size = 0.5 # [mm]

#Preprocessing
HU_range_normalize = [-1, 1]
HU_range_cutoff = [-200, 1200] #Define HU range. Only takes effekt if HU_cutoff is set to 1
#######################################################
#######################################################
#######################################################


#Define list of scans
if all_scans:
    all_subjects = []
    for filename in listdir(dir_data_localisation):
        subject = filename.split("_")[0]
        #if subject.find('verse') != -1: #PLOTTER KUN VERSE. IKKE GL
        all_subjects.append(subject)
    all_subjects = np.unique(all_subjects)
    #Sorterer fil '.DS' fra
    all_subjects = all_subjects[all_subjects != '.DS']
else:
    all_subjects = list_of_subjects

print(all_subjects)
padding_specifications = {}

#Create folders for saving data
img_path = os.path.join(Output_folder,'img') #Create output-folders if it does not exist
heatmap_path = os.path.join(Output_folder,'heatmaps') #Create output-folders if it does not exist
msk_path = os.path.join(Output_folder,'msk') #Create output-folders if it does not exist
if not os.path.exists(img_path):
    os.makedirs(img_path)
if not os.path.exists(heatmap_path):
    os.makedirs(heatmap_path)
if not os.path.exists(msk_path):
    os.makedirs(msk_path)


for subject in tqdm(all_subjects):
    #LOAD IMAGE
    filename_img = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('img.nii.gz'))][0]
    img_nib = nib.load(os.path.join(dir_data,filename_img))

    #LOAD MASK
    filename_msk = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('msk.nii.gz'))][0]
    msk_nib = nib.load(os.path.join(dir_data,filename_msk))

    #LOAD CENTROIDS
    filename_ctd = [f for f in listdir(dir_data) if (f.startswith(subject) and f.endswith('json'))][0]
    ctd_list = load_centroids(os.path.join(os.path.join(dir_data,filename_ctd)))

    #Get info
    zooms = img_nib.header.get_zooms() #Voxel sizes
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine)) #Image orientation
    ctd_code = ctd_list[0] #Centroid orientation
    data_shape_voxels = img_nib.header.get_data_shape() #Shape of data
    data_shape_mm = np.array(data_shape_voxels)*np.array(zooms) #Data measures
    data_type = img_nib.header.get_data_dtype() #Data type


    #Gaussian smoothing
    #Get data
    data_img = np.asanyarray(img_nib.dataobj, dtype=img_nib.dataobj.dtype)
    #Smooth
    sigma_smooth = [0.75/zooms[0],0.75/zooms[1],0.75/zooms[2]]
    data_img = gaussian_filter(data_img, sigma=sigma_smooth)
    #Save as Nifti file
    img_nib = nib.Nifti1Image(data_img, img_nib.affine)

    #RESAMPLE AND REORIENT
    vs = (New_voxel_size,New_voxel_size,New_voxel_size)
    #Image
    img_resampled = resample_nib(img_nib, voxel_spacing=vs, order=3)
    img_resampled_reoriented = reorient_to(img_resampled, axcodes_to=New_orientation)
    #Mask
    msk_resampled = resample_nib(msk_nib, voxel_spacing=vs, order=0) # or resample based on img: resample_mask_to(msk_nib, img_iso)
    msk_resampled_reoriented = reorient_to(msk_resampled, axcodes_to=New_orientation)
    #Centroids
    ctd_resampled = rescale_centroids(ctd_list, img_nib, vs)
    ctd_resampled_reoriented = reorient_centroids_to(ctd_resampled, img_resampled_reoriented)

    #Load data
    data_img = np.asanyarray(img_resampled_reoriented.dataobj, dtype=img_resampled_reoriented.dataobj.dtype)
    data_msk = np.asanyarray(msk_resampled_reoriented.dataobj, dtype=msk_resampled_reoriented.dataobj.dtype)

    #Change hounsfield units
    data_img[data_img<HU_range_cutoff[0]] = HU_range_cutoff[0]
    data_img[data_img>HU_range_cutoff[1]] = HU_range_cutoff[1]

    #Normalize HU
    data_img = (HU_range_normalize[1]-HU_range_normalize[0])*(data_img - data_img.min()) / (data_img.max() - data_img.min()) + HU_range_normalize[0]


    #Crop image and mask based on centroids!
    for ctd in ctd_resampled_reoriented[1:]:
        if ctd[0] == 20:
            x = np.round(ctd[1]).astype(int)
            y = np.round(ctd[2]).astype(int)
            z = np.round(ctd[3]).astype(int)
            
            centroid = (x,y,z)

            #Crop image and mask
            data_img_temp, restrictions = center_and_pad(data=data_img, new_dim=new_dim, pad_value=-1,centroid=centroid)
            data_msk_temp, restrictions = center_and_pad(data=data_msk, new_dim=new_dim, pad_value=-1,centroid=centroid)
            #Extract values
            x_min_restrict, _, y_min_restrict, _, z_min_restrict, _ = restrictions

            #Remove all other masks than the relevant one and convert to binary
            data_msk_temp = np.where(data_msk_temp == ctd[0],1,0)
            

            subject_ID = subject + '-' + str(ctd[0])
            padding_specifications.update({subject_ID: restrictions}) #Burde jeg sige minus her?

            #Apply transformation to centroid coordinates (cropping and padding)
            x+=x_min_restrict #PLUS, because we are applying changes. Not reverting.
            y+=y_min_restrict #PLUS, because we are applying changes. Not reverting.
            z+=z_min_restrict #PLUS, because we are applying changes. Not reverting.

            #Generate heatmap
            origins = (x,y,z) #Convert for cropping and padding
            meshgrid_dim = new_dim
            heatmap = gaussian_kernel_3d_new(origins,meshgrid_dim,gamma = 1,sigma = sigma)
            #Normalize
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            #Thresholding
            heatmap[heatmap < 0.001] = 0

            #Define filenames and save data
            img_filename = subject_ID + "_img.npy" #Input
            heatmap_filename = subject_ID + "_heatmap.npy" #Input
            msk_filename = subject_ID + "_msk.npy" #Target
            np.save(os.path.join(img_path,img_filename), data_img_temp)
            np.save(os.path.join(heatmap_path,heatmap_filename), heatmap)
            np.save(os.path.join(msk_path,msk_filename), data_msk_temp)


#Save padding-directory
if not os.path.exists(Padding_output_directory): #Create the directory if it does not exist
    os.makedirs(Padding_output_directory)
with open(os.path.join(Padding_output_directory,Padding_output_filename), 'wb') as f:
    pickle.dump(padding_specifications, f)
