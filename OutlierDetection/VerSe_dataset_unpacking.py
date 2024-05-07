#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:04:46 2023

@author: s174197
"""

from os import listdir
import os
import shutil

student_id = 's214725'
Type = 'test'
    
#Input directories
dir_rawdata = f'/scratch/{student_id}/Data/Verse20/dataset-verse20{Type}/rawdata' 
dir_derivatives = f'/scratch/{student_id}/Data/Verse20/dataset-verse20{Type}/derivatives' 
#Output directories (Will create a folder if it does not exist)
dir_destination = f'/scratch/{student_id}/Data/Verse20/Verse20_{Type}_unpacked'

#Define list of scans
scans = [f for f in listdir(dir_rawdata) if f.startswith('sub-v')] #Remove file .DS_Store and sub-gl
#Create training folder if it does not exist
if not os.path.exists(dir_destination):
   os.makedirs(dir_destination)

#FOR LOOP START
for subject in scans:
    print("       SUBJECT: "+str(subject)+"\n")
    try:
        # Define file names
        filename_img = [f for f in listdir(os.path.join(dir_rawdata,subject)) if f.endswith('.gz')][0]
        filename_msk = [f for f in listdir(os.path.join(dir_derivatives,subject)) if f.endswith('.gz')][0]
        filename_ctd = [f for f in listdir(os.path.join(dir_derivatives,subject)) if f.endswith('subreg_ctd.json')][0]
        # Get directory of source
        source_img = os.path.join(dir_rawdata,subject,filename_img)
        source_msk = os.path.join(dir_derivatives,subject,filename_msk)
        source_ctd = os.path.join(dir_derivatives,subject,filename_ctd)
        # Get new file names for image
        ending = 'ct.nii.gz'
        new_ending = 'img.nii.gz'
        name = filename_img[:-len(ending)]
        new_filename_img = name + new_ending
        destination_dir_img = os.path.join(dir_destination,new_filename_img)
        # Get final directory of destination
        dir_destination_img = os.path.join(dir_destination,new_filename_img)
        dir_destination_msk = os.path.join(dir_destination,filename_msk)
        dir_destination_ctd = os.path.join(dir_destination,filename_ctd)
        # Copy files instead of moving
        shutil.copy(source_img, dir_destination_img)
        shutil.copy(source_msk, dir_destination_msk)
        shutil.copy(source_ctd, dir_destination_ctd)
        print("Subject "+str(subject)+" has been copied.")
    except:
        print("Subject "+str(subject)+" has already been copied.")


#Deleting files with sub-gl
scans_g = [f for f in listdir(dir_rawdata) if f.startswith('sub-gl')] 

for subject in scans_g:
    try:
        # Define file paths
        file_path_img = os.path.join(dir_rawdata, subject)
        file_path_msk = os.path.join(dir_derivatives, subject)
        # Delete files
        shutil.rmtree(file_path_img)
        shutil.rmtree(file_path_msk)
        print("Files for subject "+str(subject)+" starting with 'sub-g' have been deleted.")
    except Exception as e:
        print("An error occurred while deleting files for subject "+str(subject)+":", e)