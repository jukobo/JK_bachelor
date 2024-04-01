#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:04:46 2023

@author: s174197
"""

from os import listdir
import os
import shutil

    
#Input directories
dir_rawdata = 'Data/dataset-verse20training/rawdata' 
dir_derivatives = 'Data/dataset-verse20training/derivatives' 

raw_dir_destination = 'Data/Verse20_training_unpacked_new/raw' 
msk_dir_destination = 'Data/Verse20_training_unpacked_new/msk' 
ctd_dir_destination = 'Data/Verse20_training_unpacked_new/ctd' 


#Define list of scans
scans = [f for f in listdir(dir_rawdata) if f.startswith('sub')] #Remove file .DS_Store
#Create training folder if it does not exist
if not os.path.exists(raw_dir_destination):
   os.makedirs(raw_dir_destination)
   os.makedirs(msk_dir_destination)
   os.makedirs(ctd_dir_destination)


#FOR LOOP START
for subject in scans:
    print("       SUBJECT: "+str(subject)+"\n")
    try:
        # Define file names
        filename_img = [f for f in listdir(os.path.join(dir_rawdata,subject)) if f.endswith('.gz')][0]
        filename_msk = [f for f in listdir(os.path.join(dir_derivatives,subject)) if f.endswith('.gz')][0]
        filename_ctd = [f for f in listdir(os.path.join(dir_derivatives,subject)) if f.endswith('subreg_ctd.json')][0]
        #Get directory of source
        source_img = os.path.join(dir_rawdata,subject,filename_img)
        source_msk = os.path.join(dir_derivatives,subject,filename_msk)
        source_ctd = os.path.join(dir_derivatives,subject,filename_ctd)
        #Get new file names for image
        ending = 'ct.nii.gz'
        new_ending = 'img.nii.gz'
        name = filename_img[:-len(ending)]
        new_filename_img = name + new_ending
        # destination_dir_img = os.path.join(dir_destination,new_filename_img)
        #Get final directory of destination
        dir_destination_img = os.path.join(raw_dir_destination,new_filename_img)
        dir_destination_msk = os.path.join(msk_dir_destination,filename_msk)
        dir_destination_ctd = os.path.join(ctd_dir_destination,filename_ctd)
        #Move files
        shutil.move(source_img, dir_destination_img)
        shutil.move(source_msk, dir_destination_msk)
        shutil.move(source_ctd, dir_destination_ctd)
        print("Subject "+str(subject)+" has been moved.")
    except:
        print("Subject "+str(subject)+" has already been moved.")