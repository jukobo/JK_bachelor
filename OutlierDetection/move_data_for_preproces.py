from os import listdir
import os
import shutil

    
#Input directories
# dir_crops = 'Data/Output_spinetools/crops'  #Output folder from Spinetools
dir_crops = 'Data/Output_spinetools/dist_fields'  #Output folder from Spinetools


dir_destination = 'Data/Verse20/Outlier_detection/crops_training_unpacked' 
 


#Define list of scans
scans = [f for f in listdir(dir_crops)] #Remove file .DS_Store
if not os.path.exists(dir_destination):
   os.makedirs(dir_destination)

#FOR LOOP START
for subject in scans:
    print("       SUBJECT: "+str(subject)+"\n")
    try:
        
        st = str(subject)
        #Get directory of source
        source_img = os.path.join(dir_crops,subject)
        
        #Get new file names for image

        if st.split('_')[-1]=='label.nii.gz':
            new_ending = 'msk.nii.gz'
            name = st[:-len(st.split('_')[-1])]
            subject = name + new_ending
        
        elif st.split('_')[-2]=='label':
            new_ending = 'outlier_msk.nii.gz'
            name = st[:-(len(st.split('_')[-1])+len(st.split('_')[-2])+1)]
            subject = name + new_ending

        elif st.split('_')[-1]=='crop.nii.gz':
            new_ending = 'crop_img.nii.gz'
            name = st[:-len(st.split('_')[-1])]
            subject = name + new_ending

        
       
        #Get final directory of destination
        dir_destination_img = os.path.join(dir_destination,subject)
        dir_destination_dist = os.path.join(dir_destination,subject)
        #Move files
        shutil.move(source_img, dir_destination_img)
        print("Subject "+str(subject)+" has been moved.")
    
    except shutil.Error as e:
        if "already exists" in str(e):
            print("Subject "+str(subject)+" has already been moved.")
        else:
            raise e

