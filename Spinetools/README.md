# DTUSpineTools
Tools for processing and analysis of the spine and vertebra

- **dtu_spine_tools.py** : The main script that should be callled
- **dtu_spine_config.py** : For reading and taking care of the config file
- **dtu_spine_util.py** : functions for general processing of image, label images, surfaces and so on

To use the functions they should be called with a config file as argument
```
dtu_spine_tools -c configs/dtu-spine-config-rapa.json
```

Where the config file is a JSON file with user specifik folders and so on. You should **start** by copying the existing JSON file and make your own copy reflecting where your files are stored etc.

# How to run

In order to run you need:
- A folder with the scan NIFTI files
- A folder with the segmentations corresponding to the NIFTI files
- A csv file with the scan ids and the scan image file names
- The JSON file where the above files and folders are specified

An example csv file can be (with 4 scans):

```csv
sub-gl279, sub-gl279_dir-ax_ct.nii.gz
sub-verse502, sub-verse502_dir-iso_ct.nii.gz
sub-verse517, sub-verse517_dir-iso_ct.nii.gz
sub-verse551, sub-verse551_dir-iso_ct.nii.gz
```

## The output
The output is:

- **crops** found in the crops folder. Each crop is a NIFTI volume cropped around the given vertebra. They are also sampled to be isotropic. The corresponding crop of the label image is also there.
- **outlier crops** found in the crops folder. An image artefact is put on the crop to create an outlier. By default it is a circle centered at the border of the vertebra, where the CT values are now randomly sampled. A corresponding label map is also generated, where the outlier area is set to label=0.
- **surfaces** found in the surfaces folder. The surface of given label (vertebra)
- **outlier surfaces** found in the surfaces folder. The surface of given label where the outlier has been applied (vertebra)
- **registered template** found in the registration folder. The template surface registered to all target surfaces specified in the csv id file.
- **distance fields** found in the dist_field folder. The distance fields corresponding to the given label map. Also the outlier distance maps are produced.

# Code content

The examples in the codes shows:
- how to read CT images and label maps. 
- how to generate VTK surface files (.vtk) from label maps
- how to read a file with SimpleITK and convert the data to Numpy
- Computing the center-of-mass of a label map
- Extracting and resampling a crop from an image (the resampling results in isotropic voxels)
- Registering two distance fields using Elastix
- Propagating a template mesh to a target shape


## Installing Elastix

We use the precompiled [Elastix](https://elastix.lumc.nl/index.php) that can be installed from:

(https://github.com/SuperElastix/elastix/releases)

When you have installed Elastix you should update the JSON configuration files, so the tag **elastix_dir** points to where the executable are placed.

## External libraries

We mostly use [SimpleITK](https://simpleitk.readthedocs.io/en/master/index.html#) for image reading/writing and for 3D image manipulations. 

For 3D surface processing, we use [The Visualization Toolkit (VTK)](https://vtk.org/). VTK also has very powerfull 3D visualization tools.

Sometimes it is necessary to convert from SimpleITK to VTK and the other way around. SimpleITK has some very good image readers and writers while VTK for example can extract iso-surfaces from scans and label maps.

[3D Slicer](https://www.slicer.org/) is built on top of VTK.




## Coordinate systems

One major head ache when dealing with 3D images is the choice of coordinate systems. We are, at least, using these systems:

- The CT scans *physical* coordinate system that is typically measured in mm and describes the coordinates inside the patient/scanner.
- The index based coordinate system of the CT scan (indeces are integers)
- The index based coordinate system as a Numpy 3D array. A transpose operation is needed to convert between SimpleITK and Numpy indices.

For example:
```python
# Do the transpose of the coordinates (SimpleITK vs. numpy)
com_itk = [com_np[2], com_np[1], com_np[0]]
```

There are also different conventions used in medical scanners (LPS, RAS etc)

More details here:

[SimpleITK coordinate systems](https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html)

[3D slicer coordinate systems](https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html)



### SimpleITK, VTK and spline transformations (mostly notes for Rasmus)

- https://medium.com/@fanzongshaoxing/image-augmentation-based-on-3d-thin-plate-spline-tps-algorithm-for-ct-data-fa8b1b2a683c
- https://discourse.itk.org/t/solved-bsplinetransformation-from-displacement-vectors/5372/6
- https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1LandmarkBasedTransformInitializerFilter.html
- https://github.com/SimpleITK/SimpleITK/issues/197
- https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/65_Registration_FFD.ipynb
- https://gitlab.kitware.com/vtk/vtk/-/blob/v9.3.0/Common/Transforms/Testing/Python/TestThinPlateWarp3D.py
- https://gitlab.kitware.com/vtk/vtk/-/blob/v9.3.0/Common/Transforms/Testing/Python/TestThinPlateWarp.py


#### Conversions
- https://github.com/SimpleITK/SimpleITKUtilities
- https://discourse.vtk.org/t/axes-direction-cosines-please-help/8094
