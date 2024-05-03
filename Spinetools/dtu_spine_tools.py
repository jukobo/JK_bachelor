import vtk
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import os
from pathlib import Path
import subprocess
from shutil import copyfile 
import argparse
from dtu_spine_config import DTUConfig
import dtu_spine_utils as dsu
from scipy import ndimage
import random


def extract_label_surfaces(settings, scan_id, label_id, on_crop, on_outlier=False):
    """
    Extract label surfaces for the given scan_id
    Use the label id provided so several different labels can be extracted
    Can also deal with crops
    """
    print(f"Extracting label surfaces for id {scan_id}")
    base_dir = settings["base_dir"]
    crop_dir = os.path.join(base_dir, "crops")
    surface_dir = os.path.join(base_dir, "surfaces")
    # Create folders if they don't exist
    Path(surface_dir).mkdir(parents=True, exist_ok=True)

    segm_dir = settings["segmentation_dir"]

    if on_crop:
        i_name = os.path.join(crop_dir, f"{scan_id}_segment_{label_id}_crop_label.nii.gz")
        if on_outlier:
            i_name = os.path.join(crop_dir, f"{scan_id}_segment_{label_id}_crop_label_outlier.nii.gz")
    else:
        i_name = os.path.join(segm_dir, f"{scan_id}_PREDICTIONafter.nii.gz")
    surf_name = os.path.join(surface_dir, f"{scan_id}_label_{label_id}_surface.vtk")
    if on_outlier:
        surf_name = os.path.join(surface_dir, f"{scan_id}_label_{label_id}_outlier_surface.vtk")
    dsu.convert_label_map_to_surface(i_name, surf_name, segment_id=label_id, only_largest_component=True)


def compute_template_mesh(settings, scan_id, label_id, on_crop):
    """
    Compute a template mesh that we can propagate to all other segmentations and thereby creating point correspondence
    Base it on the surface from scan_id
    """
    print("Computing template mesh")
    base_dir = settings["base_dir"]
    surface_dir = os.path.join(base_dir, "surfaces")
    # Create folders if they don't exist
    Path(surface_dir).mkdir(parents=True, exist_ok=True)

    template_id = scan_id

    surf_name = os.path.join(surface_dir, f"{template_id}_label_{label_id}_surface.vtk")
    surf_name_template = os.path.join(surface_dir, f"{template_id}_label_{label_id}_template_surface.vtk")
    pd = vtk.vtkPolyDataReader()
    pd.SetFileName(surf_name)
    pd.Update()
    # template_pd = dsu.smooth_and_refine_mesh(pd.GetOutput(), on_crop)
    template_pd = dsu.smooth_and_refine_mesh_constrained_smoother(pd.GetOutput(), on_crop)

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(template_pd)
    writer.SetFileName(surf_name_template)
    writer.SetFileVersion(42)
    writer.SetFileTypeToASCII()
    writer.Write()


def compute_distance_fields(settings, scan_id, label_id, on_crop):
    """
    Compute distance field given a scan_id and a label_id
    Can also work with crops
    """
    print("Computing distance fields")
    base_dir = settings["base_dir"]
    crop_dir = os.path.join(base_dir, "crops")
    dist_field_dir = os.path.join(base_dir, "dist_fields")
    # Create folders if they don't exist
    Path(dist_field_dir).mkdir(parents=True, exist_ok=True)

    segm_dir = settings["segmentation_dir"]

    if on_crop:
        i_name = os.path.join(crop_dir, f"{scan_id}_segment_{label_id}_crop_label.nii.gz")
    else:
        i_name = os.path.join(segm_dir, f"{scan_id}_PREDICTIONafter.nii.gz")
    dist_name = os.path.join(dist_field_dir, f"{scan_id}_distance_field.nii.gz")
    dsu.compute_distance_field(i_name, dist_name, label_id)


def register_segmentations(settings, template_id, target_id, label_id, on_outlier=False):
    print("Doing registration")
    elastix_dir = settings["elastix_dir"]
    elastix_exe = os.path.join(elastix_dir, 'elastix.exe')
    transformix_exe = os.path.join(elastix_dir, 'transformix.exe')
    elastix_settings_dir = settings["elastix_settings_dir"]

    base_dir = settings["base_dir"]
    surface_dir = os.path.join(base_dir, "surfaces")
    dist_field_dir = os.path.join(base_dir, "dist_fields")
    registration_dir = os.path.join(base_dir, "registration")
    # Create folders if they don't exist
    Path(registration_dir).mkdir(parents=True, exist_ok=True)

    # The goal is to bring the template surface onto the target surface
    # IMPORTANT:
    # Here we use the TEMPLATE as the fixed volume and the TARGET as the moving volume.
    # The transformation is then computed to bring the TARGET over in the TEMPLATE
    # (which seems like the inverse of what we want)
    # It is actually the inverse transform that is computed (due to the resampling method) by Elastix
    # Therefore we can apply the transformation to the surface of the TEMPLATE (the fixed)
    # this will bring it onto the surface of the TARGET
    fixed_distance_field = os.path.join(dist_field_dir, f"{template_id}_distance_field.nii.gz")
    moving_distance_field = os.path.join(dist_field_dir, f"{target_id}_distance_field.nii.gz")
    surf_name_template = os.path.join(surface_dir, f"{template_id}_label_{label_id}_template_surface.vtk")
    template_mesh_fitted = os.path.join(registration_dir, f"{template_id}_label_{label_id}_fitted_to_{target_id}.vtk")
    parameter_file_rigid = os.path.join(elastix_settings_dir, "Parameters_Rigid.txt")
    parameter_file_bspline = os.path.join(elastix_settings_dir, "Parameters_BSpline.txt")
    output_dir_trans = os.path.join(registration_dir, "transformation_parameters/")
    output_dir_result = os.path.join(registration_dir, "transformation_result/")
    transformation_parms = f"{output_dir_trans}/TransformParameters.1.txt"
    if on_outlier:
        moving_distance_field = os.path.join(dist_field_dir, f"{target_id}_distance_field_outlier.nii.gz")
        template_mesh_fitted = os.path.join(registration_dir,
                                            f"{template_id}_label_{label_id}_fitted_to_{target_id}_outlier.vtk")

    if not os.path.exists(fixed_distance_field) or not os.path.exists(moving_distance_field):
        print(f"Could not find distance fields {fixed_distance_field} or {moving_distance_field}")
        return

    Path(output_dir_trans).mkdir(parents=True, exist_ok=True)
    Path(output_dir_result).mkdir(parents=True, exist_ok=True)

    # Run Elastix registration using a rigid and a bspline transformation
    # the result is stored in the output_dir_trans folder
    command_line = f"{elastix_exe} -f {fixed_distance_field} -m {moving_distance_field} " \
                   f"-p {parameter_file_rigid} -p {parameter_file_bspline} -out {output_dir_trans}"

    subprocess.run(command_line)

    command_line = f"{transformix_exe} -def {surf_name_template} -tp {transformation_parms} " \
                   f"-out {output_dir_result}"
    subprocess.run(command_line)
    copyfile(os.path.join(output_dir_result, "outputpoints.vtk"), template_mesh_fitted)


def extract_crop_around_vertebra(settings, scan_id, scan_image, label_id):
    """
    Generate an image that is the crop around a specified label.
    The crop is centered on the center of mass of the given label
    Also create a cropped label image
    """
    print("Extracting crop around vertebra")
    # Side length in mm
    crop_side_length = 120 
    base_dir = settings["base_dir"]
    image_dir = settings["image_dir"]
    surface_dir = os.path.join(base_dir, "surfaces")
    crop_dir = os.path.join(base_dir, "crops")
    # Create folders if they don't exist
    Path(surface_dir).mkdir(parents=True, exist_ok=True)
    Path(crop_dir).mkdir(parents=True, exist_ok=True)

    segm_dir = settings["segmentation_dir"]
    segm_name = os.path.join(segm_dir, f"{scan_id}_PREDICTIONafter.nii.gz")
    scan_name = os.path.join(image_dir, scan_image)
    crop_name = os.path.join(crop_dir, f"{scan_id}_segment_{label_id}_crop.nii.gz")
    segm_crop_name = os.path.join(crop_dir, f"{scan_id}_segment_{label_id}_crop_label.nii.gz")
    com_name = os.path.join(surface_dir, f"{scan_id}_segment_{label_id}_com.txt")

    # Compute the center of mass of the given label
    # Read the segmentation and turn into a numpy array
    try:
        img = sitk.ReadImage(segm_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {segm_name}")
        return

    segm_np = sitk.GetArrayFromImage(img)
    if np.sum(segm_np == label_id) == 0:
        print(f"Label {label_id} not found in {segm_name}")
        return
    com_np = ndimage.center_of_mass(segm_np == label_id)

    # Do the transpose of the coordinates (SimpleITK vs. numpy)
    com_itk = [com_np[2], com_np[1], com_np[0]]
    # Transform the index to physical coordinates
    com_phys = img.TransformIndexToPhysicalPoint([int(com_itk[0]), int(com_itk[1]), int(com_itk[2])])
    with open(com_name, 'w') as f:
        f.write(f"{com_phys[0]} {com_phys[1]} {com_phys[2]}\n")
    # print(com_phys)

    # Compute the crop around the center of mass
    # The crop is defined in physical coordinates
    crop_center = [com_phys[0], com_phys[1], com_phys[2]]
    voxel_side_length = 0.5  

    try:
        img_ct = sitk.ReadImage(scan_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {scan_name}")
        return
    # print(f"Pixel type: {img_ct.GetPixelIDTypeAsString()}")

    crop = dsu.crop_and_resample_roi(img_ct, crop_center, crop_side_length, voxel_side_length)
    sitk.WriteImage(crop, crop_name)

    crop_segment = dsu.crop_and_resample_roi(img, crop_center, crop_side_length, voxel_side_length, label_map=True)
    sitk.WriteImage(crop_segment, segm_crop_name)


def create_outlier(settings, scan_id, label_id):
    """
    Create an outlier by adding a sphere to the cropped image. The sphere is placed at a random position on the surface
    of the label. The sphere is created by sampling from a normal distribution with mean and standard deviation
    estimated from the cropped image. The outlier is created for both the CT and the label.
    A distance field is also created for the outlier label.
    """
    base_dir = settings["base_dir"]
    dist_field_dir = os.path.join(base_dir, "dist_fields")
    surface_dir = os.path.join(base_dir, "surfaces")
    crop_dir = os.path.join(base_dir, "crops")
    crop_name = os.path.join(crop_dir, f"{scan_id}_segment_{label_id}_crop.nii.gz")
    segm_crop_name = os.path.join(crop_dir, f"{scan_id}_segment_{label_id}_crop_label.nii.gz")
    surf_name = os.path.join(surface_dir, f"{scan_id}_label_{label_id}_surface.vtk")
    outlier_ct_name = os.path.join(crop_dir, f"{scan_id}_segment_{label_id}_crop_outlier.nii.gz")
    outlier_label_name = os.path.join(crop_dir, f"{scan_id}_segment_{label_id}_crop_label_outlier.nii.gz")
    outlier_dist_name = os.path.join(dist_field_dir, f"{scan_id}_distance_field_outlier.nii.gz")

    try:
        img_ct = sitk.ReadImage(crop_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {crop_name}")
        return

    try:
        img_label = sitk.ReadImage(segm_crop_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {segm_crop_name}")
        return

    # Find random position on surface
    surface = vtk.vtkPolyDataReader()
    surface.SetFileName(surf_name)
    surface.Update()
    template_pd = surface.GetOutput()
    n_points = template_pd.GetNumberOfPoints()
    p_idx = np.random.randint(0, n_points)
    p = template_pd.GetPoint(p_idx)

    # Convert to numpy - now remember to transpose coordinates
    img_np = sitk.GetArrayFromImage(img_ct)
    pidx = img_ct.TransformPhysicalPointToIndex(p)
    outlier_p = [pidx[2], pidx[1], pidx[0]]
    # Radius in Pixels
    # Inside create_outlier function
    Type = random.randint(1, 3)
    print(Type)
    radius = random.randint(10, 20) 

     
    # generates different radius
    # estimate mean and standard deviation of full image to be able to sample from a normal distribution
    mean = np.mean(img_np)
    std = np.std(img_np)
    print(f"Creating outlier with mean {mean} and std {std}")
    # img_np_outlier = dsu.draw_sphere_on_numpy_image(img_np, outlier_p, radius, value=mean, std_dev=std)
    img_np_outlier = dsu.draw_shape_on_numpy_image(img_np, outlier_p, Type, radius, value=mean, std_dev=std)


    # now put voxels values back into ITK and save
    img_o = sitk.GetImageFromArray(img_np_outlier)
    img_o.CopyInformation(img_ct)

    print(f"Saving {outlier_ct_name}")
    sitk.WriteImage(img_o, outlier_ct_name)

    # Do the same for the label
    pidx = img_label.TransformPhysicalPointToIndex(p)
    outlier_p = [pidx[2], pidx[1], pidx[0]]
    img_np = sitk.GetArrayFromImage(img_label)
    # img_np_outlier = dsu.draw_sphere_on_numpy_image(img_np, outlier_p, radius, 0)
    img_np_outlier = dsu.draw_shape_on_numpy_image(img_np, outlier_p, Type, radius, 0)


    # now put voxels values back into ITK and save
    img_o = sitk.GetImageFromArray(img_np_outlier)
    img_o.CopyInformation(img_label)

    print(f"Saving {outlier_label_name}")
    sitk.WriteImage(img_o, outlier_label_name)

    # Create distance field for outlier image
    print(f"Computing outlier distance field")
    dsu.compute_distance_field(outlier_label_name, outlier_dist_name, label_id)
    extract_label_surfaces(settings, scan_id, label_id, on_crop=True, on_outlier=True)


def create_randomly_displaced_points(settings):
    """
    Currently not in use
    """
    base_dir = settings["base_dir"]
    surface_dir = os.path.join(base_dir, "surfaces")
    registration_dir = os.path.join(base_dir, "registration")
    # Create folders if they don't exist
    Path(registration_dir).mkdir(parents=True, exist_ok=True)

    template_id = settings["template_id"]
    # target_id = settings["target_id"]
    # segm_dir = settings["segmentation_dir"]
    label_id = 22

    surf_name_template = os.path.join(surface_dir, f"{template_id}_label_{label_id}_template_surface.vtk")
    points_out = os.path.join(surface_dir, f"{template_id}_label_{label_id}_template_random_points.txt")

    surface = vtk.vtkPolyDataReader()
    surface.SetFileName(surf_name_template)
    surface.Update()

    template_pd = surface.GetOutput()
    n_points = 8
    sample_points = dsu.sample_random_points_from_cloud_maximum_spread(template_pd, n_points)
    # print(sample_points)
    with open(points_out, 'w') as f:
        for p in sample_points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    # Add random displacement to the points
    # The displacement is in mm
    displacement = 10
    points_out_displaced = os.path.join(surface_dir, f"{template_id}_label_{label_id}_template_random_points_displaced.txt")
    for p in sample_points:
        p[0] += np.random.uniform(-displacement, displacement)
        p[1] += np.random.uniform(-displacement, displacement)
        p[2] += np.random.uniform(-displacement, displacement)
    with open(points_out_displaced, 'w') as f:
        for p in sample_points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def landmark_based_spline_transformation(settings):
    """
    Currently out-of-order - do not use
    Now works but only for VTK version 9.3.0 and above due to the direction matrices in vtkImageReslice
    """
    print("Doing landmark-based spline transformation")
    base_dir = settings["base_dir"]
    surface_dir = os.path.join(base_dir, "surfaces")
    dist_field_dir = os.path.join(base_dir, "dist_fields")
    registration_dir = os.path.join(base_dir, "registration")
    scan_image = settings["template_image"]
    image_dir = settings["image_dir"]
    crop_dir = os.path.join(base_dir, "crops")
    crop_side_length = 120

    # create_randomly_displaced_points(settings)

    template_id = settings["template_id"]
    # target_id = settings["target_id"]
    # segm_dir = settings["segmentation_dir"]
    label_id = 22
    points_src_name = os.path.join(surface_dir, f"{template_id}_label_{label_id}_template_random_points.txt")
    points_trg_name = os.path.join(surface_dir, f"{template_id}_label_{label_id}_template_random_points_displaced.txt")
    # scan_name = os.path.join(image_dir, scan_image)
    scan_name_ct = os.path.join(crop_dir, f"{template_id}_segment_{label_id}_crop.nii.gz")
    trans_image_out_ct = os.path.join(crop_dir, f"{template_id}_segment_{label_id}_crop_transformed.nii.gz")
    scan_name = os.path.join(crop_dir, f"{template_id}_segment_{label_id}_crop_label.nii.gz")
    trans_image_out = os.path.join(crop_dir, f"{template_id}_segment_{label_id}_crop_label_transformed.nii.gz")
    trans_surface_out = os.path.join(surface_dir, f"{template_id}_label_{label_id}_template_surface_transformed.vtk")
    surf_name_template = os.path.join(surface_dir, f"{template_id}_label_{label_id}_template_surface.vtk")
    points_out = os.path.join(surface_dir, f"{template_id}_label_{label_id}_template_random_points_enhanced.txt")
    points_out_disp = os.path.join(surface_dir, f"{template_id}_label_{label_id}_template_random_points_disp_enhanced.txt")

    surface = vtk.vtkPolyDataReader()
    surface.SetFileName(surf_name_template)
    surface.Update()
    template_pd = surface.GetOutput()

    # The goal is to bring the template surface onto the target surface
    # src_points = dsu.read_landmarks(points_src_name)
    # trg_points = dsu.read_landmarks(points_trg_name)

    src_points_org = dsu.read_landmarks_as_vtk(points_src_name)
    trg_points_org = dsu.read_landmarks_as_vtk(points_trg_name)

    sample_spacing = crop_side_length / 10.0
    pd_center, pd_radius = dsu.compute_bounding_sphere(template_pd)
    pd_radius += 20
    side_length = crop_side_length + 20.0
    new_points = dsu.sample_points_in_square_exclude_sphere(pd_center, side_length, pd_center, pd_radius,
                                                            sample_spacing=sample_spacing)
    print(f"Number of new points: {new_points.GetNumberOfPoints()}")

    spo = vtk.vtkPolyData()
    spo.SetPoints(src_points_org)
    tpo = vtk.vtkPolyData()
    tpo.SetPoints(trg_points_org)
    npd = vtk.vtkPolyData()
    npd.SetPoints(new_points)

    append_filter = vtk.vtkAppendPoints()
    append_filter.AddInputData(spo)
    append_filter.AddInputData(npd)
    append_filter.Update()
    src_points = append_filter.GetOutput().GetPoints()
    dsu.write_vtk_points_to_txt_file(src_points, points_out)

    append_filter_trg = vtk.vtkAppendPoints()
    append_filter_trg.AddInputData(tpo)
    append_filter_trg.AddInputData(npd)
    append_filter_trg.Update()
    trg_points = append_filter_trg.GetOutput().GetPoints()
    dsu.write_vtk_points_to_txt_file(trg_points, points_out_disp)

    # fixed_pts = trg_points
    # moving_pts = src_points
    # fixed_pts = np.array(list(src_points.flatten()))
    # moving_pts = np.array(list(trg_points.flatten()))

    try:
        img_label = sitk.ReadImage(scan_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {scan_name}")
        return
    img_vtk = dsu.sitk2vtk(img_label)

    try:
        img_ct = sitk.ReadImage(scan_name_ct)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {scan_name_ct}")
        return
    img_vtk_ct = dsu.sitk2vtk(img_ct)

    transform = vtk.vtkThinPlateSplineTransform()
    # Remember that we need the inverse transform - so here we switch the source and target
    transform.SetSourceLandmarks(trg_points)
    transform.SetTargetLandmarks(src_points)

    # transform.SetSourceLandmarks(trg_points_org)
    # transform.SetTargetLandmarks(src_points_org)

    # transform.SetSourceLandmarks(src_points)
    # transform.SetTargetLandmarks(trg_points)
    transform.SetBasisToR()
    # print("Doing inverse")
    # transform.Inverse()
    # transform.Modified()
    transform.Update()

    print("Starting reslice")

    # you must invert the transform before passing it to vtkImageReslice
    # transform.Inverse()
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(img_vtk)
    reslice.SetResliceTransform(transform)
    # reslice.SetOutputDirection(1, 0, 0, 0, 1, 0, 0, 0, 1)
    # reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 1, 0, 0, 0, 1)
    # Remember to set the interpolation mode something else when using real image
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    # print(f"Reslice matrix : {reslice.GetResliceAxesDirectionCosines()}")
    # print(f"image direction : {img_ct.GetDirection()}")

    # writer = vtk.vtkNIFTIImageWriter()
    # writer.SetInputConnection(reslice.GetOutputPort())
    # writer.SetFileName(trans_image_out)
    # writer.Write()

    img_trans = dsu.vtk2sitk(reslice.GetOutput())
    img_trans.CopyInformation(img_ct)
    sitk.WriteImage(img_trans, trans_image_out)
    dsu.convert_label_map_to_surface(trans_image_out, trans_surface_out, segment_id=label_id, only_largest_component=True)

    print("Starting reslice ct")
    # Now do the same for the CT
    reslice.SetInputData(img_vtk_ct)
    reslice.SetInterpolationModeToLinear()
    reslice.Modified()
    reslice.Update()

    img_trans_ct = dsu.vtk2sitk(reslice.GetOutput())
    img_trans_ct.CopyInformation(img_ct)
    sitk.WriteImage(img_trans_ct, trans_image_out_ct)


    # print(f"Direction: {img_ct.GetDirection()} Origin: {img_ct.GetOrigin()}  Spacing: {img_ct.GetSpacing()}  Size: {img_ct.GetSize()}  Pixel type: {img_ct.GetPixelIDTypeAsString()}")
    #
    # print("Initialize the transform")
    # # Determine the number of BSpline control points using the physical spacing we want for the control grid.
    # grid_physical_spacing = [1.0, 1.0, 1.0]  # A control point every 5mm
    # image_physical_size = [
    #     size * spacing
    #     for size, spacing in zip(img_ct.GetSize(), img_ct.GetSpacing())
    # ]
    # mesh_size = [
    #     int(image_size / grid_spacing + 0.5)
    #     for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)
    # ]
    # ix = sitk.BSplineTransformInitializer(img_ct, mesh_size, order=3)
    #
    # # ix = sitk.BSplineTransformInitializer(img_ct, (10, 10, 10), order=3)
    # #
    # # transformDomainMeshSize = [10] * img_ct.GetDimension()
    # # ix = sitk.BSplineTransformInitializer(img_ct, transformDomainMeshSize)
    # #
    # print("Set the landmarks")
    # landmarkTx = sitk.LandmarkBasedTransformInitializerFilter()
    # landmarkTx.SetFixedLandmarks(fixed_pts)
    # landmarkTx.SetMovingLandmarks(moving_pts)
    # landmarkTx.SetBSplineNumberOfControlPoints(8)
    # landmarkTx.SetReferenceImage(img_ct)
    # initialTx = landmarkTx.Execute(ix)
    # #


    # # Create landmark transform
    # landmark_tx = sitk.LandmarkBasedTransformInitializer(
    #         #transform = sitk.VersorRigid3DTransform(),
    #         transform = sitk.BSplineTransform(3),   # Dimension=2 (2D Image)
    #         transformDomainMeshSize=mesh_size,
    #         fixedLandmarks = fixed_pts,
    #         movingLandmarks = moving_pts,
    #         referenceImage = img_ct,
    #         numberOfControlPoints = 8
    #     )
    #
    # # Apply transform to image
    # rimg = sitk.Resample(img_ct, landmark_tx)
    # sitk.WriteImage(rimg, trans_image_out)

# def analyze_template_mesh_points():
#     p_idx_1 = 3845
#     p_idx_2 = 12769
#
#     template_mesh_fitted = "C:/data/Spine/Caroline-Nov-2023/vertebra_502_template_fitted_to_517.vtk"
#     points_out = "C:/data/Spine/Caroline-Nov-2023/vertebra_502_template_fitted_to_517_points.txt"
#     pd = vtk.vtkPolyDataReader()
#     pd.SetFileName(template_mesh_fitted)
#     pd.Update()
#
#     points = pd.GetOutput().GetPoints()
#     p1 = points.GetPoint(p_idx_1)
#     p2 = points.GetPoint(p_idx_2)
#
#     with open(points_out, 'w') as f:
#         f.write(f"{p1[0]} {p1[1]} {p1[2]}\n")
#         f.write(f"{p2[0]} {p2[1]} {p2[2]}\n")


def spine_tools(cfg):
    """
    Main function for spine_tools
    """
    print("Running spine_tools")
    # Should be adapted to the specific segmentations
    label_id = 20 #L1
    template_id = cfg.settings["template_id"]
    template_image = cfg.settings["template_image"]
    print(f"Creating template with id {template_id} and label id {label_id}")
    extract_crop_around_vertebra(cfg.settings, template_id, template_image, label_id)
    extract_label_surfaces(cfg.settings, template_id, label_id, on_crop=True)
    compute_template_mesh(cfg.settings, scan_id=template_id, label_id=label_id, on_crop=True)
    compute_distance_fields(cfg.settings, scan_id=template_id, label_id=label_id, on_crop=True)

    # Get list of all scans
    base_dir = cfg.settings["base_dir"]
    id_list = cfg.settings["id_list"]
    id_list_file = os.path.join(base_dir, id_list)
    all_scan_ids = np.loadtxt(str(id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} scans")

    for idx in all_scan_ids:
        scan_id = idx[0].strip()
        scan_image = idx[1].strip()
        print(f"Processing scan {scan_id} with image {scan_image}")
        extract_crop_around_vertebra(cfg.settings, scan_id, scan_image, label_id)
        extract_label_surfaces(cfg.settings, scan_id, label_id=label_id, on_crop=True)
        compute_distance_fields(cfg.settings, scan_id=scan_id, label_id=label_id, on_crop=True)
        # register_segmentations(cfg.settings, template_id=template_id, target_id=scan_id, label_id=label_id,
        #                        on_outlier=False)

    # # Now create outliers
    # for idx in all_scan_ids:
    #     scan_id = idx[0].strip()
    #     scan_image = idx[1].strip()
    #     print(f"Creating outlier from scan {scan_id} with image {scan_image}")
    #     create_outlier(cfg.settings, scan_id, label_id)
    #     # register_segmentations(cfg.settings, template_id=template_id, target_id=scan_id, label_id=label_id,
    #     #                        on_outlier=True)


def debug_testing(cfg):
    """
    Debugging functions
    """
    landmark_based_spline_transformation(cfg.settings)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='dtu_do_visualization')
    config = DTUConfig(args)
    if config.settings is not None:
        spine_tools(config)
        # debug_testing(config)

