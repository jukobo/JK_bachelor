import os.path

import vtk
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import vtk.util.numpy_support as vtknp
import skimage.filters as filters


def sitk2vtk(img, flip_for_volume_rendering=False, debugOn=False):
    """Convert a SimpleITK image to a VTK image, via numpy."""
    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()
    direction = img.GetDirection()

    # convert the SimpleITK image to a numpy array
    i2 = sitk.GetArrayFromImage(img)
    if debugOn:
        i2_string = i2.tostring()
        print("data string address inside sitk2vtk", hex(id(i2_string)))

    vtk_image = vtk.vtkImageData()

    # VTK expects 3-dimensional parameters
    if len(size) == 2:
        size.append(1)

    if len(origin) == 2:
        origin.append(0.0)

    if len(spacing) == 2:
        spacing.append(spacing[0])

    if len(direction) == 4:
        direction = [
            direction[0],
            direction[1],
            0.0,
            direction[2],
            direction[3],
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion() < 9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        vtk_image.SetDirectionMatrix(direction)

    # TODO: Volume rendering does not support direction matrices (27/5-2023)
    # so sometimes the volume rendering is mirrored
    # this a brutal hack to avoid that
    if flip_for_volume_rendering:
        if direction[4] < 0:
            i2 = np.fliplr(i2)

    depth_array = numpy_to_vtk(i2.ravel(), deep=True)
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)

    vtk_image.Modified()

    return vtk_image

# https://github.com/SimpleITK/SimpleITKUtilities/tree/main
def vtk2sitk(image: vtk.vtkImageData) -> sitk.Image:
    """Convert a VTK image to a SimpleITK image.

    Note that VTK images are fundamentally 3D, even if the Z
    dimension is 1.

    :param image: Image to convert.
    :return: A SimpleITK image.
    """
    sd = image.GetPointData().GetScalars()
    npdata = vtknp.vtk_to_numpy(sd)
    dims = list(image.GetDimensions())
    dims.reverse()
    ncomp = image.GetNumberOfScalarComponents()
    if ncomp > 1:
        dims.append(ncomp)

    npdata.shape = tuple(dims)

    sitk_image = sitk.GetImageFromArray(npdata)
    sitk_image.SetSpacing(image.GetSpacing())
    sitk_image.SetOrigin(image.GetOrigin())
    # By default, direction is identity.

    if vtk.vtkVersion.GetVTKMajorVersion() >= 9:
        # Copy the direction matrix into a list
        dir_mat = image.GetDirectionMatrix()
        direction = [0] * 9
        dir_mat.DeepCopy(direction, dir_mat)
        sitk_image.SetDirection(direction)

    return sitk_image


def convert_label_map_to_surface(label_name, output_file, reset_direction_matrix=False, segment_id=1,
                                 only_largest_component=False):
    """
    Convert a label map to a surface using marching cubes and save it to a file
    The specifik label that should be converted should be specified with segment_id
    """
    try:
        img = sitk.ReadImage(label_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {label_name}")
        return None

    vtk_img = sitk2vtk(img, flip_for_volume_rendering=False)
    if vtk_img is None:
        return False

    # Check if there is any data
    vol_np = vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    if np.sum(vol_np) < 1:
        print(f"Only zeros in {label_name}")
        return False

    if reset_direction_matrix:
        direction = [1, 0, 0.0, 0, 1, 0.0, 0.0, 0.0, 1.0]
        vtk_img.SetDirectionMatrix(direction)

    print(f"Generating: {output_file}")
    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetNumberOfContours(1)
    mc.SetValue(0, segment_id)
    mc.Update()

    if mc.GetOutput().GetNumberOfPoints() < 10:
        print(f"No isosurface found in {label_name}")
        return False

    # Save in VTK version 4.2 and ASCII format so Elastix can read it
    if only_largest_component:
        conn = vtk.vtkConnectivityFilter()
        conn.SetInputConnection(mc.GetOutputPort())
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputConnection(conn.GetOutputPort())
        writer.SetFileTypeToASCII()
        writer.SetFileVersion(42)
        writer.SetFileName(output_file)
        writer.Write()
    else:
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputConnection(mc.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.SetFileTypeToASCII()
        writer.SetFileVersion(42)
        writer.SetFileName(output_file)
        writer.Write()

    return True


def compute_distance_field(segm_name, dist_name, label_id):
    """
    Compute the distance field for a label map by given the specifik label that should be used
    as zero level
    """
    # print("Computing distance field")
    # if os.path.exists(dist_name):
    #     print(f"Distance field already exists: {dist_name}")
    #     return

    try:
        img = sitk.ReadImage(segm_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {segm_name}")
        return

    # Extract image data in numpy format
    img_t = sitk.GetArrayFromImage(img)
    mask = img_t == label_id

    dist_image = distance_transform_edt(mask)

    # Outside mask
    mask_2 = img_t != label_id
    dist_image_2 = distance_transform_edt(mask_2)

    final_dist = dist_image_2 - dist_image
    img_o = sitk.GetImageFromArray(final_dist)
    img_o.CopyInformation(img)

    print(f"saving")
    sitk.WriteImage(img_o, dist_name)


def smooth_and_refine_mesh(vtk_in, on_crop):
    """
    Generate a smooth mesh from another mesh
    If on_crop is True, the mesh is processed a little harder since there are more triangles
    """
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(vtk_in)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    # print("Filling holes")
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(conn.GetOutput())
    fill_holes.SetHoleSize(1000.0)
    fill_holes.Update()

    # print("Triangle filter")
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(fill_holes.GetOutput())
    triangle.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(triangle.GetOutput())
    cleaner.Update()

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(cleaner.GetOutput())
    decimate.SetTargetReduction(0.20)
    if on_crop:
        decimate.SetTargetReduction(0.50)
    decimate.PreserveTopologyOn()
    decimate.Update()

    smooth_filter = vtk.vtkSmoothPolyDataFilter()
    smooth_filter.SetInputData(decimate.GetOutput())
    smooth_filter.SetNumberOfIterations(20)
    smooth_filter.SetRelaxationFactor(0.1)
    if on_crop:
        smooth_filter.SetNumberOfIterations(200)
    smooth_filter.FeatureEdgeSmoothingOff()
    smooth_filter.BoundarySmoothingOn()
    smooth_filter.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(smooth_filter.GetOutput())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def smooth_and_refine_mesh_constrained_smoother(vtk_in, on_crop):
    """
    Generate a smooth mesh from another mesh using a constrained smoother
    No decimation so the number of triangles is preserved
    """
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(vtk_in)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    # print("Filling holes")
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(conn.GetOutput())
    fill_holes.SetHoleSize(1000.0)
    fill_holes.Update()

    # print("Triangle filter")
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(fill_holes.GetOutput())
    triangle.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(triangle.GetOutput())
    cleaner.Update()

    # decimate = vtk.vtkDecimatePro()
    # decimate.SetInputData(cleaner.GetOutput())
    # decimate.SetTargetReduction(0.20)
    # if on_crop:
    #     decimate.SetTargetReduction(0.50)
    # decimate.PreserveTopologyOn()
    # decimate.Update()

    smooth_filter = vtk.vtkConstrainedSmoothingFilter()
    smooth_filter.SetInputData(cleaner.GetOutput())
    smooth_filter.SetNumberOfIterations(1000)
    smooth_filter.SetRelaxationFactor(0.01)
    smooth_filter.SetConstraintDistance(1)
    smooth_filter.SetConstraintStrategyToConstraintDistance()
    smooth_filter.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(smooth_filter.GetOutput())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def crop_and_resample_roi(image, roi_center, roi_side_length, voxel_side_length, label_map=False):
    # Create the sampled image with same direction
    direction = image.GetDirection()

    # Desired voxel spacing for new image
    new_spacing = [voxel_side_length, voxel_side_length, voxel_side_length]
    nvox_side = int(roi_side_length / voxel_side_length + 1)

    dir_x = direction[0]
    dir_y = direction[4]
    dir_z = direction[8]

    new_origin_x = roi_center[0] - dir_x * roi_side_length / 2
    new_origin_y = roi_center[1] - dir_y * roi_side_length / 2
    new_origin_z = roi_center[2] - dir_z * roi_side_length / 2

    new_size = [nvox_side, nvox_side, nvox_side]
    new_image = sitk.Image(new_size, image.GetPixelIDValue())
    new_image.SetOrigin([new_origin_x, new_origin_y, new_origin_z])
    new_image.SetSpacing(new_spacing)
    new_image.SetDirection(direction)

    # Make translation with no offset, since sitk.Resample needs this arg.
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((0, 0, 0))

    if label_map:
        default_value = 0
        interpolator = sitk.sitkNearestNeighbor
    else:
        default_value = -2048.0
        interpolator = sitk.sitkLinear
    # Create final resampled image
    resampled_image = sitk.Resample(image, new_image, translation, interpolator, default_value)

    return resampled_image


def sample_random_points_from_cloud_maximum_spread(pd, n_points):
    """
    Sample a set of points from a point cloud
    The points are sampled so that the spread is maximized
    """
    sampled_points = [list(pd.GetPoint(0))]

    for i in range(1, n_points):
        max_dist = -1
        max_point = None
        for j in range(pd.GetNumberOfPoints()):
            point = pd.GetPoint(j)
            min_dist = 1e10
            for k in range(len(sampled_points)):
                dist = np.linalg.norm(np.array(sampled_points[k]) - np.array(point))
                min_dist = min(min_dist, dist)
            if min_dist > max_dist:
                max_dist = min_dist
                max_point = point
        sampled_points.append(list(max_point))

    return sampled_points


def read_landmarks(filename):
    lms = []
    x, y, z = 0, 0, 0
    with open(filename) as f:
        for line in f:
            if len(line) > 1:
                temp = line.split()  # Remove whitespaces and line endings and so on
                x, y, z = np.double(temp)
                # lms.append([x, y, z])
                lms.append(x)
                lms.append(y)
                lms.append(z)
    return lms


def read_landmarks_as_vtk(filename):
    points = vtk.vtkPoints()
    with open(filename) as f:
        # id_lm = 0
        for line in f:
            if len(line) > 1:
                temp = line.split()  # Remove whitespaces and line endings and so on
                x, y, z = np.double(temp)
                points.InsertNextPoint(x, y, z)
    return points

def write_vtk_points_to_txt_file(points, filename):
    with open(filename, "w") as f:
        for i in range(points.GetNumberOfPoints()):
            point = points.GetPoint(i)
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


# def draw_sphere_on_numpy_image(img_np, center, radius, value, std_dev=None):
#     """
#     Draw a sphere on a numpy image.
#     If std_dev is given, the sphere is drawn with values following a Gaussian distribution with mean = value and
#     standard deviation = std_dev
#     """
#     # Create a meshgrid for the image
#     x, y, z = np.ogrid[0:img_np.shape[0], 0:img_np.shape[1], 0:img_np.shape[2]]
#     dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)

#     if std_dev is not None:
#         value = np.random.normal(value, std_dev, img_np.shape)
#         gauss_sigma = 1
#         value = filters.gaussian(value, sigma=gauss_sigma)
#         img_np[dist < radius] = value[dist < radius]
#     else:
#         img_np[dist < radius] = value
#     return img_np

import numpy as np
import scipy.ndimage as ndi
import cv2
import random

def draw_shape_on_numpy_image(img_np, center, shape, radius, value, std_dev=None):
    """
    Draw a shape (sphere, square, blob) on a numpy image.
    If std_dev is given, the shape is drawn with values following a Gaussian distribution with mean = value and
    standard deviation = std_dev
    """
    if shape == 1:
        "Sphere"
        # Create a meshgrid for the image
        x, y, z = np.ogrid[0:img_np.shape[0], 0:img_np.shape[1], 0:img_np.shape[2]]
        dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)

        if std_dev is not None:
            value = np.random.normal(value, std_dev, img_np.shape)
            gauss_sigma = 1
            value = filters.gaussian(value, sigma=gauss_sigma)
            img_np[dist < radius] = value[dist < radius]
        else:
            img_np[dist < radius] = value
    
    elif shape == 2:
        "Square"
        size = 30
        half_size = size // 2
        start = [max(0, center[0] - half_size), max(0, center[1] - half_size), max(0, center[2] - half_size)]
        end = [min(img_np.shape[0], center[0] + half_size), min(img_np.shape[1], center[1] + half_size), min(img_np.shape[2], center[2] + half_size)]
        img_np[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = value
    
    elif shape == 3:
        "Cylinder"
        length = 150

        # Create a meshgrid for the image
        x, y, z = np.ogrid[0:img_np.shape[0], 0:img_np.shape[1], 0:img_np.shape[2]]
        # Calculate the distance from the center along the x and y axes
        dist_xy = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        # Define the condition for points inside the cylinder
        in_cylinder = (dist_xy < radius) & (z >= center[2] - length // 2) & (z < center[2] + length // 2)
        # Assign the value to points inside the cylinder
        img_np[in_cylinder] = value
    else:
        raise ValueError("Unsupported shape")
    
    return img_np






def compute_bounding_sphere(pd):
    if pd.GetNumberOfPoints() < 1:
        print(f"Error: No points in point cloud")
        return None, None

    # find center of points
    center = [0, 0, 0]
    for i in range(pd.GetNumberOfPoints()):
        point = pd.GetPoint(i)
        center[0] += point[0]
        center[1] += point[1]
        center[2] += point[2]
    center[0] /= pd.GetNumberOfPoints()
    center[1] /= pd.GetNumberOfPoints()
    center[2] /= pd.GetNumberOfPoints()

    # find max distance from center
    max_dist = 0
    for i in range(pd.GetNumberOfPoints()):
        point = pd.GetPoint(i)
        dist = np.linalg.norm(np.array(center) - np.array(point))
        max_dist = max(max_dist, dist)

    return center, max_dist


def sample_points_in_square_exclude_sphere(center_square, side_length, center_sphere, radius, sample_spacing):
    """
    Sample points in a square, excluding a sphere
    """
    points = vtk.vtkPoints()
    x = center_square[0] - side_length / 2
    while x < center_square[0] + side_length / 2:
        y = center_square[1] - side_length / 2
        while y < center_square[1] + side_length / 2:
            z = center_square[2] - side_length / 2
            while z < center_square[2] + side_length / 2:
                point = [x, y, z]
                if np.linalg.norm(np.array(point) - np.array(center_sphere)) > radius:
                    points.InsertNextPoint(point)
                z += sample_spacing
            y += sample_spacing
        x += sample_spacing
    return points
