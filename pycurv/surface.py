import vtk

from . import pexceptions
from . import pycurv_io as io
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from .linalg import dot_norm
import time

"""
Set of functions for generating a single-layer, signed surface from a
membrane segmentation and postprocessing the surface, using the VTK library.

Authors: Maria Salfer and Antonio Martinez-Sanchez (Max Planck Institute for
Biochemistry)
"""

__author__ = 'Antonio Martinez-Sanchez and Maria Salfer'


# CONSTANTS
MAX_DIST_SURF = 3
"""int: a constant determining the maximal distance in pixels of a point on the
surface from the segmentation mask, used in gen_isosurface and gen_surface
functions.
"""

THRESH_SIGMA1 = 0.699471735
"""float: when convolving a binary mask with a gaussian kernel with sigma 1,
values at the boundary with 0's become this value
"""


def reverse_sense_and_normals(vtk_algorithm_output):
    """
    Sometimes the contouring algorithm can create a volume whose gradient
    vector and ordering of polygon (using the right hand rule) are
    inconsistent. vtkReverseSense cures this problem.

    Args:
        vtk_algorithm_output (vtkAlgorithmOutput): output of a VTK algorithm,
            to get with: algorithm_instance.GetOutputPort()

    Returns:
        surface with reversed normals (vtk.vtkPolyData)
    """

    reverse = vtk.vtkReverseSense()
    reverse.SetInputConnection(vtk_algorithm_output)
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOn()
    reverse.Update()
    return reverse.GetOutput()


import numpy as np
import vtk
from scipy.ndimage import distance_transform_edt
# 假设 io, pexceptions, reverse_sense_and_normals, MAX_DIST_SURF, dot_norm 都已正确导入

def gen_surface(tomo, lbl=1, mask=True, other_mask=None, purge_ratio=1,
                field=False, mode_2d=False, verbose=True):
    """
    Generates a VTK PolyData surface from a segmented tomogram.
    """
    if verbose:
        print("Starting gen_surface...")

    # 读取分割数据
    if verbose:
        print("Reading input segmentation...")
    if isinstance(tomo, str):
        tomo = io.load_tomo(tomo)
        if verbose:
            print(f"Loaded segmentation from file: {tomo.shape}")
    elif not isinstance(tomo, np.ndarray):
        raise pexceptions.PySegInputError(
            expr='gen_surface',
            msg='Input must be either a file name or a ndarray.')
    else:
        if verbose:
            print(f"Using input ndarray: {tomo.shape}")

    # 使用向量化操作生成点云，而不是三重循环
    nx, ny, nz = tomo.shape
    if verbose:
        print(f"Segmentation shape: ({nx}, {ny}, {nz})")
        print("Generating point cloud using vectorization...")

    # 创建坐标网格
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
    )

    # 找到标签等于lbl的位置
    mask_condition = (tomo == lbl)

    if purge_ratio <= 1:
        # 直接获取所有符合条件的点
        if verbose:
            print("Finding all points matching label without purging...")
        valid_coords = np.where(mask_condition)
        x_valid, y_valid, z_valid = valid_coords
    else:
        # 应用purge_ratio
        if verbose:
            print(f"Applying purge ratio of {purge_ratio}...")
        purge_mask = np.random.randint(0, purge_ratio, size=tomo.shape) == 0
        combined_mask = mask_condition & purge_mask
        valid_coords = np.where(combined_mask)
        x_valid, y_valid, z_valid = valid_coords

    n_points = len(x_valid)
    if verbose:
        print(f'Point cloud generated. Total points: {n_points}')

    # 创建VTK点云（使用numpy数组一次性设置点）
    if verbose:
        print("Creating VTK point cloud...")
    cloud = vtk.vtkPolyData()
    points = vtk.vtkPoints()

    # 预分配点的数量
    points.SetNumberOfPoints(n_points)

    # 批量设置点坐标
    for i in range(n_points):
        points.SetPoint(i, float(x_valid[i]), float(y_valid[i]), float(z_valid[i]))

    cloud.SetPoints(points)

    # --- 关键计算步骤：VTK表面重建 ---
    if verbose:
        print(f"Starting VTK surface reconstruction (SampleSpacing: {purge_ratio})...")
        print("This step can take a significant amount of time for large point clouds.")
    surf = vtk.vtkSurfaceReconstructionFilter()
    surf.SetSampleSpacing(purge_ratio)
    surf.SetInputData(cloud)

    contf = vtk.vtkContourFilter()
    contf.SetInputConnection(surf.GetOutputPort())
    contf.SetValue(0, 0)

    rsurf = reverse_sense_and_normals(contf.GetOutputPort())
    if verbose:
        print('VTK surface reconstruction and contouring completed.')

    # 优化缩放和变换
    if verbose:
        print("Computing bounds and applying transformation...")
    cloud.ComputeBounds()
    rsurf.ComputeBounds()
    xmin, xmax, ymin, ymax, zmin, zmax = cloud.GetBounds()
    rxmin, rxmax, rymin, rymax, rzmin, rzmax = rsurf.GetBounds()

    # 计算缩放因子
    scale_x = (xmax - xmin) / (rxmax - rxmin) if (rxmax - rxmin) != 0 else 1
    scale_y = (ymax - ymin) / (rymax - rymin) if (rymax - rymin) != 0 else 1
    scale_z = (zmax - zmin) / (rzmax - rzmin) if (rzmax - rzmin) != 0 else 1

    # 创建变换
    transp = vtk.vtkTransform()
    transp.Translate(xmin, ymin, zmin)
    transp.Scale(scale_x, scale_y, scale_z)
    transp.Translate(-rxmin, -rymin, -rzmin)

    tpd = vtk.vtkTransformPolyDataFilter()
    tpd.SetInputData(rsurf)
    tpd.SetTransform(transp)
    tpd.Update()
    tsurf = tpd.GetOutput()
    if verbose:
        print('Transformation applied.')

    # 应用掩码（如果需要）
    if mask:
        # --- 关键计算步骤：距离变换 ---
        if verbose:
            print("Starting distance transform for masking...")
            print("This step can be slow for large volumes.")
        mask_to_use = other_mask if other_mask is not None else tomo
        distance_mask = np.invert(mask_to_use == lbl)
        tomod = distance_transform_edt(distance_mask)
        if verbose:
            print("Distance transform for masking completed.")

        if tsurf.GetNumberOfCells() > 0:
            # --- 计算步骤：掩码应用 ---
            if verbose:
                print("Applying mask to surface...")
                print(f"Processing {tsurf.GetNumberOfCells()} cells.")
            # 获取所有单元格的点坐标
            cells_to_delete = []
            for i in range(tsurf.GetNumberOfCells()):
                cell = tsurf.GetCell(i)
                points_cell = cell.GetPoints()

                # 检查当前单元格的所有点
                should_delete = False
                for j in range(points_cell.GetNumberOfPoints()):
                    x, y, z = points_cell.GetPoint(j)
                    if tomod[int(round(x)), int(round(y)), int(round(z))] > MAX_DIST_SURF:
                        should_delete = True
                        break

                if should_delete:
                    cells_to_delete.append(i)

            # 批量删除单元格
            for cell_id in cells_to_delete:
                tsurf.DeleteCell(cell_id)

            tsurf.RemoveDeletedCells()
            if verbose:
                print(f"Mask applied. {len(cells_to_delete)} cells deleted.")

        if verbose:
            print('Masking completed.')

    # 字段距离计算（如果需要）
    if field:
        # --- 关键计算步骤：法向量计算 ---
        if verbose:
            print("Starting VTK normal calculation...")
            print("This step can be slow for large meshes.")
        # 计算法向量
        norm_flt = vtk.vtkPolyDataNormals()
        norm_flt.SetInputData(tsurf)
        norm_flt.ComputeCellNormalsOn()
        norm_flt.AutoOrientNormalsOn()
        norm_flt.ConsistencyOn()
        norm_flt.Update()
        tsurf = norm_flt.GetOutput()
        if verbose:
            print("VTK normal calculation completed.")

        # 获取法向量数组
        array = tsurf.GetCellData().GetNormals()

        # 使用向量化操作构建掩码
        if verbose:
            print("Building mask and normal arrays...")
        tomoh = np.ones(shape=tomo.shape, dtype=bool)
        tomon = np.zeros(shape=(tomo.shape[0], tomo.shape[1], tomo.shape[2], 3),
                        dtype=np.float32)

        # 优化：先收集所有表面点，然后批量更新
        for i in range(tsurf.GetNumberOfCells()):
            cell = tsurf.GetCell(i)
            points_cell = cell.GetPoints()
            for j in range(points_cell.GetNumberOfPoints()):
                x, y, z = points_cell.GetPoint(j)
                x, y, z = int(round(x)), int(round(y)), int(round(z))
                if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz and tomo[x, y, z] == lbl:
                    tomoh[x, y, z] = False
                    if array and i < array.GetNumberOfTuples():
                        tomon[x, y, z, :] = array.GetTuple(i)
        if verbose:
            print("Mask and normal arrays built.")

        # --- 关键计算步骤：距离变换 (第二次) ---
        if verbose:
            print("Starting distance transform for field calculation...")
            print("This step can be slow for large volumes.")
        # 计算距离变换
        tomod, ids = distance_transform_edt(tomoh, return_indices=True)
        if verbose:
            print("Distance transform for field completed.")

        # --- 计算步骤：极性计算 ---
        if verbose:
            print("Starting polarity field calculation...")
            print("This step can be slow for large volumes.")
        # 向量化计算极性（而不是三重循环）
        if mode_2d:
            # 创建坐标网格用于向量化计算
            x_grid, y_grid, z_grid = np.meshgrid(
                np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
            )

            # 获取索引
            i_x, i_y, i_z = ids[0], ids[1], ids[2]

            # 获取法向量和位置向量
            norms = tomon[i_x, i_y, i_z]  # shape: (nx, ny, nz, 3)
            norms_2d = norms.copy()
            norms_2d[..., 2] = 0  # 设置z分量为0

            # 计算点积
            pnorm = np.stack([i_x.astype(np.float32), i_y.astype(np.float32),
                             np.zeros_like(i_x, dtype=np.float32)], axis=-1)
            p = np.stack([x_grid.astype(np.float32), y_grid.astype(np.float32),
                         np.zeros_like(x_grid, dtype=np.float32)], axis=-1)

            # 计算点积
            dot_products = np.sum(p * pnorm * norms_2d, axis=-1)
            tomod = tomod * np.sign(dot_products)
        else:
            # 3D模式的向量化计算
            x_grid, y_grid, z_grid = np.meshgrid(
                np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
            )

            i_x, i_y, i_z = ids[0], ids[1], ids[2]
            norms = tomon[i_x, i_y, i_z]

            pnorm = np.stack([i_x.astype(np.float32), i_y.astype(np.float32),
                             i_z.astype(np.float32)], axis=-1)
            p = np.stack([x_grid.astype(np.float32), y_grid.astype(np.float32),
                         z_grid.astype(np.float32)], axis=-1)

            # 计算点积
            dot_products = np.sum(pnorm * p * norms, axis=-1)
            tomod = tomod * np.sign(dot_products)
        if verbose:
            print("Polarity field calculation completed.")

        if verbose:
            print('Distance field generation completed.')

        return tsurf, tomod

    if verbose:
        print('Surface generation completed!')

    return tsurf

def gen_isosurface(tomo, lbl, grow=0, sg=0, thr=1.0, mask=None):
    """
    Generates a isosurface using the Marching Cubes method.

    Args:
        tomo (str or numpy.ndarray): segmentation input file in one of the
            formats: '.mrc', '.em' or '.vti', or 3D array containing the
            segmentation
        lbl (int): the label to be considered (> 0)
        grow (int, optional): if > 0 the surface is grown by so many voxels
            (default 0 - no growing)
        sg (int, optional): sigma for gaussian smoothing in voxels (default 0 -
            no smoothing)
        thr (optional, float): thr for isosurface (default 1.0)
        mask (int or numpy.ndarray, optional): if given (default None), the
            surface will be masked with it: if integer, this label is extracted
            from the input segmentation to generate the binary mask, otherwise
            it has to be given as a numpy.ndarray with same dimensions as the
            input segmentation

    Returns:
        a surface (vtk.vtkPolyData)
    """
    # Read in the segmentation (if file is given) and check format
    if isinstance(tomo, str):
        tomo = io.load_tomo(tomo)
    elif not isinstance(tomo, np.ndarray):
        raise pexceptions.PySegInputError(
            expr='gen_isosurface',
            msg='Input must be either a file name or a ndarray.')

    # Binarize the segmentation
    data_type = tomo.dtype
    binary_seg = tomo == lbl

    # Growing
    if grow > 0:
        binary_seg = binary_dilation(binary_seg, iterations=grow)
        # 3x3 structuring element with connectivity 1 is used by default

    binary_seg = binary_seg.astype(data_type)

    # Smoothing
    if sg > 0:
        binary_seg = gaussian_filter(binary_seg.astype(np.float32), sg)

    # Generate isosurface
    smoothed_seg_vti = io.numpy_to_vti(binary_seg)
    surfaces = vtk.vtkMarchingCubes()
    surfaces.SetInputData(smoothed_seg_vti)
    surfaces.ComputeNormalsOn()
    surfaces.ComputeGradientsOn()
    surfaces.SetValue(0, thr)
    surfaces.Update()

    surf = reverse_sense_and_normals(surfaces.GetOutputPort())

    # Apply the mask
    if mask is not None:
        if isinstance(mask, int):  # mask is a label inside the segmentation
            mask = (tomo == mask).astype(data_type)
        elif not isinstance(mask, np.ndarray):
            raise pexceptions.PySegInputError(
                expr='gen_isosurface',
                msg='Input mask must be either an integer or a ndarray.')
        dist_from_mask = distance_transform_edt(mask == 0)
        for i in range(surf.GetNumberOfCells()):
            # Check if all points which made up the polygon are in the mask
            points_cell = surf.GetCell(i).GetPoints()
            count = 0
            for j in range(0, points_cell.GetNumberOfPoints()):
                x, y, z = points_cell.GetPoint(j)
                if (dist_from_mask[
                        int(round(x)), int(round(y)), int(round(z))] >
                        MAX_DIST_SURF):
                    count += 1
            # Mark cells that are not completely in the mask for deletion
            if count > 0:
                surf.DeleteCell(i)
        # Delete
        surf.RemoveDeletedCells()

    return surf


def run_gen_surface(tomo, outfile_base, lbl=1, mask=True, other_mask=None,
                    save_input_as_vti=False, verbose=False, isosurface=False,
                    grow=0, sg=0, thr=1.0):
    """
    Generates a VTK PolyData triangle surface for objects in a segmented volume
    with a given label.

    Removes triangles with zero area, if any are present, from the resulting
    surface.

    Args:
        tomo (str or numpy.ndarray): segmentation input file in one of the
            formats: '.mrc', '.em' or '.vti', or 3D array containing the
            segmentation
        outfile_base (str): the path and filename without the ending for saving
            the surface (ending '.surface.vtp' will be added automatically)
        lbl (int, optional): the label to be considered, 0 will be ignored,
            default 1
        mask (boolean, optional): if True (default), a mask of the binary
            objects is applied on the resulting surface to reduce artifacts
            (in case isosurface=False)
        other_mask (numpy.ndarray, optional): if given (default None), this
            segmentation is used as mask for the surface
        save_input_as_vti (boolean, optional): if True (default False), the
            input is saved as a '.vti' file ('<outfile_base>.vti')
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out
        isosurface (boolean, optional): if True (default False), generate
            isosurface (good for filled segmentations) - last three parameters
            are used in this case
        grow (int, optional): if > 0 the surface is grown by so many voxels
            (default 0 - no growing)
        sg (int, optional): sigma for gaussian smoothing in voxels (default 0 -
            no smoothing)
        thr (optional, float): thr for isosurface (default 1.0)

    Returns:
        the triangle surface (vtk.PolyData)
    """
    t_begin = time.time()

    # Generating the surface (vtkPolyData object)
    if isosurface:
        surface = gen_isosurface(tomo, lbl, grow, sg, thr, mask=other_mask)
    else:
        surface = gen_surface(tomo, lbl, mask, other_mask, verbose=verbose)

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('Surface generation took: {} min {} s'.format(minutes, seconds))

    # Writing the vtkPolyData surface into a VTP file
    io.save_vtp(surface, outfile_base + '.surface.vtp')
    print('Surface was written to the file {}.surface.vtp'.format(outfile_base))

    if save_input_as_vti is True:
        # If input is a file name, read in the segmentation array from the file:
        if isinstance(tomo, str):
            tomo = io.load_tomo(tomo)
        elif not isinstance(tomo, np.ndarray):
            raise pexceptions.PySegInputError(
                expr='run_gen_surface',
                msg='Input must be either a file name or a ndarray.')

        # Save the segmentation as VTI for opening it in ParaView:
        io.save_numpy(tomo, outfile_base + '.vti')
        print('Input was saved as the file {}.vti'.format(outfile_base))

    return surface


def add_curvature_to_vtk_surface(surface, curvature_type, invert=True):
    """
    Adds curvatures (Gaussian, mean, maximum or minimum) calculated by VTK to
    each triangle vertex of a vtkPolyData surface.

    Args:
        surface (vtk.vtkPolyData): a surface of triangles
        curvature_type (str): type of curvature to add: 'Gaussian', 'Mean',
            'Maximum' or 'Minimum'
        invert (boolean, optional): if True (default), VTK will calculate
            curvatures as for meshes with opposite pointing normals (their
            convention is outwards pointing normals, opposite from ours)

    Returns:
        the vtkPolyData surface with '<type>_Curvature' property added to each
        triangle vertex
    """
    if isinstance(surface, vtk.vtkPolyData):
        curvature_filter = vtk.vtkCurvatures()
        curvature_filter.SetInputData(surface)
        if curvature_type == "Gaussian":
            curvature_filter.SetCurvatureTypeToGaussian()
        elif curvature_type == "Mean":
            curvature_filter.SetCurvatureTypeToMean()
        elif curvature_type == "Maximum":
            curvature_filter.SetCurvatureTypeToMaximum()
        elif curvature_type == "Minimum":
            curvature_filter.SetCurvatureTypeToMinimum()
        else:
            raise pexceptions.PySegInputError(
                expr='add_curvature_to_vtk_surface',
                msg=("One of the following strings required as the second "
                     "input: 'Gaussian', 'Mean', 'Maximum' or 'Minimum'."))
        if invert:
            curvature_filter.InvertMeanCurvatureOn()  # default Off
        curvature_filter.Update()
        surface_curvature = curvature_filter.GetOutput()
        return surface_curvature
    else:
        raise pexceptions.PySegInputError(
            expr='add_curvature_to_vtk_surface',
            msg="A vtkPolyData object required as the first input.")
    # How to get the curvatures later, e.g. for point with ID 0:
    # point_data = surface_curvature.GetPointData()
    # curvatures = point_data.GetArray(n)
    # where n = 2 for Gaussian, 3 for Mean, 4 for Maximum or Minimum
    # curvature_point0 = curvatures.GetTuple1(0)


def add_point_normals_to_vtk_surface(surface, reverse_normals=False):
    """
    Adds a normal to each triangle vertex of a vtkPolyData surface.

    Args:
        surface (vtk.vtkPolyData): a surface of triangles
        reverse_normals (boolean, optional): if True (default False), VTK will
            flip the normals (their convention is outwards pointing normals,
            opposite from ours)

    Returns:
        the vtkPolyData surface with '<type>_Curvature' property added to each
        triangle vertex
    """
    if isinstance(surface, vtk.vtkPolyData):
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(surface)
        normals.ComputePointNormalsOn()
        if reverse_normals:
            normals.FlipNormalsOn()
        else:
            normals.FlipNormalsOff()
        normals.Update()
        surface_normals = normals.GetOutput()
        return surface_normals
    else:
        raise pexceptions.PySegInputError(
            expr='add_point_normals_to_vtk_surface',
            msg="A vtkPolyData object required as the first input.")


def rescale_surface(surface, scale):
    """
    Rescales the given vtkPolyData surface with a given scaling factor in each
    of the three dimensions.

    Args:
        surface (vtk.vtkPolyData): a surface of triangles
        scale (tuple): a scaling factor in 3D (x, y, z)

    Returns:
        rescaled surface (vtk.vtkPolyData)
    """
    try:
        assert isinstance(surface, vtk.vtkPolyData)
    except AssertionError:
        raise pexceptions.PySegInputError(
            expr='rescale_surface',
            msg="A vtkPolyData object required as the first input.")
    transf = vtk.vtkTransform()
    transf.Scale(scale[0], scale[1], scale[2])
    tpd = vtk.vtkTransformPolyDataFilter()
    tpd.SetInputData(surface)
    tpd.SetTransform(transf)
    tpd.Update()
    scaled_surface = tpd.GetOutput()
    return scaled_surface
