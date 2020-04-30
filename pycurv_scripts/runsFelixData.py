import time
import sys

from pycurv_scripts import (
    new_workflow, extract_curvatures_after_new_workflow,
    convert_vtp_to_stl_surface_and_mrc_curvatures)
from pycurv import pycurv_io as io

"""
Applying new_workflow and extract_curvatures_after_new_workflow on Felix' data.

Author: Maria Salfer (Max Planck Institute for Biochemistry),
date: 2018-05-29
"""

__author__ = 'Maria Salfer'


def main(tomo):
    t_begin = time.time()

    # parameters for all tomograms:
    fold = "/fs/pool/pool-ruben/Maria/curvature/Felix/corrected_method/"
    base_filename = "{}_ER".format(tomo)
    pixel_size = 2.526  # nm
    radius_hit = 10  # nm
    seg_file = ""
    lbl = 1
    cube_size = 3
    min_component = 100

    # parameters for each tomogram:
    if tomo == "t112":
        fold = "{}diffuseHtt97Q/".format(fold)
        seg_file = "{}_final_ER1_vesicles2_notER3_NE4.Labels.mrc".format(tomo)
    elif tomo == "t122":
        fold = "{}away_from_Htt97Q_IB/".format(fold)
        seg_file = "{}_final_ER1_vesicle2_notER3_NE4.Labels.mrc".format(tomo)
    elif tomo == "t158":
        fold = "{}diffuseHtt25Q/".format(fold)
        seg_file = "{}_final_ER1_NE2.Labels.mrc".format(tomo)
    elif tomo == "t166":
        fold = "{}Htt64Q_IB/".format(fold)
        seg_file = "{}_cleaned_ER.mrc".format(tomo)
    elif tomo == "t85":
        fold = "{}Htt97Q_IB_t85/".format(fold)
        seg_file = "{}_mask_membrane_final_ER.mrc".format(tomo)
    elif tomo == "t84":
        fold = "{}Htt97Q_IB_t84/".format(fold)
        seg_file = "{}_mask_membrane_final_ER.mrc".format(tomo)
    elif tomo == "t92":
        fold = "{}Htt97Q_IB_t92/".format(fold)
        seg_file = "{}_final_ER1_vesicles2_NE3.Labels.mrc".format(tomo)
    elif tomo == "t138":
        fold = "{}Htt97Q_IB_t138/".format(fold)
        pixel_size = 2.84  # nm
        seg_file = "{}_final_ER1_notInHttContact2.Labels.mrc".format(tomo)

    new_workflow(
        base_filename, seg_file, fold, pixel_size, radius_hit, methods=['VV'],
        label=lbl, holes=cube_size, remove_wrong_borders=True,
        min_component=min_component)

    print("\nExtracting curvatures for ER")
    extract_curvatures_after_new_workflow(
        fold, base_filename, radius_hit, methods=['VV'], exclude_borders=1)

    surf_vtp_file = '{}{}.{}_rh{}.vtp'.format(
        fold, base_filename, 'AVV', radius_hit)
    outfile_base = '{}{}.{}_rh{}'.format(
        fold, base_filename, 'AVV', radius_hit)
    scale = (pixel_size, pixel_size, pixel_size)
    seg = io.load_tomo(fold + seg_file)
    size = seg.shape
    convert_vtp_to_stl_surface_and_mrc_curvatures(
        surf_vtp_file, outfile_base, scale, size)

    t_end = time.time()
    duration = t_end - t_begin
    minutes, seconds = divmod(duration, 60)
    print('\nTotal elapsed time: {} min {} s'.format(minutes, seconds))


if __name__ == "__main__":
    tomo = sys.argv[1]
    main(tomo)
