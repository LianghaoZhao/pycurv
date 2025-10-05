from pycurv import pycurv_io as io
from pycurv import run_gen_surface, THRESH_SIGMA1, TriangleGraph, MAX_DIST_SURF
import numpy as np
from scipy import ndimage
from graph_tool import load_graph
import mrcfile
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str,help='input segmentation in mrc')
parser.add_argument('--output',type=str,help='base name for output',default=None)
parser.add_argument('--label',type=int,help='Value of label to generate surface',nargs='+',default=None)
parser.add_argument('--pixel',type=float,help='pixel size of input segmentation in nanometer',default=0.968)
parser.add_argument('--sigma',type=int,help='sigma for smoothing',default=5)
parser.add_argument('--closing',type=int,default=5,help='kernal size for binary closing operation,0 for skip binary closing')
parser.add_argument('--fill',type=bool,help='Is the seg are filled or not(default False)',default=False)
args = parser.parse_args()

fold = ''  # output will be also written there
if args.output == None:
    base_filename = os.path.basename(args.input).split('.')[0]
else:
    base_filename = args.output
pixel_size = args.pixel  # pixel size of the (underlying) segmentation
radius_hit = 10  # radius of the smallest feature of interest (neighborhood)

# alternative or optional:
# for step 1.:
#   for segmentation input:
cube_size = args.closing  # try 3 or 5
label = args.label # if compartment segmentation
#   for surface input:
#surf_file = <your_surface_file>  # VTP in this example
# for step 2.:
# to remove small disconnected surface components within this size (default 100)
min_component = 100
# for step 3.:
methods = ["VV", "SSVV"]  # list of algorithms to run (default "VV")
area2 = True  # if method "VV": True for AVV (default), False for RVV
cores = 24
seg = io.load_tomo(args.input)
data_type = seg.dtype

if label == None:
    seg[seg !=0] = 1
else:
    seg = np.isin(seg ,label).astype(data_type)

if args.fill == True:
    surf = run_gen_surface(seg, fold + base_filename, lbl=1, isosurface=True, sg=args.sigma, thr=THRESH_SIGMA1)
else:
    if cube_size > 0 :
        cube = np.ones((cube_size, cube_size, cube_size))
        seg = ndimage.binary_closing(seg, structure=cube, iterations=1).astype(data_type)
    surf = run_gen_surface(seg, fold + base_filename, lbl=1)

tg = TriangleGraph()
scale = (pixel_size, pixel_size, pixel_size)
tg.build_graph_from_vtk_surface(surf, scale)
tg.find_vertices_near_border(MAX_DIST_SURF * pixel_size, purge=True)
tg.find_small_connected_components(threshold=min_component, purge=True, verbose=True)
clean_graph_file = '{}.scaled_cleaned.gt'.format(base_filename)
clean_surf_file = '{}.scaled_cleaned.vtp'.format(base_filename)
tg.graph.save(fold + clean_graph_file)
surf_clean = tg.graph_to_triangle_poly()
io.save_vtp(surf_clean, fold + clean_surf_file)
