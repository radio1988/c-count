# this is a simple code for ad-hoc use
# for f in res/classification1/pos/*gz;do python auto_neg.py $f >> auto_neg/auto_neg.log;done

import sys, re, os
from pathlib import Path
home = str(Path.home())
sys.path.append(home+'/ccount/ccount/workflow/scripts/')
sys.path.append(home+'/ccount/ccount/workflow/scripts/ccount/')
from os.path import exists

from ccount.blob.io import load_blobs, save_crops, load_blobs
from ccount.blob.misc import get_label_statistics
from ccount.clas.pca_tsne import pca_tsne

from ccount.img.read_czi import read_czi, parse_image_obj
from ccount.blob.crop_blobs import crop_blobs
from ccount.blob.plot import plot_flat_crop, plot_flat_crops, pop_label_flat_crops, show_rand_crops

import numpy as np
import pandas as pd
import subprocess

import matplotlib
import matplotlib.pyplot as plt

# assumed file structure
# data:
# S0A1_FIRST_SCAN-Stitching-24.czi   S0B2_SECOND_SCAN-Stitching-15.czi
# 
# res/blob_locs/
# S0A1_FIRST_SCAN-Stitching-24.0.locs.npy.gz   S0A3_FIRST_SCAN-Stitching-20.0.locs.npy.gz   S0B2_FIRST_SCAN-Stitching-16.0.locs.npy.gz
# S0A1_FIRST_SCAN-Stitching-24.1.locs.npy.gz   S0A3_FIRST_SCAN-Stitching-20.1.locs.npy.gz   S0B2_FIRST_SCAN-Stitching-16.1.locs.npy.gz
# 
# res/classification1/pos/
# S0A1_FIRST_SCAN-Stitching-24.0.crops.clas.npy.gz   S0A3_FIRST_SCAN-Stitching-20.0.crops.clas.npy.gz   S0B2_FIRST_SCAN-Stitching-16.0.crops.clas.npy.gz
# S0A1_FIRST_SCAN-Stitching-24.1.crops.clas.npy.gz   S0A3_FIRST_SCAN-Stitching-20.1.crops.clas.npy.gz   S0B2_FIRST_SCAN-Stitching-16.1.crops.clas.npy.gz

fname = sys.argv[1] # ./path/S0A1_FIRST_SCAN-Stitching-24.0.crops.clas.npy.gz
name =  os.path.basename(fname)
name = name.replace('.crops.clas.npy.gz','')
print('name:',name)


# prep file names
match = re.search(r'(\.)(\d+)$', name)
if match:
    plate_name = name[:match.start()]
    image_index = match.group(2)
    print("plate_name:", plate_name)
    print("image_index:", image_index)
else:
    # No match found, handle accordingly
    print("No match found")
    
czi_name = plate_name + '.czi'
czi_file = './data/' + czi_name
print('czi_file:', czi_file, exists(czi_file))

pos_loc_file = './res/classification1/pos/' + name + '.crops.clas.npy.gz'  # for labeling
print('pos_loc_file:', pos_loc_file, exists(pos_loc_file))

out_name = './auto_neg/' + name + '.labeled.npy.gz'

# READ DATA
pos_locs = load_blobs(pos_loc_file)  # contains y,x,r in the first 3 columns

czi = read_czi(czi_file)  # image array of 4 scanned areas
image = parse_image_obj(czi, i=image_index)  # one of the scanned areas, takes 30s to load
czi=[] # release RAM

# Special Case
# save auto-negs for myeloid_fp_labeling
# which are the first round negs with older version of trained weights (observed false negative is low)
from ccount.blob.intersect import setdiff_blobs

all_loc_file =  './res/blob_locs/' + name + '.locs.npy.gz' # pos and neg, to autocreate negs
print('all_loc_file:', all_loc_file, exists(all_loc_file))

all_locs = load_blobs(all_loc_file)  # contains y,x,r in the first 3 columns
all_locs.shape

auto_neg_locs = setdiff_blobs(all_locs, pos_locs)
auto_neg_crops = crop_blobs(auto_neg_locs, image, crop_width=80)
auto_neg_crops[:, 3] = 0
print("auto_neg_crops", auto_neg_crops.shape)
auto_neg_crops_outname = './auto_neg/' + name + '.auto_neg.npy.gz'

print('save to:', auto_neg_crops_outname)
save_crops(auto_neg_crops, auto_neg_crops_outname)
