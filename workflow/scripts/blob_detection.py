from ccount import read_czi, block_equalize, down_scale
from ccount import find_blobs_and_crop, remove_edge_crops, vis_blob_on_block
import argparse, os, re, matplotlib
matplotlib.use('Agg')  # for plotting without GUI
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess

# Parse args
parser = argparse.ArgumentParser(
    description='Read czi, output blobs'
    )
parser.add_argument(
    '-i', type=str,
    help='input file name: xxx.czi'
    )
parser.add_argument(
    '-f', type=str, default="2019",
    help='czi file format: 2018/2019/2020')
parser.add_argument(
    '-odir', type=str, default="blobs",
    help='output dir: default blobs'
    )

args = parser.parse_args()
print('input fname:', args.i)
print('input format:', args.f)
print('output dir:', args.odir)

outname = os.path.basename(args.i)
corename = re.sub('.czi$', '', outname)

# Prep output dir
Path(os.path.join(args.odir, "vis")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(args.odir, "hist")).mkdir(parents=True, exist_ok=True)

# User Params (you can adjust these parameters to suit your data)
if args.f=='2019':
    block_height = 2000
    block_width = 2400 # pixcels, if 0, use whole image as block (still buggy whole image equalization)
    scaling_factor = 4 # 1: original dimension, 2: 1/2, 4: 1/4
else:
    # todo
    raise ValueError("format not supported")
    block_height = 2000
    block_width = 2400 # pixcels, if 0, use whole image as block (still buggy whole image equalization)
    scaling_factor = 4 # 1: original dimension, 2: 1/2, 4: 1/4

# Developer Params (please don't change unless necessary)
# Jan 2020, jpg -> png; save images in workdir
# Jan 2020, scaling_factor=1, larger num_sigma 5->10, min_sigma 11->6
visualization = True
equalization = True
blob_extention_ratio = 1.4 # soft extend blob radius manually (1.4)
blob_extention_radius = 2 # pixcels to extend (2)
crop_width = 80  # padding width, which is cropped img width/2 (50)
overlap=.0

if scaling_factor == 1:
    max_sigma=50 
    min_sigma=4
    num_sigma=15
    threshold=0.1
elif scaling_factor == 2:
    max_sigma=40
    min_sigma=2
    num_sigma=10
    threshold=0.1
elif scaling_factor == 4:
    max_sigma=20
    min_sigma=8
    num_sigma=10  # Aug2019: 10 better sensitivity than 5, 1/2 speed
    threshold=0.04
else:
    raise Exception("scaling factor not implemented")
test = True

# Read
images = read_czi(args.i, Format=args.f)
for i,image in enumerate(images):
    i=str(i)
    print ("\n\nFor image:", i)
    out_blob_fname = os.path.join(args.odir, corename+"."+i+".npy")
    out_img_fname = os.path.join(args.odir, "vis", corename+"."+i+".jpg")
    hist_img_fname = os.path.join(args.odir, "hist", corename+"."+i+".hist.pdf")
    print("output blob:", out_blob_fname)
    print("output histogram:", hist_img_fname)
    print("output_img_fname:", out_img_fname)

    image = image - np.min(image)

    if test:
        image = image[0:4000, 0:4000]
        print("in test mode")

    # Blob Detection
    print(">>> Equalizing image..")
    if block_width <=0:
        image_equ = equalize(image)
    else:
        image_equ = block_equalize(image, block_height=block_height, block_width=block_width)

    print(">>> Detecting blobs..")
    image_flat_crops = find_blobs_and_crop(
        image, image_equ,
        crop_width=crop_width, 
        # blob_detection parameters
        scaling_factor=scaling_factor,
        max_sigma=max_sigma, 
        min_sigma=min_sigma, 
        num_sigma=num_sigma, 
        threshold=threshold, 
        overlap=overlap,
        # blob yellow circle visualization parameters
        blob_extention_ratio=blob_extention_ratio,
        blob_extention_radius=blob_extention_radius,
        fname=out_img_fname  # if None, will plot inline
    )

    # Remove blobs containing edges
    good_flats = remove_edge_crops(image_flat_crops)

    # Hist
    r_ = good_flats[:,2]
    plt.hist(r_, 40)
    plt.title("Histogram of blob size")
    plt.savefig(hist_img_fname)

    # Saving
    print('there are {} blobs detected'.format(len(image_flat_crops)))
    print("there are {} blobs passed edge filter".format(len(good_flats)))
    print(good_flats.shape)
    np.save(out_blob_fname, good_flats)
    print('saved into {}'.format(out_blob_fname))
    bashCommand = "gzip -f " + out_blob_fname
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Visualizing filtered blobs
    print("Visualizing blobs")
    vis_blob_on_block(good_flats, image_equ,image, 
        blob_extention_ratio=blob_extention_ratio, 
        blob_extention_radius=blob_extention_radius, 
        fname=out_img_fname)

    if test:
        Path("./blobs/vis_beforeEdgeFilter").mkdir(parents=True, exist_ok=True)
        out_img_fnameBefore = os.path.join(args.odir, "vis_beforeEdgeFilter", corename+".jpg")
        vis_blob_on_block(image_flat_crops, image_equ,image, 
            blob_extention_ratio=blob_extention_ratio, 
            blob_extention_radius=blob_extention_radius, 
            fname=out_img_fnameBefore)


