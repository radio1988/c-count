import argparse
import os
import re
import matplotlib
from ccount import read_czi, block_equalize, down_scale

# Parse args
parser = argparse.ArgumentParser(description='Convert czi to jpg images')
parser.add_argument('-i', type=str,
                   help='input file name')
parser.add_argument('-f', type=str, 
                   help='czi file format: 2018, 2019, 2020')
args = parser.parse_args()
print('input fname:', args.i)
print('input format:', args.f)

outname = os.path.basename(args.i)
out_img_fname = re.sub('.czi$', '.jpg', outname)
equ_img_fname = re.sub('.czi$', '.equ.jpg', outname)
print("output_img_fname:", os.path.join("jpg", out_img_fname))
print("equalized_output_img_fname:", os.path.join("equ",equ_img_fname))

# Read Equ Write
image = read_czi(args.i, Format=args.f)
if not os.path.exists("jpg"):
    os.mkdir("jpg")
matplotlib.image.imsave(os.path.join("jpg", out_img_fname), image, cmap = "gray")
if not os.path.exists("equ"):
    os.mkdir("equ")
image_equ = block_equalize(image, block_height=2000, block_width=2400)
matplotlib.image.imsave(os.path.join("equ",equ_img_fname), image_equ, cmap = "gray")

