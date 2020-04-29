import argparse

# Parse args
parser = argparse.ArgumentParser(description='Convert czi to jpg images')
parser.add_argument('-i', type=str,
                   help='input file name')
parser.add_argument('-f', type=str, 
                   help='czi file format: 2018, 2019')
parser.add_argument('-odir', type=str, default="img",
                    help='output dir: e.g. img')
args = parser.parse_args()
print('input fname:', args.i)
print('input format:', args.f)


from pathlib import Path
import os
import re
import matplotlib
from ccount import read_czi, block_equalize, down_scale
Path("img/jpg").mkdir(parents=True, exist_ok=True)
Path("img/equ").mkdir(parents=True, exist_ok=True)
outname = os.path.basename(args.i)
out_img_fname = re.sub('.czi$', '.jpg', outname)
equ_img_fname = re.sub('.czi$', '.equ.jpg', outname)
oimgname = os.path.join(args.odir, "jpg", out_img_fname)
oequname = os.path.join(args.odir, "equ", equ_img_fname)
print("output_img_fname:", oimgname)
print("equalized_output_img_fname:", oequname)

# Read Equ Write
image = read_czi(args.i, Format=args.f)
matplotlib.image.imsave(oimgname, image, cmap = "gray")
image_equ = block_equalize(image, block_height=2000, block_width=2400)
matplotlib.image.imsave(oequname, image_equ, cmap = "gray")

