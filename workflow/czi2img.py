import argparse
from pathlib import Path
import os
import re
import matplotlib
from ccount import read_czi, block_equalize, down_scale

# Parse args
parser = argparse.ArgumentParser(
	description='Convert czi to jpg images'
	)
parser.add_argument(
	'-i', type=str,
	help='input file name'
	)
parser.add_argument(
	'-f', type=str, 
	help='czi file format: 2018, 2019'
	)
parser.add_argument(
	'-odir', type=str, default="img",
	help='output dir: e.g. img'
	)
args = parser.parse_args()
print('input fname:', args.i)
print('input format:', args.f)
print('odir:', args.odir)

outname = os.path.basename(args.i)
oimgname = re.sub('.czi$', '.jpg', outname)
oimgname = os.path.join(args.odir, "jpg", oimgname)
oequname = re.sub('.czi$', '.equ.jpg', outname)
oequname = os.path.join(args.odir, "equ", oequname)
print("output_img_fname:", oimgname)
print("equalized_output_img_fname:", oequname)

# Prep ourdir
Path(os.path.join(args.odir, "jpg")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(args.odir, "equ")).mkdir(parents=True, exist_ok=True)

# Work and output
images = read_czi(args.i, Format=args.f, concatenation=False)
for i,image in enumerate(images):
	image_equ = block_equalize(image, block_height=2000, block_width=2400)
	
	name1 = re.sub("jpg$", str(i)+".jpg", oimgname)
	matplotlib.image.imsave(name1, image, cmap = "gray")
	
	name2 = re.sub("equ.jpg$", str(i)+".equ.jpg", oequname)
	matplotlib.image.imsave(name2, image_equ, cmap = "gray")

