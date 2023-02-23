from ccount.img.read_czi import read_czi, parse_image_obj
from ccount.img.auto_contrast import uint16_image_auto_contrast

from pathlib import Path
from matplotlib.pyplot import imsave
import argparse, os, re, yaml


def parse_cmd_and_prep ():
	parser = argparse.ArgumentParser(
		description='Convert czi to jpg images')
	parser.add_argument('-i', type=str,
		help='input file name')
	parser.add_argument('-c', type=str, default="config.yaml", 
	    help='path to config.yaml file')
	parser.add_argument('-odir', type=str, default="img",
		help='output dir: e.g. img')
	args = parser.parse_args()
	print('input fname:', args.i)
	print('config fname::', args.c)
	print('output dir:', args.odir)

	corename = os.path.basename(args.i)
	corename = re.sub('.czi$', '.jpg', corename)
	ofname = os.path.join(args.odir, corename)
	Path(args.odir).mkdir(parents=True, exist_ok=True)

	with open(args.c, 'r') as stream:
	    config = yaml.safe_load(stream)

	return [args, config, ofname]



##################Start####################
[args, config, ofname] = parse_cmd_and_prep()

image_obj = read_czi(args.i, Format=config['FORMAT'])
for i in range(len(img_obj.scenes)):
	print('For area', i)
	image_obj = read_czi(args.i, Format=config['FORMAT'])
	image = parse_image_obj(image_obj, i=i, Format=config['FORMAT'])
	image = uint16_image_auto_contrast(image) # still uint16
	
	ofname_i = re.sub("jpg$", str(i)+".jpg", ofname)
	imsave(ofname_i, image, cmap = "gray")
	print('saved into', ofname_i)
