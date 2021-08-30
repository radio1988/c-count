from ccount.img.read_czi import read_czi, parse_image_arrays

from ccount.blob.io import save_crops, load_crops
from ccount.blob.plot import visualize_blob_detection

from pathlib import Path

import argparse, os, re, yaml
import numpy as np


def parse_cmd_and_prep ():
    parser = argparse.ArgumentParser(
        description='\
        Read czi and locs/crops, \
        output images with blobs circled; \
        CMD: visualize_locs_on_czi.py -locs locs.npy.gz \
        -czi image.czi  -index 0  -config config.yaml \
        -output image.blobs.jpg')
    parser.add_argument('-crops', type=str,
        help='locs/crops filename: path/xxx.npy.gz')
    parser.add_argument('-czi', type=str,
        help='czi image filename: path/xxx.czi')
    parser.add_argument('-index', type=int, default=0, 
        help='index of scanned image in czi file: 0, 1, 2, 3')
    parser.add_argument('-config', type=str, default="config.yaml", 
        help='path to config.yaml file, to get radius extension info')
    parser.add_argument('-output', type=str, default="image.czi.jpg",
        help='output image with blobs circled')

    args = parser.parse_args()
    print('-crops::', args.crops)
    print('-czi:', args.czi)
    print('-index:', args.index)
    print('-config:', args.config)
    print('-output:', args.output)

    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    return [args, config]



##################Start####################
[args, config] = parse_cmd_and_prep()

image_arrays = read_czi(args.czi)
image = parse_image_arrays(image_arrays, args.index)
image_arrays = []

crops = load_crops(args.crops)

visualize_blob_detection(
	image, crops,
	blob_extention_ratio=config['blob_extention_ratio'], 
	blob_extention_radius=config['blob_extention_radius'], 
	fname=args.output)

