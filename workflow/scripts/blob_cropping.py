"""
reads czi, locs.npy.gz, config.yaml
outputs crops.npy.gz
"""

from ccount.img.read_czi import read_czi, parse_image_obj
from ccount.img.auto_contrast import uint16_image_auto_contrast

from ccount.blob.crop_blobs import crop_blobs
from ccount.blob.io import save_crops, load_blobs
from ccount.blob.misc import get_label_statistics


from pathlib import Path
import argparse, os, re, matplotlib, subprocess, yaml


def parse_cmd_and_prep():
    # ARGS
    parser = argparse.ArgumentParser(
        description='''\
        >>> Function: read czi, blob_locs, output square crop images of each blob 
        >>> Example cmd:
        python workflow/scripts/blob_cropping.py -czi data/IL17A_POINT1_EPO_1.czi 
        -locs res/blob_locs/IL17A_POINT1_EPO_1.0.locs.npy.gz -i 0 
        -config config.yaml -o res/blob_crops/IL17A_POINT1_EPO_1.0.crops.npy.gz''')
    parser.add_argument('-czi', type=str,
                        help='czi file name: path/xxx.czi')
    parser.add_argument('-locs', type=str, default="path/blobs.locs.npy.gz",
                        help='blob_locs file name, in npy.gz format')
    parser.add_argument('-i', type=int,
                        help='area index, e.g. 1,2,3,4')
    parser.add_argument('-config', type=str,
                        help='path/config.yaml')
    parser.add_argument('-o', type=str, default="path/blobs.crops.npy.gz",
                        help='outfname, in npy.gz format')

    args = parser.parse_args()
    print('czi fname:', args.czi)
    print('blob_locs fname:', args.locs)
    print('index name:', args.i)
    print('config.yaml:', args.config)
    print('output fname:', args.o, "\n")
    corename = re.sub('.czi$', '', os.path.basename(args.czi))

    Path(os.path.dirname(args.o)).mkdir(parents=True, exist_ok=True)

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    return [args, corename, config]


[args, corename, config] = parse_cmd_and_prep()

print("\nreading image:")
image_obj = read_czi(args.czi, Format=config['FORMAT'])  # fast already
image = parse_image_obj(image_obj, i=args.i, Format=config['FORMAT'])
image = uint16_image_auto_contrast(image)
print('image.shape:', image.shape)

print("\nreading locs:")
blob_locs = load_blobs(args.locs)

print('\ncropping from image and locs:')
crops = crop_blobs(blob_locs, image, crop_width=config['crop_width'])
get_label_statistics(crops)

save_crops(crops, args.o)
