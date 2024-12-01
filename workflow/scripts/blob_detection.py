import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ccount.img.equalize import block_equalize
from ccount.img.read_czi import read_czi, parse_image_obj
from ccount.img.auto_contrast import uint16_image_auto_contrast

from ccount.blob.find_blob import find_blob
from ccount.blob.crop_blobs import crop_blobs
from ccount.blob.io import save_locs
from ccount.blob.plot import visualize_blobs_on_img

from pathlib import Path

import argparse, os, re, yaml
import numpy as np

"""
Input: czi
Output: locs.gz or crops.gz for all scenes available in the czi (4 or less)

params:
-i: input czi file
-c: config.yaml for blob detection params
-odir: output dir

example usage:
python workflow/scripts/blob_detection.py -i data/IL17A_POINT1_EPO_1.czi -c config.yaml -odir res/blob_locs/
"""


def parse_cmd_and_prep():
    parser = argparse.ArgumentParser(
        description='Read czi, output blobs')
    parser.add_argument('-i', type=str,
                        help='input file name: path/xxx.czi')
    parser.add_argument('-c', type=str, default="config.yaml",
                        help='path to config.yaml file')
    parser.add_argument('-odir', type=str, default="blobs",
                        help='output dir: default blobs')

    args = parser.parse_args()
    print('input fname:', args.i)
    print('config file:', args.c)
    print('output dir:', args.odir, "\n")
    Path(args.odir).mkdir(parents=True, exist_ok=True)
    corename = re.sub('.czi$', '', os.path.basename(args.i))

    with open(args.c, 'r') as stream:
        config = yaml.safe_load(stream)

    if config['blob_detection_scaling_factor'] not in [1, 2, 4]:
        raise Exception(
            'scaling_factor', config['blob_detection_scaling_factor'],
            'not implemented, only support 1,2,4'
        )

    return [args, corename, config]


[args, corename, config] = parse_cmd_and_prep()

image_obj = read_czi(args.i, Format=config['FORMAT'])

for i in range(len(image_obj.scenes)):
    i = str(i)
    out_blob_fname = os.path.join(args.odir, corename + "." + i + ".locs.npy")
    print("Searching blobs for area", i, "; output:", out_blob_fname)

    image_obj = read_czi(args.i, Format=config['FORMAT'])
    image = parse_image_obj(image_obj, i=i, Format=config['FORMAT'])
    image = uint16_image_auto_contrast(image)

    if config['test_mode']:
        print("in test mode")
        image = image[0:3000, 0:3000]

    image_equ = block_equalize(
        image,
        block_height=config['block_height'],
        block_width=config['block_width']
    )

    image_equ = image_equ.astype(np.single)  # save RAM

    blob_locs = find_blob(
        1 - image_equ,
        scaling_factor=config['blob_detection_scaling_factor'],
        max_sigma=config['max_sigma'],
        min_sigma=config['min_sigma'],
        num_sigma=config['num_sigma'],
        threshold=config['threshold'],
        overlap=config['overlap'],
    )
    print('there are {} blobs detected\n'.format(blob_locs.shape[0]))

    save_locs(blob_locs, out_blob_fname)

    # Visualizing filtered blobs
    Path(os.path.join(args.odir, "vis_blob_detection")).mkdir(parents=True, exist_ok=True)
    out_img_fname = os.path.join(args.odir, "vis_blob_detection", corename + "." + i + ".jpg")
    print("output_img_fname:", out_img_fname)
    visualize_blobs_on_img(image, blob_locs,
                             blob_extention_ratio=config['blob_extention_ratio'],
                             blob_extention_radius=config['blob_extention_radius'],
                             fname=out_img_fname
                             )

# test: skipped edge filter (we have neg plates now), 08/23/21
# good_flats = remove_edge_crops(crops)
# print("there are {} blobs passed edge filter".format(len(good_flats)))
# good_flats = crops
