from ccount.img.equalize import block_equalize
from ccount.img.read_czi import read_czi, parse_image_arrays
from ccount.img.auto_contrast import uint16_image_auto_contrast

from ccount.blob.find_blob import find_blob
from ccount.blob.crop_blobs import crop_blobs
from ccount.blob.io import save_crops
from ccount.blob.plot import visualize_blob_detection

from pathlib import Path

import argparse, os, re, yaml
import numpy as np

# usage: 
# python workflow/scripts/blob_detection.py -i data/IL17A_POINT1_EPO_1.czi \
# -c config.yaml -odir res/blob_locs/

def parse_cmd_and_prep ():
    parser = argparse.ArgumentParser(
        description='Read czi, output blobs')
    parser.add_argument('-i', type=str,
        help='input file name: path/xxx.czi')
    parser.add_argument('-c', type=str, default="config.yaml", 
        help='path to config.yaml file')
    parser.add_argument('-odir', type=str, default="blobs",
        help='output dir: default blobs'        )

    args = parser.parse_args()
    print('input fname:', args.i)
    print('config file:', args.c)
    print('output dir:', args.odir, "\n")
    Path(args.odir).mkdir(parents=True, exist_ok=True)
    corename = re.sub('.czi$', '', os.path.basename(args.i))

    with open(args.c, 'r') as stream:
        config = yaml.safe_load(stream)
    if config['scaling_factor'] not in [1,2,4]:
        raise Exception('scaling_factor', scaling_factor, 'not implemented',
                        'only support 1,2,4')

    return [args, corename, config]


##################Start####################
[args, corename, config] = parse_cmd_and_prep()

image_arrays = read_czi(args.i, Format=config['FORMAT'])  # fast already
for i in range(len(image_arrays)):
    # names
    i=str(i)
    out_blob_fname = os.path.join(args.odir, corename+"."+i+".locs.npy")
    print("Searching blobs for area", i, "; output:", out_blob_fname)

    image_arrays = read_czi(args.i, Format=config['FORMAT'])  # fast already
    image = parse_image_arrays(image_arrays, i=i, Format=config['FORMAT'])
    image_arrays = [] # todo: release RAM, 08/21/21 use ~10G

    image = uint16_image_auto_contrast(image) # still uint16

    if config['test_mode']:
        print("in test mode")
        image = image[0:3000, 0:3000]

    image_equ = block_equalize(
        image, 
        block_height=config['block_height'], 
        block_width=config['block_width'])
    image_equ = image_equ.astype(np.single)  # save RAM

    blob_locs = find_blob (
        1-image_equ, 
        scaling_factor=config['scaling_factor'],
        max_sigma=config['max_sigma'], 
        min_sigma=config['min_sigma'], 
        num_sigma=config['num_sigma'], 
        threshold=config['threshold'], 
        overlap=config['overlap'],
        )
    print('there are {} blobs detected'.format(blob_locs.shape[0]))
    save_crops(blob_locs, out_blob_fname)


    # Visualizing filtered blobs
    # todo: split into another script to avoid RAM crash
    if config['blob_detection_visualization']:
        Path(os.path.join(args.odir, "vis_blob_detection")).mkdir(parents=True, exist_ok=True)
        out_img_fname = os.path.join(args.odir, "vis_blob_detection", corename+"."+i+".jpg")
        print("output_img_fname:", out_img_fname)
        visualize_blob_detection(image, blob_locs,
            blob_extention_ratio=config['blob_extention_ratio'], 
            blob_extention_radius=config['blob_extention_radius'], 
            fname=out_img_fname)

    # test: skipped edge filter (we have neg plates now), 08/23/21
    # good_flats = remove_edge_crops(image_flat_crops)
    # print("there are {} blobs passed edge filter".format(len(good_flats)))
    # good_flats = image_flat_crops





