import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ccount.img.equalize import block_equalize
from ccount.img.read_czi import read_czi, parse_image_obj
from ccount.img.auto_contrast import uint16_image_auto_contrast

from ccount.blob.find_blobs import find_blobs
from ccount.blob.crop_blobs import crop_blobs
from ccount.blob.io import save_locs
from ccount.blob.plot import visualize_blobs_on_img

from pathlib import Path

import argparse, os, re, yaml
import numpy as np

"""
Input: 
    plate image in czi format
    e.g. {plateName}.czi
Output: 
    {input_prefix}.{sceneIndex}.npy.gz
    the blobs detected for that specific scene
    there are usually multiple scenes scanned for a plate

params:
-input: input czi file
-sceneIndex: {0,1,2,3} for Merav lab czi file
-config: config.yaml for blob detection params

-odir: output dir

example usage:
python workflow/scripts/blob_detection.py -input data/IL17A_POINT1_EPO_1.czi -sceneIndex 0 \
       -config config.yaml -odir res/blob_locs/
"""


def parse_cmd_and_prep():
    parser = argparse.ArgumentParser(
        prog='blob_detection.singleScene.py',
        description='Read czi, detect blobs for one scene, save in a locs.npy.gz file'
    )
    parser.add_argument(
        '-input', type=str,
        help='input file name, e.g. path/xxx.czi'
    )
    parser.add_argument(
        '-sceneIndex', type=str,
        help='sceneIndex: {0,1,2,3}, e.g. 0'
    )
    parser.add_argument(
        '-config', type=str, default="config.yaml",
        help='path to config.yaml file'
    )
    parser.add_argument(
        '-odir', type=str, default="res/blobs",
        help='output dir, e.g. res/blobs'
    )

    args = parser.parse_args()

    Path(args.odir).mkdir(parents=True, exist_ok=True)

    plateName = re.sub('.czi$', '', os.path.basename(args.input))

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    if config['blob_detection_scaling_factor'] not in [1, 2, 4]:
        raise Exception(
            'scaling_factor', config['blob_detection_scaling_factor'],
            'not implemented, only support 1,2,4'
        )

    return [args, plateName, config]


### START
[args, plateName, config] = parse_cmd_and_prep()
print('<blob_detection.singleScene.py>')
print(args)

image_obj = read_czi(args.input, Format=config['FORMAT'])

i = args.sceneIndex

out_blob_fname = os.path.join(
    args.odir,
    plateName + "." + i + ".locs.npy.gz"
)
print("Searching blobs in scene", i)
print("output:", out_blob_fname)

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

blob_locs = find_blobs(
    1 - image_equ,
    scaling_factor=config['blob_detection_scaling_factor'],
    max_sigma=config['max_sigma'],
    min_sigma=config['min_sigma'],
    num_sigma=config['num_sigma'],
    threshold=config['threshold'],
    overlap=config['overlap'],
)

save_locs(blob_locs, out_blob_fname)

# Visualizing detected blobs
print('<visualizing blob detection> ')
Path(os.path.join(args.odir, "view")).mkdir(parents=True, exist_ok=True)
out_img_fname = os.path.join(args.odir, "view", plateName + "." + i + ".jpg")
print("output_img_fname:", out_img_fname)
visualize_blobs_on_img(
    image, blob_locs,
    blob_extention_ratio=config['blob_extention_ratio'],
    blob_extention_radius=config['blob_extention_radius'],
    fname=out_img_fname
)

# dev notes:
# test: skipped edge filter (we have neg plates now), 08/23/21
# good_flats = remove_edge_crops(crops)
