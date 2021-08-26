from ccount import block_equalize
from ccount.img.read_czi import read_czi, parse_image_arrays
from ccount.img.uint16_image_auto_contrast import uint16_image_auto_contrast


from ccount import crop_blobs, load_from_npygz, save_into_npygz
from pathlib import Path
import argparse, os, re, matplotlib, subprocess, yaml


# Example Usage:
# python workflow/scripts/blob_cropping.py -czi data/IL17A_POINT1_EPO_1.czi -i 0 -locs test/IL17A_POINT1_EPO_1.0.npy.gz -config config.yaml -o test/crops/IL17A_POINT1_EPO_1.0.crops.npy.gz


def parse_cmd_and_prep ():
    # ARGS
    parser = argparse.ArgumentParser(
        description='Read czi, blob_locs, output square crop images of each blob')
    parser.add_argument('-czi', type=str,
        help='czi file name: path/xxx.czi')
    parser.add_argument('-i', type=int,
        help='area index, e.g. 1,2,3,4')



from ccount import save_into_npygz
from ccount import find_blob, crop_blobs, remove_edge_crops, vis_blob_on_block
from pathlib import Path
import argparse, os, re, yaml
import numpy as np

# usage: 
# python workflow/scripts/blob_detection.py -i data/IL17A_POINT1_EPO_1.czi \
# -c config.yaml -odir res/blob_locs/

def parse_cmd_and_prep ():
    # ARGS
    parser = argparse.ArgumentParser(
        description='Read czi, output blobs')
    parser.add_argument('-i', type=str,
        help='input file name: path/xxx.czi')
    parser.add_argument('-c', type=str, default="config.yaml", 
        help='path to config.yaml file')
    parser.add_argument('-odir', type=str, default="blobs",
        help='output dir: default blobs'
        )

    args = parser.parse_args()
    print('input fname:', args.i)
    print('config file:', args.c)
    print('output dir:', args.odir, "\n")
    corename = re.sub('.czi$', '', os.path.basename(args.i))

    with open(args.c, 'r') as stream:
        config = yaml.safe_load(stream)
    if config['scaling_factor'] not in [1,2,4]:
        raise Exception('scaling_factor', scaling_factor, 'not implemented',
                        'only support 1,2,4')

    # Prep output dir
    if config['visualization']:
        Path(os.path.join(args.odir, "vis")).mkdir(parents=True, exist_ok=True)
    return [args, corename, config]


##################Start####################
[args, corename, config] = parse_cmd_and_prep()

image_arrays = read_czi(args.i, Format=config['FORMAT'])  # fast already
for i in range(len(image_arrays)):
    # names
    i=str(i)
    out_blob_fname = os.path.join(args.odir, corename+"."+i+".npy")
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
    save_into_npygz(blob_locs, out_blob_fname)


    # Visualizing filtered blobs
    # todo: split into another script to avoid RAM crash
    if config['visualization']:
        out_img_fname = os.path.join(args.odir, "vis", corename+"."+i+".jpg")
        print("output_img_fname:", out_img_fname)
        vis_blob_on_block(blob_locs, image_equ,image, 
            blob_extention_ratio=config['blob_extention_ratio'], 
            blob_extention_radius=config['blob_extention_radius'], 
            scaling = 2,
            fname=out_img_fname)

    # test: skipped edge filter (we have neg plates now), 08/23/21
    # good_flats = remove_edge_crops(image_flat_crops)
    # print("there are {} blobs passed edge filter".format(len(good_flats)))
    # good_flats = image_flat_crops





