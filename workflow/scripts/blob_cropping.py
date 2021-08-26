# todo: get square crops from blob_locs and czi
from ccount import uint16_image_auto_contrast
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
    parser.add_argument('-locs', type=str, default="path/blobs.locs.npy.gz", 
        help='blob_locs file name, in npy.gz format')
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
    if config['scaling_factor'] not in [1,2,4]:
        raise Exception('scaling_factor', scaling_factor, 'not implemented',
                        'only support 1,2,4')

    return [args, corename, config]



##################Start####################
[args, corename, config] = parse_cmd_and_prep()

# image czi
image_arrays = read_czi(args.czi, Format=config['FORMAT'])  # fast already
image = parse_image_arrays(image_arrays, i=args.i, Format=config['FORMAT'])
image_arrays = [] # todo: release RAM  08/21/21 use ~10G
image = uint16_image_auto_contrast(image)
print('image.shape:', image.shape)

# blob_locs
blob_locs = load_from_npygz(args.locs)
print('blob_locs.shape:', blob_locs.shape)

# cropping # test
image_flat_crops = crop_blobs(blob_locs[1:100], image, 
                        crop_width=config['crop_width'])
print('image_flat_crops.shape:', image_flat_crops.shape)


save_into_npygz(image_flat_crops, args.o)