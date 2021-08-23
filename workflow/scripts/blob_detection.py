from ccount import read_czi, parse_image_arrays, block_equalize, down_scale
from ccount import find_blob, crop_blobs, remove_edge_crops, vis_blob_on_block
from pathlib import Path
import argparse, os, re, matplotlib, subprocess, yaml
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # for plotting without GUI
import numpy as np

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

    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    if config['scaling_factor'] not in [1,2,4]:
        raise Exception('scaling_factor', scaling_factor, 'not implemented',
                        'only support 1,2,4')

    # Prep output dir
    if config['visualization']:
        Path(os.path.join(args.odir, "vis")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.odir, "hist")).mkdir(parents=True, exist_ok=True)  # todo: move out
    return [args, corename, config]



def uint16_image_auto_contrast(image):
    '''
    pos image
    output forced into uint16
    max_contrast_achieved
    for 2019 format, input is also uint16
    '''
    image = image - np.min(image)  # pos image
    image = image/np.max(image) * 2**16
    image = image.astype(np.uint16)
    return image

## START ##
[args, corename, config] = parse_cmd_and_prep()


## Work ## 
image_arrays = read_czi(args.i, Format=config['FORMAT'])  # fast already
for i in range(len(image_arrays)):
    # names
    i=str(i)
    out_blob_fname = os.path.join(args.odir, corename+"."+i+".npy")
    print("Searching blobs for area", i, "; output:", out_blob_fname)
    if config['visualization']:
        out_img_fname = os.path.join(args.odir, "vis", corename+"."+i+".jpg")
        hist_img_fname = os.path.join(args.odir, "hist", corename+"."+i+".hist.pdf")
        print("output histogram:", hist_img_fname)
        print("output_img_fname:", out_img_fname)

    image_arrays = read_czi(args.i, Format=config['FORMAT'])  # fast already
    image = parse_image_arrays(image_arrays, i=i, Format=config['FORMAT'])
    image_arrays = [] # todo: release RAM

    image = uint16_image_auto_contrast(image)

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
    np.save('blob_locs.npy', blob_locs) # test
    print('there are {} blobs detected'.format(blob_locs.shape[0]))

    image_flat_crops = crop_blobs(blob_locs, image, 
                            crop_width=config['crop_width'])

    # test: skipped edge filter (we have neg plates now), 08/23/21
    # good_flats = remove_edge_crops(image_flat_crops)
    # print("there are {} blobs passed edge filter".format(len(good_flats)))
    good_flats = image_flat_crops


    # Saving, todo: split into another script to avoid RAM crash
    np.save(out_blob_fname, good_flats)
    print('saved into {}'.format(out_blob_fname))
    bashCommand = "gzip -f " + out_blob_fname
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Visualizing filtered blobs, todo: split into another script to avoid RAM crash
    if config['visualization']:
        r_ = good_flats[:,2]
        plt.hist(r_, 40)
        plt.title("Histogram of blob size")
        plt.savefig(hist_img_fname)

        vis_blob_on_block(good_flats, image_equ,image, 
            blob_extention_ratio=config['blob_extention_ratio'], 
            blob_extention_radius=config['blob_extention_radius'], 
            scaling = 2,
            fname=out_img_fname)


