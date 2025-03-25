import argparse, os, re, yaml, textwrap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
from ccount_utils.img import read_czi, parse_image_obj
from ccount_utils.blob import save_crops, load_blobs
from ccount_utils.blob import intersect_blobs
from ccount_utils.blob import visualize_blobs_on_img, visualize_blob_compare
from pathlib import Path


def parse_cmd_and_prep():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
                    Read czi and locs/crops, 
                    output images with blobs circled; 

                    CMD1: 
                    visualize_locs_on_czi.py 
                    -crops locs.npy.gz  
                    -czi image.czi  
                    -index 0  
                    -output image.blobs.jpg (gray circles)
                    
                    CMD2: 
                    visualize_locs_on_czi.py 
                    -crops locs.npy.gz 
                    -crops2 ground_truth.locs.npy.gz       
                    -czi image.czi  
                    -index 0  
                    -output image.blobs.jpg (colored circles for TN, FP, FN, TP)
                    '''))
    parser.add_argument('-crops', type=str,
                        help='locs/crops filename: path/xxx.npy.gz')
    parser.add_argument('-crops2', type=str,
                        help='locs/crops filename, optional: path/ground_truth.npy.gz')
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
    print('-crops2::', args.crops2)
    print('-czi:', args.czi)
    print('-index:', args.index)
    print('-output:', args.output)

    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)

    # with open(args.config, 'r') as stream:
    #     config = yaml.safe_load(stream)

    return [args]


def main():
    args = parse_cmd_and_prep()
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    image_obj = read_czi(args.czi, Format=config['FORMAT'])
    image = parse_image_obj(image_obj, args.index)
    crops = load_blobs(args.crops)

    if args.crops2 is None:
        visualize_blobs_on_img(
            image, crops,
            fname=args.output)

    if args.crops2 is not None:
        crops2 = load_blobs(args.crops2)
        crops, crops2 = intersect_blobs(crops, crops2)
        visualize_blob_compare(
            image, crops, crops2,
            fname=args.output)


if __name__ == "__main__":
    main()
