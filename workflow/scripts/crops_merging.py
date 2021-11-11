from ccount.blob.io import load_crops, save_crops
from ccount.blob.misc import crops_stat
from pathlib import Path
import numpy as np
import sys, argparse, os, re, yaml

def parse_cmd_and_prep ():
    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(
        description='reads crops.npy.gz, set label, output')
    parser.add_argument("-crops", type=str,nargs='+',
                    help="blob-crops files, e.g. name1.crops.npy.gz,name2.crops.npy.gz")
    parser.add_argument("-output", type=str,
                    help="output, e.g. merged.crops.npy.gz")

    args = parser.parse_args()
    print("crops:", args.crops)
    print("output:", args.output)

    odir=os.path.dirname(args.output)
    Path(odir).mkdir(parents=True, exist_ok=True)

    return args


args = parse_cmd_and_prep()

for i, crop_name in enumerate(args.crops):
    crops = load_crops(crop_name)
    if i == 0:
        output_crops = crops
    else:
        print("Merging...")
        output_crops = np.vstack((output_crops, crops))

save_crops(output_crops, args.output)