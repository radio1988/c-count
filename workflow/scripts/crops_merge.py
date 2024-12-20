import argparse
import os
import sys
import numpy as np
from pathlib import Path
from ccount_utils.blob import get_blob_statistics
from ccount_utils.blob import load_blobs, save_crops


def parse_cmd_and_prep():
    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(
        description='Input: blobs.npy.gz files, Output: merged.npy.gz')
    parser.add_argument("-crops", type=str, nargs='+',
                        help="blob files, labeled_blobs(yxrL), \
                        labeled_crops(yxrL+crops) formats both okay")
    parser.add_argument("-output", type=str,
                        help="output, e.g. merged.crops.npy.gz")

    args = parser.parse_args()
    print("\nCommand typed: ", " ".join(sys.argv), "\n")
    print("{} input-blobs:".format(len(args.crops)), args.crops)
    print("output:", args.output)

    odir = os.path.dirname(args.output)
    Path(odir).mkdir(parents=True, exist_ok=True)

    return args

def main():
    args = parse_cmd_and_prep()

    for i, crop_name in enumerate(args.crops):
        crops = load_blobs(crop_name)
        if i == 0:
            output_crops = crops
        else:
            print("Merging...")
            output_crops = np.vstack((output_crops, crops))
            get_blob_statistics(output_crops)

    save_crops(output_crops, args.output)

if __name__ == "__main__":
    main()
