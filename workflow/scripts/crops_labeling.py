from ccount_utils.blob import load_blobs, save_crops
from ccount_utils.blob import get_blob_statistics
import sys, argparse, os, re, yaml
from pathlib import Path

def parse_cmd_and_prep ():
    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(
        description='reads crops.npy.gz, set label, output')
    parser.add_argument("-crops", type=str,
                    help="blob-crops file, e.g. res/blob_crops/xxx.crops.npy.gz")
    parser.add_argument("-label", type=int,
                    help="label to be applied to all crops in the file, e.g. 0")
    parser.add_argument("-output", type=str,
                    help="output, e.g. labeled/neg.npy.gz")

    args = parser.parse_args()
    print("crops:", args.crops)
    print("label:", args.label)
    print("output:", args.output)

    odir=os.path.dirname(args.output)
    Path(odir).mkdir(parents=True, exist_ok=True)

    return args


args = parse_cmd_and_prep()
crops = load_blobs(args.crops)
print(">>> changing all labels to:", args.label)
crops[:,3] = [args.label] * crops.shape[0]
get_blob_statistics(crops)
save_crops(crops, args.output)
