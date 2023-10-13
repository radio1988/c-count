import argparse, textwrap
from ccount.clas.split_data import split_data
from ccount.blob.io import load_crops, save_crops


def parse_cmd_and_prep ():
    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(
    	formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        	>>> Usage: 
        	crops_stats.py -crops crops.labled.npy.gz 
        	>>> Output:
                stdout
        	'''))
    parser.add_argument("-crops", type=str,
        help="labled blob-crops file, e.g. labeled/labeled.crops.npy.gz")

    args = parser.parse_args()

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    return args


args = parse_cmd_and_prep()
crops = load_crops(args.crops)
