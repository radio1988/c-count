import argparse, textwrap
from ccount_utils.clas import split_data
from ccount_utils.blob import load_blobs, save_crops


def parse_cmd_and_prep ():
    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(
    	formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        	>>> Usage: 
        	crops_sampling.py -crops crops.labled.npy.gz -ratio 0.1
        	>>> Output:
        	crops.0.1.npy.gz
        	'''))
    parser.add_argument("-crops", type=str,
        help="labled blob-crops file, e.g. labeled/labeled.crops.npy.gz")
    parser.add_argument("-ratio", type=float,
        help="ratio of down-sampling e.g. 0.1")
    parser.add_argument("-output", type=str,
        help="e.g. output.npy.gz")

    args = parser.parse_args()

    if args.ratio > 1 or args.ratio < 0:
    	raise ValueError('ratio should be between 0, 1')

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    return args


args = parse_cmd_and_prep()
crops = load_blobs(args.crops)

[crops1, crops2] = split_data(crops, args.ratio)
print(crops1.shape)
save_crops(crops1, args.output)
