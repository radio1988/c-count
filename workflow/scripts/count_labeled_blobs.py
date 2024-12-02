import argparse, sys, os
import pandas as pd
from ccount_utils.blob import load_blobs, get_blob_statistics

"""
Input: labeled locs files (also accept any labeled blob files)
output: count.csv

aggregate label count in the jpg2npy.Snakemake workflow
currently, it parses log file to find 'Positive' nums

e.g. 
python workflow/scripts/aggr_count_info.py.py \
res/label_crops/log/point0625unitsEpo_4-Stitching-24.1.label.npy.gz.log \
...\
output.csv
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Input: labeled_locs files, Output: count.csv")
    parser.add_argument(
        '-input',
        nargs='+',
        required=True,
        type=str,
        help="List of labeled_locs files, e.g. -input A.npy.gz B.npy.gz"
    )
    parser.add_argument(
        '-output', type=str,
        required=True,
        help="Output file name, e.g. res/count.csv"
    )
    args = parser.parse_args()
    print("\nCommand received: ", " ".join(sys.argv))
    return args


def main():
    print("<aggr_label_count>")
    args = parse_args()
    print("num of inputs:{}".format(len(args.input)))

    print('processing..')
    names = []
    counts = []
    for file in args.input:
        print(file)
        blobs = load_blobs(file)
        stats = get_blob_statistics(blobs)
        count = stats['positive']
        name = file.replace('.npy.gz', '')
        counts.append(count)
        names.append(name)
    count_df = pd.DataFrame(zip(names, counts), columns=['NAME', 'COUNT'])
    count_df.to_csv(args.output)
    return count_df


if __name__ == '__main__':
    main()
