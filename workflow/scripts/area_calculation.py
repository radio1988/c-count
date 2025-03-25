"""
area_calculation.py -crops <crops.npy.gz> -output <output.txt>

caveat: if > 65535, will have problems (as npy.gz is only 16 bits)
"""


import sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ccount_utils.blob import load_blobs, save_crops
from ccount_utils.blob import area_calculations

def parse_args():
    parser = argparse.ArgumentParser(description='Area calculation for blobs; areas.txt, areas.npy.gz, and histogram.pdf will be generated')
    parser.add_argument('-crops', type=str, required=True,
                        help='Input blob file (npy.gz)')
    parser.add_argument('-output', type=str, required=True,
                        help='Output area file (txt)')
    return parser.parse_args()

def main():
    args = parse_args()
    inname = args.crops
    outname_txt = args.output
    outname_core = outname_txt.replace('.txt', '')
    outname_hist = outname_core + '.hist.pdf'
    outname_crops = outname_core + '.npy.gz'
    print('inname: ', inname,
        "\noutname_txt: ", outname_txt,
        "\noutname_crops ", outname_crops)

    crops = load_blobs(inname)
    areas = area_calculations(crops)
    # save txt
    np.savetxt(outname_txt, areas, fmt='%i', delimiter='')
    # save crops
    crops[:, 4] = areas
    save_crops(crops, outname_crops)
    plt.hist(areas, 40)
    plt.title(outname_core)
    plt.savefig(outname_hist)
    plt.close()
    print("Area calculation completed successfully.")

if __name__ == "__main__":
    main()
