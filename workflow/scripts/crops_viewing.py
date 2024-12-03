# %matplotlib inline
import sys, os
from os import path
from ccount_utils.blob import load_blobs
from ccount_utils.blob import show_rand_crops

if len(sys.argv) == 6:
    print("> work_dir:", os.getcwd())
    print("> cmd:", sys.argv)

    in_name = sys.argv[1]
    label_filter = sys.argv[2]
    num_shown = int(sys.argv[3])
    seed = int(sys.argv[4])
    out_name = sys.argv[5]

    if path.exists(in_name):
        print(in_name, "->", out_name + ".rnd{}.jpg".format(num_shown))
    else:
        sys.exit("input not found error:", in_name)
else:
    print(
        "usage: python view_npy.py <crops.npy.gz> <label_filter[0/1]> <num_shown[int]> " +
        "<rand-seed[num]> <output-suffix[string]>")
    print("example: python crops_view.py  test.npy.gz  1  25  0  test_yes ")
    sys.exit("cmd err")

crops = load_blobs(in_name)

show_rand_crops(crops=crops,
                label_filter=label_filter,
                num_shown=num_shown,
                fname=out_name,
                seed=seed)

print("plotting finished")
