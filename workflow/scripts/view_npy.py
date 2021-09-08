#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from ccount.blob.io import load_crops
from ccount.blob.plot import show_rand_crops
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
from os import environ, path
import sys, os
# todo: clean code, matching style

print("usage: python view_npy.py <path_to_area.npy.gz> <label_filter> <num_shown> <seed>")
# todo: area filter
print("<label_filter>: only blobs with the corresponding label (0/1) will be plotted, 'na' will skip filtering")
print("<num_shown>: num of blobs to be plotted, if this number larger than filtered blobs, all will be plotted")
print("<seed>: seed to reproduce same ramdom sampling from filtered blobs, e.g. 1, 2, 123, etc")
print("\n")

pwd = os.getcwd()
print("Work Dir:", pwd)

if len(sys.argv) == 6:
	in_name = sys.argv[1]
	label_filter = sys.argv[2]
	num_shown = int(sys.argv[3])
	seed = int(sys.argv[4])
	out_name = sys.argv[5]
	if path.exists(in_name):
		pass
	else:
		in_name = os.path.join(pwd, in_name)
		in_name = os.path.abspath(in_name)
	if path.exists(in_name):
		print("npy_file_name:", in_name)
	else:
		sys.exit("File not found error:", in_name)
else:
	sys.exit("cmd err")



print(in_name, "->", out_name)


# load
image_flat_crops = load_crops(in_name)

show_rand_crops(crops=image_flat_crops, label_filter=label_filter, num_shown=num_shown, fname=out_name, seed = seed)
print("Plotting finished")
