#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from ccount import load_blobs_db, show_rand_crops
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
from os import environ, path
import sys, os


print("usage: python view_npy.py <path_to_area.npy.gz> <label_filter> <num_shown> <seed>")
print("<label_filter>: only blobs with the corresponding label (0/1) will be plotted, 'na' will skip filtering")
print("<num_shown>: num of blobs to be plotted, if this number larger than filtered blobs, all will be plotted")
print("<seed>: seed to reproduce same ramdom sampling from filtered blobs, e.g. 1, 2, 123, etc")
print("\n")

pwd = os.getcwd()
print("Work Dir:", pwd)

if len(sys.argv) == 5:
	in_name = sys.argv[1]
	label_filter = sys.argv[2]
	num_shown = int(sys.argv[3])
	seed = int(sys.argv[4])
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
#in_name = "/home/rl44w/mount/ccount/analysis/develop/res/classification1/area/NO_EPO_1_FIRST_SCAN-Stitching-10.1.area.txt.npy.gz"



out_name = in_name.replace(".area.txt.npy.gz", ".label"+str(label_filter)+".seed"+str(seed))
print("\n", in_name, "->", out_name+'.labelX.seedX.rndX.png')

# Please don't change unless have to
block_height = 2048 
block_width = 2048 # pixcels

blob_extention_ratio = 1.4 # extend blob radius manually (1.4)
blob_extention_radius = 30 # pixcels to extend (2)

# load
image_flat_crops = load_blobs_db(in_name)
w = int(sqrt(image_flat_crops.shape[1]-6)) # padding width & cropped img width/2


r_ = image_flat_crops[:,2]
plt.hist(r_, 40)
plt.show()


show_rand_crops(crops=image_flat_crops, label_filter=label_filter, num_shown=num_shown, fname=out_name, plot_area=True, seed = seed)
print("Plotting finished")
