 import environ
 from ccount import read_czi, block_equalize
 import numpy as np
 import matplotlib.pyplot as plt
 import os
 import re

 # Read Image
 format="2019"
 if environ.get('fname') is not None:
     fname = environ['fname']  # for runipy
 else:
     # for notebook running
                 fname = '../../ccount_data/E2F4_CFUe_14JUN19_stitching/Beta_CFUe_1-Stitching-07.czi'
                 print('fname:', fname)


                 outname = os.path.basename(fname)
                 out_img_fname = re.sub('.czi$', '.png', outname)
                 equ_img_fname = re.sub('.czi$', '.equ.png', outname)
                 print("out_img_fname", out_img_fname)
                 print("equ_img_fname", equ_img_fname)

print(fname)
image = read_czi(fname, format="2019")
image = np.divide(image, np.max(image))  # from 0-255 or any range -> 0-1
dims = np.divide(image.shape, 128) # out jpg size: 256 big, 512 mid, 1024 small

plt.figure(figsize=(dims[0],dims[1]))
plt.imshow(image, 'gray')
plt.savefig(out_img_fname)

image_equ = block_equalize(image, block_height=2000, block_width=2400)

print("Visualizing equalization")
plt.figure(figsize=(dims[0],dims[1]))
plt.imshow(image_equ, 'gray')
plt.savefig(equ_img_fname)
