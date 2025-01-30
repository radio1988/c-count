"""
Input:
    - crops.npy.gz
    - trained_weights.hdf5
    - config.yaml # for neural network parameters

Output:
    - crops.clas.npy.gz  # crop file with labels
"""

import sys
import os
import re
import argparse
import keras
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from ccount_utils.img import equalize
from ccount_utils.img import float_image_auto_contrast
from ccount_utils.img import down_scale
from ccount_utils.blob import load_blobs, load_blobs, save_crops, save_locs
from ccount_utils.blob import mask_blob_img
from ccount_utils.blob import get_blob_statistics, parse_crops, crop_width
from ccount_utils.clas import F1
from pyimagesearch.cnn.networks.lenet import LeNet


def parse_cmd_and_prep():
    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(
        description='perfrom classification on crops with trained weights')
    parser.add_argument("-crops", type=str,
                        help="blob-crops file, e.g. res/blob_crops/xxx.crops.npy.gz")
    parser.add_argument("-weight", type=str,
                        help="weights file, e.g. resources/weights/trained.hdf5")
    parser.add_argument("-config", type=str,
                        help="config file, e.g. config.yaml")
    parser.add_argument("-output", type=str,
                        help="output name, e.g. res/xxx.crops.clas.npy.gz")

    args = parser.parse_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    if config['clas_scaling_factor'] not in [1, 2, 4]:
        raise Exception(config['clas_scaling_factor'], 'not implemented', 'only support 1,2,4')

    return [args, config]


# START
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

args, config = parse_cmd_and_prep()

crops = load_blobs(args.crops)
w = crop_width(crops)
print("crops:", crops[0:3, 0:5])

images, labels, rs = parse_crops(crops)
print("Expanding r by", config['r_ext_ratio'], config['r_ext_pixels'])
rs = rs * config['r_ext_ratio'] + config['r_ext_pixels']

print("Downscaling images by ", config['clas_scaling_factor'])
images = np.array([down_scale(image, scaling_factor=config['clas_scaling_factor']) for image in images])
w = int(w / config['clas_scaling_factor'])
rs = rs / config['clas_scaling_factor']

# todo: more channels (scaled + equalized + original)
if config['classification_equalization']:
    print("Equalizing images...")
    images = np.array([equalize(image) for image in images])

print("Auto contrasting images...")
images = np.array([float_image_auto_contrast(image) for image in images])

print("Masking images...")
images = np.array([mask_blob_img(image, r=rs[ind]) for ind, image in enumerate(images)])

images = images.reshape((images.shape[0], 2 * w, 2 * w, 1))
print("max pixel value: ", np.max(images))
print("min pixel value: ", np.min(images))

# Initialize the optimizer and model
# todo: feature normalization (optional)
print("Compiling model...")

opt = Adam(learning_rate=config['learning_rate'])

model = LeNet.build(numChannels=1,
                    imgRows=2 * w,
                    imgCols=2 * w,
                    numClasses=config['numClasses'],
                    weightsPath=args.weight)

model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=[F1])

print('Making predictions...')
probs = model.predict(images)  # shape: (n, 2), column 0: prob of being 0, column 1: prob of being 1 (use this)

# Get classifications
# classifications = probs.argmax(axis=1)  # old
classifications = [x for x in probs[:, 1] > 0.5]  # adjustable threshold

positive_idx = [i for i, x in enumerate(classifications) if x == 1]

# Save 
print("Saving classifications..")
crops[:, 3] = classifications  # int16 in npy, so probs not useful
# crops[:, 3] = probs[:, 1]
get_blob_statistics(crops)

save_locs(crops, args.output.replace('crops', 'locs'))  # int16, todo: fix potential name bug in non-workflow situations
save_crops(crops, args.output)

txt_name = args.output.replace('.npy.gz', '.txt')
np.savetxt(txt_name, classifications, fmt='%d')
np.savetxt(txt_name.replace('txt', 'probs'), probs, fmt='%f')

