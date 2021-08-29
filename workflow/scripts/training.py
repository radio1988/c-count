from ccount.img.equalize import equalize
from ccount.img.auto_contrast import float_image_auto_contrast
from ccount.img.transform import down_scale

from ccount.blob.io import load_crops, save_crops
from ccount.blob.mask_image import mask_image
from ccount.blob.misc import crops_stat, parse_crops, crop_width


from ccount.clas.split_data import split_data
from ccount.clas.balance_data import balance_by_duplication
from ccount.clas.augment_images import augment_images
from ccount.clas.metrics import F1, F1_calculation

import sys, argparse, os, re, yaml, keras
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


from pyimagesearch.cnn.networks.lenet import LeNet
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from collections import Counter

# Show CPU/GPU info
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


def parse_cmd_and_prep ():
    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(
        description='load labeled crops.npy.gz, train model and output trained weights\n\
         python workflow/scripts/training.py -crops ~/ccount/dev2021/training_data/FL/FL.t.npy.gz \
         -config config.yaml -output test/test.hdf5')
    parser.add_argument("-crops", type=str,
                    help="labled blob-crops file, e.g. labeled/labeled.crops.npy.gz")
    parser.add_argument("-config", type=str,
                    help="config file, e.g. config.yaml")
    parser.add_argument("-output", type=str,
                    help="output weights file, e.g. resources/weights/trained.hdf5")

    args = parser.parse_args()
    print("labeled crops:", args.crops)
    print("config file:", args.config)
    print("output:", args.output)
    odir = os.path.dirname(args.output)
    print("odir:", odir)
    Path(odir).mkdir(parents=True, exist_ok=True)

    corename = args.output.replace(".hdf5", "")
    print("output corename:", corename)
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    if config['clas_scaling_factor'] not in [1,2,4]:
        raise Exception(config['clas_scaling_factor'], 'not implemented', 'only support 1,2,4')

    return [args, corename, config]


args, corename, config = parse_cmd_and_prep()

crops = load_crops(args.crops)
w = crop_width(crops)

print("Removing unlabeled crops (label == 5)")
crops = crops[crops[:, 3] != 5, :]
crops_stat(crops)

# set other laberls as no
if config['numClasses'] == 2:
    print("Remove uncertain and artifacts")  # todo: user decide
    crops[crops[:, 3] == 3, 3] = 0  # uncertain
    crops[crops[:, 3] == 4, 3] = 0  # artifacts, see ccount.blob.readme.txt
    crops_stat(crops)


[train_crops, val_crops] = split_data(crops, config['training_ratio'])
print("{} Split ratio, split into {} training crops and {} validating crops".\
      format(config['training_ratio'], train_crops.shape[0], val_crops.shape[0]))

print('For training split:')
train_crops = balance_by_duplication(train_crops)
print('For validation split:')
val_crops = balance_by_duplication(val_crops)  #todo: skip this if F1 working well

trainimages, trainlabels, trainrs = parse_crops(train_crops)
valimages, vallabels, valrs = parse_crops(val_crops)

trainrs = trainrs * config['r_ext_ratio'] + config['r_ext_ratio']
valrs = valrs * config['r_ext_ratio'] + config['r_ext_ratio']

print("Before Aug:", trainimages.shape, trainrs.shape, trainlabels.shape)
trainimages = augment_images(trainimages, config['aug_sample_size'])  # todo: augment to more samples

## match sample size of labels and rs with augmented images
while trainrs.shape[0] < config['aug_sample_size']:
    trainrs = np.concatenate((trainrs, trainrs))
    trainlabels = np.concatenate((trainlabels, trainlabels))
trainrs = trainrs[0:config['aug_sample_size']]
trainlabels = trainlabels[0:config['aug_sample_size']]

print("After Aug:", trainimages.shape, trainrs.shape, trainlabels.shape)
print('pixel value max', np.max(trainimages), 'min', np.min(trainimages))

print("Downscaling images by ", config['scaling_factor'])
trainimages = np.array([down_scale(image, scaling_factor=config['scaling_factor']) for image in trainimages])
valimages = np.array([down_scale(image, scaling_factor=config['scaling_factor']) for image in valimages])
w = int(w/config['scaling_factor'])
trainrs = trainrs/config['scaling_factor']
valrs = valrs/config['scaling_factor']

# todo: more channels (scaled + equalized + original)
if config['classification_equalization']:
    print("Equalizing images...")
    trainimages = np.array([equalize(image) for image in trainimages])
    valimages = np.array([equalize(image) for image in valimages])

print("Normalizing images...")
trainimages = np.array([float_image_auto_contrast(image) for image in trainimages])
valimages = np.array([float_image_auto_contrast(image) for image in valimages])

print("Masking images...")
trainimages = np.array([mask_image(image, r=trainrs[ind]) for ind, image in enumerate(trainimages)])
valimages = np.array([mask_image(image, r=valrs[ind]) for ind, image in enumerate(valimages)])

# Reshape for model
trainimages = trainimages.reshape((trainimages.shape[0], 2*w, 2*w, 1))
valimages = valimages.reshape((valimages.shape[0], 2*w, 2*w, 1))
print("max pixel value: ", np.max(trainimages))
print("min pixel value: ", np.min(trainimages))

# Categorize labels for softmax
trainlabels2 = np_utils.to_categorical(trainlabels, config['numClasses'])
vallabels2 = np_utils.to_categorical(vallabels, config['numClasses'])

# Initialize the optimizer and model
# todo: feature normalization (optional)
print("[INFO] compiling model...")
opt = Adam(lr=config['learning_rate'])
model = LeNet.build(numChannels=1, imgRows=2*w, imgCols=2*w, numClasses=config['numClasses'],
                    weightsPath=None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=[F1])
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0, patience=config['patience'],
                                          verbose=config['verbose'], mode='auto',
                                          baseline=None, restore_best_weights=True)
callbacks_list = [earlystop]

print("[INFO] training...")
# todo: add radius to model
# todo: augmentation in batch training

# todo augmentation here is bad, mask issue
datagen = ImageDataGenerator(
    featurewise_std_normalization=False,
    rotation_range=90,
    shear_range=0.16,
    zoom_range=0.1,
    width_shift_range=0.03, height_shift_range=0.03,
    horizontal_flip=True, vertical_flip=True
)

datagen.fit(trainimages)

# model.fit(trainimages, trainlabels, validation_data=(valimagesMsk, vallabels),
#           batch_size=config['batch_size'], epochs=config['epochs'],
#           verbose=config['verbose'])

model.fit_generator(datagen.flow(trainimages, trainlabels2, batch_size=config['batch_size']),
                    validation_data=(valimages, vallabels2),
                    steps_per_epoch=len(trainimages) / config['batch_size'], epochs=config['epochs'],
                    callbacks=callbacks_list,
                    verbose=config['verbose'])


# Evaluation of the model
print("[INFO] evaluating...")
(loss, f1) = model.evaluate(trainimages, trainlabels2,
                                  batch_size=config['batch_size'], verbose=config['verbose'])
print("[INFO] training F1: {:.2f}%".format(f1 * 100))

print("[INFO] evaluating...")
(loss,  f1) = model.evaluate(valimages, vallabels2,
                                  batch_size=config['batch_size'], verbose=config['verbose'])
print("[INFO] validation F1: {:.2f}%".format(f1 * 100))


print("[INFO] dumping weights to file...")
model.save_weights(args.output, overwrite=True)
