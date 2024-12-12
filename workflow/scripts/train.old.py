from ccount_utils.img import equalize
from ccount_utils.img import float_image_auto_contrast
from ccount_utils.img import down_scale

from ccount_utils.blob import load_blobs, save_crops
from ccount_utils.blob import mask_blob_img
from ccount_utils.blob import get_blob_statistics, parse_crops, crop_width

from ccount_utils.clas import split_data
from ccount.clas.balance_data import balance_by_duplication
from ccount.clas.augment_images import augment_crops
from ccount_utils.clas import F1, F1_calculation

import sys, argparse, os, re, yaml, keras, textwrap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pyimagesearch.cnn.networks.lenet import LeNet
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter


# Show CPU/GPU info
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


def parse_cmd_and_prep():
    # Construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Function: Load labeled crops.npy.gz, train model and output trained weights
            Usage: python workflow/scripts/training.py 
            -crops_train labeled.train.npy.gz 
            -crops_val labeled.validation.npy.gz
            -config config.yaml 
            -output trained/trained.weights.h5
            '''))
    parser.add_argument("-crops_train", type=str,
                        help="labled blob-crops file for training, \
        e.g. labeled.train.npy.gz ")
    parser.add_argument("-crops_val", type=str,
                        help="labled blob-crops file for earlystop, \
        e.g. labeled.validation.npy.gz")
    parser.add_argument("-config", type=str,
                        help="config file, e.g. config.yaml")
    parser.add_argument("-output", type=str,
                        help="output weights file, e.g. resources/weights/trained.weights.h5")

    args = parser.parse_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))  # pring cmd
    odir = os.path.dirname(args.output)
    print("odir:", odir)
    Path(odir).mkdir(parents=True, exist_ok=True)

    if not args.output.endswith('.weights.h5'):
        raise Exception('output name does not end with .weights.h5')
    corename = args.output.replace(".weights.h5", "")
    print("output corename:", corename)

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    if config['clas_scaling_factor'] not in [1, 2, 4]:
        raise Exception(config['clas_scaling_factor'], 'not implemented', 'only support 1,2,4')

    return [args, corename, config]


def cleanup_crops(crops):
    print("Removing unlabeled crops (labelled as 5)")
    unlabeled_idx = crops[:, 3] == 5
    print('num unlabeled crops: {}'.format(sum(unlabeled_idx)))
    crops = crops[~unlabeled_idx, :]
    get_blob_statistics(crops)

    print("Removing uncertain crops (labelled as 3)")
    uncertain_idx = crops[:, 3] == 3  # uncertain
    print('num uncertain crops: {}'.format(sum(uncertain_idx)))
    crops = crops[~uncertain_idx, :]
    get_blob_statistics(crops)

    if config['numClasses'] == 2:
        print("Set artifacts (labelled as 4) as NEG")
        artifacts_idx = crops[:, 3] == 4
        print('num artifacts: {}'.format(sum(artifacts_idx)))
        crops[artifacts_idx, 3] = 0  # artifacts, see ccount.blob.readme.txt
        get_blob_statistics(crops)

    return crops

# parse cmd
args, corename, config = parse_cmd_and_prep()

# read data
train_crops = load_blobs(args.crops_train)
val_crops = load_blobs(args.crops_val)
w = crop_width(train_crops)
train_crops = cleanup_crops(train_crops)
val_crops = cleanup_crops(val_crops)

print("Training Crops:")
get_blob_statistics(train_crops)
print("Val Crops:", flush=True)
get_blob_statistics(val_crops)

# balancing
if config['balancing']:
    print('Balancing for training split:')
    train_crops = balance_by_duplication(train_crops, maxN=config['aug_sample_size'])
    # balance so yes = maxN//2, no = maxN//2

trainimages, trainlabels, trainrs = parse_crops(train_crops)
valimages, vallabels, valrs = parse_crops(val_crops)

trainrs = trainrs * config['r_ext_ratio'] + config['r_ext_ratio']
valrs = valrs * config['r_ext_ratio'] + config['r_ext_ratio']

# augmentation
print("Before Aug:", trainimages.shape, trainrs.shape, trainlabels.shape)
trainimages, trainlabels, trainrs = augment_crops(trainimages, trainlabels, trainrs,
                                                  config['aug_sample_size'])
print("After Aug:", trainimages.shape, trainrs.shape, trainlabels.shape)
print('pixel value max', np.max(trainimages), 'min', np.min(trainimages))

# downscale
print("Downscaling images by ", config['clas_scaling_factor'])
trainimages = np.array([down_scale(image, scaling_factor=config['clas_scaling_factor'])
                        for image in trainimages])
valimages = np.array([down_scale(image, scaling_factor=config['clas_scaling_factor'])
                      for image in valimages])
w = int(w / config['clas_scaling_factor'])
trainrs = trainrs / config['clas_scaling_factor']
valrs = valrs / config['clas_scaling_factor']

# equalization (skipped)
if config['classification_equalization']:
    print("Equalizing images...") # todo: more channels (scaled + equalized + original)
    trainimages = np.array([equalize(image) for image in trainimages])
    valimages = np.array([equalize(image) for image in valimages])

print("Normalizing images...")
trainimages = np.array([float_image_auto_contrast(image) for image in trainimages])
valimages = np.array([float_image_auto_contrast(image) for image in valimages])

print("Masking images...")
trainimages = np.array([mask_blob_img(image, r=trainrs[ind])
                        for ind, image in enumerate(trainimages)])
valimages = np.array([mask_blob_img(image, r=valrs[ind])
                      for ind, image in enumerate(valimages)])

# Reshape for model
trainimages = trainimages.reshape((trainimages.shape[0], 2 * w, 2 * w, 1))
valimages = valimages.reshape((valimages.shape[0], 2 * w, 2 * w, 1))
print("max pixel value: ", np.max(trainimages))
print("min pixel value: ", np.min(trainimages))

# Categorize labels for softmax
trainlabels2 = to_categorical(trainlabels, config['numClasses'])
vallabels2 = to_categorical(vallabels, config['numClasses'])

# Initialize the optimizer and model
# todo: feature normalization (optional)
print("[INFO] compiling model...")
opt = Adam(learning_rate=config['learning_rate'])
model = LeNet.build(numChannels=1,
                    imgRows=2 * w,
                    imgCols=2 * w,
                    numClasses=config['numClasses'],
                    weightsPath=None)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=[F1])
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=config['patience'],
                                          verbose=config['verbose'],
                                          mode='auto',
                                          baseline=None, restore_best_weights=True)
callbacks_list = [earlystop]

print("[INFO] training...",  flush=True)
# todo: add radius to model
# todo: augmentation in batch training

# todo augmentation here is bad, mask issue
datagen = ImageDataGenerator(
    featurewise_std_normalization=False,
    rotation_range=90,
    shear_range=0.05,
    zoom_range=0.01,
    width_shift_range=0.03, height_shift_range=0.03,
    horizontal_flip=True, vertical_flip=True
)

datagen.fit(trainimages)

# model.fit(trainimages, trainlabels, validation_data=(valimagesMsk, vallabels),
#           batch_size=config['batch_size'], epochs=config['epochs'],
#           verbose=config['verbose'])

model.fit(datagen.flow(trainimages, trainlabels2, batch_size=int(config['batch_size'])),
                    validation_data=(valimages, vallabels2),
                    steps_per_epoch= int(len(trainimages) / config['batch_size']),
                    epochs=config['epochs'],
                    callbacks=callbacks_list,
                    verbose=config['verbose'])

# Evaluation of the model
print("[INFO] evaluating...")
(loss, f1) = model.evaluate(trainimages, trainlabels2,
                            batch_size=config['batch_size'], verbose=config['verbose'])
print("[INFO] training F1: {:.2f}%".format(f1 * 100))

print("[INFO] evaluating...")
(loss, f1) = model.evaluate(valimages, vallabels2,
                            batch_size=config['batch_size'], verbose=config['verbose'])
print("[INFO] validation F1: {:.2f}%".format(f1 * 100))

print("[INFO] dumping weights to file...")
model.save_weights(args.output, overwrite=True)

#### MANUAL boosting ###
if config['BOOSTING']:
    probs = model.predict(trainimages)
    classifications = probs.argmax(axis=1)

    tricky_idx = trainlabels != classifications
    tricky_images = trainimages[tricky_idx]  # (501, 40, 40, 1)
    tricky_labels = trainlabels[tricky_idx]  #
    print("trainimages.shape:", trainimages.shape)
    print("tricky_images.shape:", tricky_images.shape)

    for i in range(2):
        tricky_images = np.concatenate((tricky_images, tricky_images))
        tricky_labels = np.concatenate((tricky_labels, tricky_labels))

    boost_size = min(len(tricky_labels), config['aug_sample_size'] // 4)
    tricky_images = tricky_images[0:boost_size]
    tricky_labels = tricky_labels[0:boost_size]
    print(">>>duplicated tricky_images.shape:", tricky_images.shape)
    print(">>>duplicated tricky_labels.shape:", tricky_labels.shape)

    tricky_labels2 = to_categorical(tricky_labels, config['numClasses'])
    print("tricky_labels2.shape:", tricky_labels2.shape)

    retrain_images = np.concatenate((tricky_images, trainimages))
    retrain_labels = np.concatenate((tricky_labels, trainlabels))
    retrain_labels2 = to_categorical(retrain_labels, config['numClasses'])

    print("retrain_images.shape:", retrain_images.shape)
    print("retrain_labels.shape:", retrain_labels.shape)
    print("retrain_labels2.shape:", retrain_labels2.shape)

    datagen.fit(retrain_images)
    model.fit_generator(datagen.flow(retrain_images, retrain_labels2,
                                     batch_size=config['batch_size']),
                        validation_data=(valimages, vallabels2),
                        steps_per_epoch=len(retrain_images) / config['batch_size'],
                        epochs=config['epochs'],
                        callbacks=callbacks_list,
                        verbose=config['verbose'])

    print("[INFO] evaluating...")
    (loss, f1) = model.evaluate(trainimages, trainlabels2,
                                batch_size=config['batch_size'], verbose=config['verbose'])
    print("[INFO] training F1: {:.2f}%".format(f1 * 100))

    print("[INFO] evaluating...")
    (loss, f1) = model.evaluate(valimages, vallabels2,
                                batch_size=config['batch_size'], verbose=config['verbose'])
    print("[INFO] validation F1: {:.2f}%".format(f1 * 100))

    print("[INFO] dumping weights to file...")
    model.save_weights(args.output, overwrite=True)
