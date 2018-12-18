# import the necessary packages
from pyimagesearch.cnn.networks.lenet import LeNet
from ccount import *
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
from keras.datasets import mnist
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import backend as K
from collections import Counter
import numpy as np
import argparse

import matplotlib.pyplot as plt  # tk not on hpcc
matplotlib.use('Agg')  # not display on hpcc
import cv2  # not on hpcc

# Communication
print('example training: python lenet_colonies.py -db mid.strict.npy.gz -s 1 -w ./output/mid.strict.hdf5')
print('example loading: python lenet_colonies.py -db mid.strict.npy.gz -l 1 -w ./output/mid.strict.hdf5')
# todo: uncertain as negative
# todo: predict for other datasets (-l 1 -w path -db new -s 0) (for one image in one npy file)
# todo: change format to


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-db", "--blobs-db", type=str,
                help="path to blobs-db file, e.g. xxx.labeled.npy.gz")
ap.add_argument("-s", "--save-model", type=int, default=-1,
                help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
                help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
                help="(optional) path to weights file")

args = vars(ap.parse_args())


# Parameters
verbose = 1  # {0, 1}
scaling_factor = 4  # input scale down  # todo: 2x scaling
training_ratio = 0.7  # proportion of data to be in training set
r_extension_ratio = 1.4  # larger (1.4) for better view under augmentation
epochs = 50  # default 50
learning_rate = 0.0001  # default 0.0001 (Adam)


# Load Labeled blobs_db
blobs = load_blobs_db(args["blobs_db"])
w = int(sqrt(blobs.shape[1]-6) / 2)  # width of img

# Remove unlabeled and uncertain (only when training)
if args["load_model"] < 0:
    blobs = blobs[blobs[:, 3] >= 0, :]

# Split train/valid
[trainBlobs, valBlobs] = split_train_valid(blobs, training_ratio)
print("{} Split ratio, split Data into {} training and {} testing".\
      format(training_ratio, trainBlobs.shape[0], valBlobs.shape[0]))

# Balancing Yes/No ratio
if args["load_model"] < 0:
    trainBlobs = balancing_by_duplicating_yes(trainBlobs)

# Parse blobs
trainImages, trainLabels, trainRs = parse_blobs(trainBlobs)
valImages, valLabels, valRs = parse_blobs(valBlobs)
print(trainImages.shape, trainLabels.shape, trainRs.shape)

# Mixed Augmentation
# todo: augmentation on blob level (keep Data, label, Rs cosistant)
# todo: change R alone with scaling
# todo: augmentation in batch training
if args["load_model"] < 0:
    trainImages = augment_images(trainImages)
    print("trainImagesAug:", trainImages.shape)
    print('max data', np.max(trainImages), 'min', np.min(trainImages))

# Downscale images
print("Downscaling images by ", scaling_factor)
trainImages = np.array([down_scale(image, scaling_factor=scaling_factor) for image in trainImages])
valImages = np.array([down_scale(image, scaling_factor=scaling_factor) for image in valImages])

# Downscale w and R
w = int(w/scaling_factor)
trainRs = trainRs/scaling_factor * r_extension_ratio
valRs = valRs/scaling_factor * r_extension_ratio

# Equalize images
print("Equalizing images...")
trainImages = np.array([equalize(image) for image in trainImages])
valImages = np.array([equalize(image) for image in valImages])

# Mask images
print("Masking images...")
trainImages = np.array([mask_image(image, r=trainRs[ind]) for ind, image in enumerate(trainImages)])
valImagesMsk = np.array([mask_image(image, r=valRs[ind]) for ind, image in enumerate(valImages)])

# Reshape for model
trainImages = trainImages.reshape((trainImages.shape[0], 2*w, 2*w, 1))
valImagesMsk = valImagesMsk.reshape((valImagesMsk.shape[0], 2*w, 2*w, 1))
valImages = valImages.reshape((valImages.shape[0], 2*w, 2*w, 1))
print("max pixel value: ", np.max(trainImages))
print("min pixel value: ", np.min(trainImages))

# Categorize labels for softmax
trainLabels = np_utils.to_categorical(trainLabels, 2)
valLabels = np_utils.to_categorical(valLabels, 2)

# Initialize the optimizer and model
# todo: use F1 score and accuracy
# todo: early stopping
# todo: feature normalization (optional)
print("[INFO] compiling model...")
opt = Adam(lr=learning_rate)
model = LeNet.build(numChannels=1, imgRows=2*w, imgCols=2*w, numClasses=2,
                    weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])  # todo: F1

# Train if not loading pre-trained weights
if args["load_model"] < 0:
    print("[INFO] training...")
    model.fit(trainImages, trainLabels,
              batch_size=64, epochs=epochs,  # test epoch should be 20, verbose should be 1
              verbose=verbose)

# Evaluation of the model
print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(trainImages, trainLabels,
                                  batch_size=20, verbose=verbose)
print("[INFO] training accuracy: {:.2f}%".format(accuracy * 100))

print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(valImagesMsk, valLabels,
                                  batch_size=20, verbose=verbose)
print("[INFO] validation accuracy: {:.2f}%".format(accuracy * 100))


# check to see if the model should be saved to file
if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)

# Prediction at load mode
if args["load_model"] > 0:
    # randomly select a few testing digits
    # todo: fix tk in hpcc
    # _tkinter.TclError: couldn't connect to display ":0.0"
    # load
    print('loading...', args["blobs_db"])
    blobs = load_blobs_db(args["blobs_db"])
    w = int(sqrt(blobs.shape[1] - 6) / 2)  # width of img
    # parse
    Images, Labels, Rs = parse_blobs(blobs)
    # equalize
    Images_ = np.array([equalize(image) for image in Images])
    # scale down
    Images_ = np.array([down_scale(image, scaling_factor=scaling_factor) for image in Images_])
    w_ = int(w / scaling_factor)
    Rs_ = Rs / scaling_factor
    # mask
    Images_ = np.array([mask_image(image, r=Rs_[ind]) for ind, image in enumerate(Images_)])
    # reshape for model
    Images_ = Images_.reshape((Images_.shape[0], 2 * w_, 2 * w_, 1))

    # Predictions
    print('Making predictions...')
    probs = model.predict(Images_)
    predictions = probs.argmax(axis=1)
    print("Predictions:", np.mean(predictions), predictions)
    idx_yes = predictions == 1


    print('Showing rand samples from', args['blobs_db'])
    while True:
        i = np.random.choice(np.arange(0, len(Images_)))
        # classify the digit
        prob = model.predict(Images_[np.newaxis, i])
        prediction = prob.argmax(axis=1)
        print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], np.argmax(Labels[i])))

        image = (Images[i] * 255).astype("uint8")
        image_ = (Images_[i] * 255).astype("uint8")
        image = np.reshape(image, image.shape[0:2])
        image_ = np.reshape(image_, image_.shape[0:2])

        r = Rs[i]
        r_ = Rs_[i] * r_extension_ratio
        prediction = predictions[i]
        label = Labels[i]

        # python-tk not on hpcc
        # todo: show two images: original and masked

        fig, axes = plt.subplots(1, 2, figsize=(8, 16), sharex=False, sharey=False)
        ax = axes.ravel()
        ax[0].set_title('For human labeling\nprediction:{}\nradius:{}'.format(int(prediction), r))
        ax[0].imshow(image, 'gray')
        c = plt.Circle((w - 1, w - 1), r, color='yellow', linewidth=1, fill=False)
        ax[0].add_patch(c)
        ax[1].set_title('For model training\ncurrent label:{}\nradius:{}'.format(int(prediction), r_))
        ax[1].imshow(image_, 'gray')
        plt.tight_layout()
        plt.show()

        # x = input("keep poping images until the input is e\n")
        # if x == 'e':
        #     break

        # out_png = 'valImage.' + args["blobs_db"] + str(i) + \
        # '.label_' + str(Labels[i]) + '.pred_' + str(prediction[0]) + '.png'
        # plt.savefig(out_png, dpi=150)


    # todo: prediction for other db (npy)
