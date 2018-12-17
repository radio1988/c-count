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

print('example training: python lenet_colonies.py -db mid.strict.npy.gz -s 1 -w ./output/mid.strict.hdf5')
print('example loading: python lenet_colonies.py -db mid.strict.npy.gz -l 1 -w ./output/mid.strict.hdf5')

# construct the argument parser and parse the arguments
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
scaling_factor = 4  # input scale down
training_ratio = 0.7  # proportion of data to be in training set


# Load Labeled blobs_db
blobs = load_blobs_db(args["blobs_db"])

# Remove unlabeled and uncertain
blobs = blobs[blobs[:, 3] >= 0, :]

# Remove bias by reducing No (in all inputs)
# todo: balancing after augmentation
idx_yes = np.arange(0, blobs.shape[0])[blobs[:, 3] == 1]
idx_no =  np.arange(0, blobs.shape[0])[blobs[:, 3] == 0]
N_Yes = len(idx_yes)
N_No = len(idx_no)

if N_No > N_Yes:
    print('number of No matched to yes by sub-sampling')
    idx_no = np.random.choice(idx_no, N_Yes, replace=False)
    idx_choice = np.concatenate([idx_yes, idx_no])
    np.random.seed(2)
    np.random.shuffle(idx_choice)
    np.random.seed()
    blobs = blobs[idx_choice,]

print('after subsampling Counter')
Counter(blobs[:, 3])


# Input stats
N = blobs.shape[0]
w = int(sqrt(blobs.shape[1]-6) / 2)  # width of img



# Reshape into 2D images
flats = blobs[:, 6:]
Data = flats.reshape(N, 2*w, 2*w)
Labels = blobs[:, 3]
Rs = blobs[:, 2]
Labels = Labels.astype(int)

# todo: augmentation
# todo: augmentation on blob level (keep Data, label, Rs cosistant)
# todo: change R alone with scaling
# todo: augmentation in batch training


# flipping
# print('flipping inputs...')
# flippedData = np.array([np.fliplr(image) for image in Data])
# # check image
# for i in range(0,2):
#     plt.imshow(Data[i], 'gray')
#     plt.title('Data'+str(i) + ' label=' + str(Labels[i]))
#     plt.show()
#     plt.imshow(flippedData[i], 'gray')
#     plt.title('flippedData'+str(i) + ' label=' + str(Labels[i]))
#     plt.show()

# Mixed
TranslatedData = np.array([np.fliplr(image) for image in Data])
import imgaug as ia
from imgaug import augmenters as iaa

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.9, aug)


Data = Data.reshape(N, 2*w, 2*w, 1)

print("Data, with channels", Data.shape)

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.01, 0.001),
            pad_mode=ia.ALL,
            pad_cval=(0, 1)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-5, 5), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 1), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
    ],
    random_order=True
)

augData = seq.augment_images(Data)

augData = augData.reshape(N, 2*w, 2*w)
Data = Data.reshape(N, 2*w, 2*w)

print("data_aug:", augData.shape)
Data = np.concatenate([Data, augData])
Labels = np.concatenate([Labels, Labels])
Rs = np.concatenate([Rs, Rs])

print(Data.shape)
def normalize_img(image):
    image = image - np.min(image)
    image = image / np.max(image)
    return image
print('max data', np.max(Data), 'min', np.min(Data))
Data = np.array([normalize_img(image) for image in Data])
print('max data', np.max(Data), 'min', np.min(Data))
sleep (8)

# check image
for i in range(0, 30):
    plt.imshow(Data[i], 'gray')
    plt.title('Data'+str(i) + ' label=' + str(Labels[i]))
    plt.show()
    plt.imshow(augData[i], 'gray')
    plt.title('augData'+str(i) + ' label=' + str(Labels[i]))
    plt.show()

sleep(5)


# Downscale images
print("Downscaling images by ", scaling_factor)
Data = np.array([down_scale(image, scaling_factor=scaling_factor) for image in Data])
w = int(w/scaling_factor)
Rs = Rs/scaling_factor * 1.2
# Equalize images
print("Equalizing images...")
Data = np.array([equalize(image) for image in Data])
# Mask images
print("Masking images...")
Data = np.array([mask_image(image, r=Rs[ind]) for ind, image in enumerate(Data)])

# Split
print("Data split into {} training and testing".format(training_ratio))
N_train = int(N*training_ratio)
trainData = Data[0:N_train]
testData = Data[N_train:]
trainLabels = Labels[0:N_train]
testLabels = Labels[N_train:]
print('training data: ', trainData.shape)
print('testing data:', testData.shape)
print(type(trainData), trainData.shape)
print(type(trainLabels), trainLabels.shape)


# # check image
# for i in range(0,10):
#     plt.imshow(trainData[i], 'gray')
#     plt.title('trainData'+str(i) + ' label=' + str(Labels[i]))
#     out_png = 'trainData' + args["blobs_db"] + str(i) + '.label_' + str(Labels[i]) + '.png'
#     plt.savefig(out_png, dpi=150)

print(trainData.shape, trainLabels.shape)

print("Training Labels:\n", Counter(trainLabels))
print("Training Labels:\n", Counter(trainLabels))


trainData = trainData.reshape((trainData.shape[0], 2*w, 2*w, 1))
testData = testData.reshape((testData.shape[0], 2*w, 2*w, 1))
print("max pixel value: ", np.max(trainData))
print("min pixel value: ", np.min(trainData))



# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 2)
testLabels = np_utils.to_categorical(testLabels, 2)


# initialize the optimizer and model
# todo: use F1 score and accuracy
# todo: early stopping
# todo: feature normalization (optional)
print("[INFO] compiling model...")
opt = Adam(lr=0.0001)  # todo: ADAM
model = LeNet.build(numChannels=1, imgRows=2*w, imgCols=2*w,
                    numClasses=2,
                    weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])  # todo: F1

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
    print("[INFO] training...")
    model.fit(trainData, trainLabels, batch_size=64, epochs=50,  # test epoch should be 20, verbose should be 1
              verbose=verbose)

# show the accuracy on the testing set
print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(trainData, trainLabels,
                                  batch_size=20, verbose=verbose)
print("[INFO] training accuracy: {:.2f}%".format(accuracy * 100))


# show the accuracy on the testing set
print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(testData, testLabels,
                                  batch_size=20, verbose=verbose)
print("[INFO] validation accuracy: {:.2f}%".format(accuracy * 100))


# check to see if the model should be saved to file
if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)


# randomly select a few testing digits
# todo: not working on HPCC, but works on mac
# _tkinter.TclError: couldn't connect to display ":0.0"
np.random.seed(1)
for i in np.random.choice(np.arange(0, len(testLabels)), size=(30,)):
    # classify the digit
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], np.argmax(testLabels[i])))

    # extract the image from the testData if using "channels_first"
    # ordering
    if K.image_data_format() == "channels_first":
        image = (testData[i][0] * 255).astype("uint8")
    else:
        # otherwise we are using "channels_last" ordering
        image = (testData[i] * 255).astype("uint8")

    # python-tk not on hpcc
    print(image.shape)
    image = np.reshape(image, image.shape[0:2])
    plt.imshow(image, 'gray')
    plt.title("Label:" + str(np.argmax(testLabels[i])) + '; Prediction:' + str(prediction[0]))
    out_png = 'testData.' + args["blobs_db"] + str(i) + \
    '.label_' + str(Labels[i]) + '.pred_' + str(prediction[0]) + '.png'
    plt.savefig(out_png, dpi=150)

    # # open-csv not on hpcc
    # # merge the channels into one image
    # image = cv2.merge([image] * 3)
    #
    # # resize the image from a 28 x 28 image to a 96 x 96 image so we
    # # can better see it
    # image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_LINEAR)
    #
    # # show the image and prediction
    # cv2.putText(image, str(prediction[0]), (5, 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # cv2.imshow("Digit", image)
    # cv2.waitKey(0)
np.random.seed()


# todo: prediction for other db (npy)
