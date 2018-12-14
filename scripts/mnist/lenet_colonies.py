# import the necessary packages
from pyimagesearch.cnn.networks.lenet import LeNet
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from collections import Counter
import numpy as np
import argparse
from ccount import *

# import matplotlib.pyplot as plt # tk not on hpcc
# import cv2 # not on hpcc

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


# Load Labeled blobs_db
blobs = load_blobs_db(args["blobs_db"])
blobs = blobs[blobs[:, 3] >= 0, :]  # remove unlabeled and uncertain

N = blobs.shape[0]
w = int(sqrt(blobs.shape[1]-6) / 2)  # width of img

# todo: equalization
# todo: masking

flats = blobs[:, 6:]
Data = flats.reshape(N, 2*w, 2*w)
Labels = blobs[:, 3]
Labels = Labels.astype(int)

# Split
N_train = int(N*0.7)
trainData = Data[0:N_train]
testData = Data[N_train:]
trainLabels = Labels[0:N_train]
testLabels = Labels[N_train:]
print('training data: ', trainData.shape)
print('testing data:', testData.shape)
print(type(trainData), trainData.shape)
print(type(trainLabels), trainLabels.shape)


# check image
import matplotlib.pyplot as plt # tk not on hpcc
i = 0
plt.imshow(trainData[i], 'gray')
plt.title('trainData:' + str(Labels[i]))
plt.show()


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
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(numChannels=1, imgRows=2*w, imgCols=2*w,
                    numClasses=2,
                    weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
    print("[INFO] training...")
    model.fit(trainData, trainLabels, batch_size=100, epochs=1,  # test epoch should be 20, verbose should be 1
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
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
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



    # # open-csv not on hpcc
	# # merge the channels into one image
	# image = cv2.merge([image] * 3)
	#
	# # resize the image from a 28 x 28 image to a 96 x 96 image so we
	# # can better see it
	# image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	#
	# # show the image and prediction
	# cv2.putText(image, str(prediction[0]), (5, 20),
	# 			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # cv2.imshow("Digit", image)
    # cv2.waitKey(0)


    # # python-tk not on hpcc
	# plt.imshow(image, 'gray')
	# plt.title("Label:" + str(np.argmax(testLabels[i])) + '; Prediction:' + str(prediction[0]))
	# plt.show()

