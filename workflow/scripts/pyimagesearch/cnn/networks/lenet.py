# import the necessary packages
from keras.models import Sequential
from keras.layers import Convolution2D # updated for Mac
from keras.layers import MaxPooling2D  # updated for Mac
from keras.layers import Dropout # updated for Mac
from keras.layers import Activation # updated for Mac
from keras.layers import Flatten # updated for Mac
from keras.layers import Dense # updated for Mac
from keras import backend as K 


class LeNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses,
              activation="relu", weightsPath=None):
        # initialize the model
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)

        # define the first set of CONV => ACTIVATION => POOL layers
        model.add(Convolution2D(20, 5, padding="same",  # 20 filters of 5x5 todo: larger conv
                         input_shape=inputShape))
        model.add(Dropout(0.5))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the second set of CONV => ACTIVATION => POOL layers
        # It’s common to see the number of CONV  filters learned increase in deeper layers of the network.
        model.add(Convolution2D(50, 5, padding="same"))
        model.add(Dropout(0.5))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the third set of CONV => ACTIVATION => POOL layers
        # It’s common to see the number of CONV  filters learned increase in deeper layers of the network.
        model.add(Convolution2D(100, 5, padding="same"))
        model.add(Dropout(0.5))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the first FC => ACTIVATION layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Dropout(0.5))
        model.add(Activation(activation))

        # define the second FC => ACTIVATION layers
        model.add(Dense(200))
        model.add(Dropout(0.5))
        model.add(Activation(activation))

        # define the second FC layer
        model.add(Dense(numClasses))

        # lastly, define the soft-max classifier
        model.add(Activation("softmax"))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # return the constructed network architecture
        return model