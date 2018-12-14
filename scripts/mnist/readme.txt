code modified from:
https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

training:
$ python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5

prediction:
$ python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5
