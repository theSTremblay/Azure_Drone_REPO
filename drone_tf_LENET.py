################################################################################
# Michael Guerzhoy and Davi Frossard, 2016
# AlexNet implementation in TensorFlow, with weights
# Details:
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
# With code from https://github.com/ethereon/caffe-tensorflow
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *

# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


# set the matplotlib backend so figures can be saved in the background
import matplotlib



matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
#from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import os
# from pylab import *
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import cv2
from imutils import paths
import os
from glob import glob
from keras.preprocessing.image import img_to_array
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
import os

import tensorflow as tf

from caffe_classes import class_names


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

            # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
                             input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model



train_x = zeros((1, 227, 227, 3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

size_of_grid = 9
from PIL import Image


def is_square(apositiveint):
  x = apositiveint // 2
  seen = set([x])
  while x * x != apositiveint:
    x = (x + (apositiveint // x)) // 2
    if x in seen: return False
    seen.add(x)
  return True

def crop2(infile,imgheight,imgwidth,imgheight2, imgwidth2):
    #im = Image.open(infile)
    #imgwidth, imgheight = im.size
    b= []
    for i in range(imgheight//imgheight2):
        for j in range(imgwidth//imgwidth2):
            crop_img = infile[i*imgheight2:(i+1)*imgheight2, j*imgwidth2:(j+1)*imgwidth2]
            b.append(crop_img)
    return b

def crop(im, k):
    k2 = 9

    boolean_var = is_square(k)
    #im = Image.open(input)
    imgheight, imgwidth = im.shape

    imgheight = int(imgheight - (imgheight % sqrt(k)))
    imgwidth = int(imgwidth - (imgwidth % sqrt(k)))

    imgheight2 = int(int(imgheight - (imgheight % sqrt(k))) / 3)
    imgwidth2 = int(int(imgwidth - (imgwidth % sqrt(k))) / 3)



    rim = cv2.resize(im, (imgwidth, imgheight))

    M = rim.shape[0] // 2
    N = rim.shape[1] // 2

    #rimg = Image.fromarray(rim, 'L')

    tiles = crop2(rim, imgheight, imgwidth,imgheight2, imgwidth2)
    return tiles

#213 X 96

    #imgwidth, imgheight = im.size



    # if boolean_var == True:
    #     #M = int(imgwidth / sqrt(k))
    #     #N = int(imgheight / sqrt(k))
    #     tiles = [rim[x:x + M, y:y + N] for x in range(0, rim.shape[0], M) for y in range(0, rim.shape[1], N)]
    #     return tiles
    # else:
    #     print("Cannot Split, please return a perfect square in crop function")
    #
    # return []

def grid_correction(tiles ):
    pass


#Read everything that is in the filepath positive and negative
# for loop splitting them into grid, then post them in the ALex Net model
#
pos_path = r'C:\Users\Sean\Drive_by_Ryan_Gosling\The_Og\drone_project\positive'
neg_path = r'C:\Users\Sean\Drive_by_Ryan_Gosling\The_Og\drone_project\negative'


# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(pos_path)))
random.seed(42)
random.shuffle(imagePaths)

pos_data=[]
pos_labels = []
neg_data=[]
neg_labels = []
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "positive" else 0
    pos_labels.append(label)

i = 0
pos_labels2 = []
for imagePath in imagePaths:

    print(imagePath)
    image = cv2.imread(imagePath, 0)
    pos_data.extend(crop(image, size_of_grid))
    b = [pos_labels[i]] * size_of_grid
    pos_labels2.extend(b)
    i = i + 1

imagePaths = sorted(list(paths.list_images(neg_path)))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "positive" else 0
    neg_labels.append(label)

# Now that the images are split and gathered into arrays
# 1. we must get a for loop where we take the positive images and then split them
# 2. Start cropping the images, and then assign them to a label extrapolated in length to the k value of grid

i = 0
neg_labels2 = []
for imagePath in imagePaths:

    print(imagePath)
    image = cv2.imread(imagePath, 0)
    neg_data.extend(crop(image, size_of_grid))
    b = [neg_labels[i]] * size_of_grid
    neg_labels2.extend(b)
    i = i + 1

pos_data2 = pos_data
pos_labels3 = pos_labels2

np.save('data_np.npy', pos_data2)
#pos_data2 = np.load('data_np.npy')

#np.save('data_np.npy', pos_labels3)
pos_labels31 = np.load('labels_np.npy')

pos_data.extend(neg_data)

pos_labels2.extend(neg_labels2)
np.save('labels_np.npy', pos_labels2)


data = pos_data
labels = pos_labels2

pos_labels3 = pos_labels2

data = np.array(data, dtype="float") / 255.0

np.save('data_floated_np.npy', data)

#pos_labels3 = np.load('labels_np.npy')

#pos_data2 = np.load('data_np.npy')
#data = np.load('data_floated_np.npy')

labels = pos_labels3

labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model

EPOCHS = 25
INIT_LR = 1e-3
BS = 32

print("[INFO] compiling model...")
model = LeNet.build(width=213, height=96, depth=1, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS, verbose=1)

# save the model to disk

path_model = r'C:\Users\Sean\Drive_by_Ryan_Gosling\The_Og\leaf_py\model\model50.hdf5'
path_plot = r'C:\Users\Sean\Drive_by_Ryan_Gosling\The_Og\leaf_py\plots\plot50'
print("[INFO] serializing network...")
model.save(path_model)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Leaves/NotLeaves")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(path_plot)


