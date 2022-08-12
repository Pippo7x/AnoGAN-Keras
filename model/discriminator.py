import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, Input, MaxPool2D, UpSampling2D, BatchNormalization, Activation, Dropout, ZeroPadding2D

class Discriminator(object):
    def __init__(self, image_shape):
        self.model = self.build_discriminator(image_shape)

    def build_discriminator(self, image_shape):

        model = Sequential(name = "Discriminator")

        model.add(Conv2D(32, kernel_size = (5, 5), strides = (2, 2), input_shape = image_shape, padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(64, kernel_size = (5, 5), strides = (2, 2), input_shape = image_shape, padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(128, kernel_size = (5, 5), strides = (2, 2), input_shape = image_shape, padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        model.add(Flatten())

        model.add(Dense(1, activation = "sigmoid"))

        return model

    def decision(self, images):
        result = self.model(images, training = False)
        return result