import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, Input, MaxPool2D, UpSampling2D, BatchNormalization, Activation

class Generator(object):
    def __init__(self, noise_dim, image_shape):

        self.model = self.build_generator(noise_dim)

    def build_generator(self, noise_dim):
        #
        # Input shape (None, 100)
        # Output shape (None, 64, 64, 3)
        #

        model = Sequential(name = "Generator")

        model.add(Dense(4*4*128, input_dim = noise_dim))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Reshape((4,4,128)))

        model.add(Conv2DTranspose(64, (5, 5), strides = (2, 2), padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        model.add(Conv2DTranspose(32, (5, 5), strides = (2, 2), padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        model.add(Conv2DTranspose(16, (5, 5), strides = (2, 2), padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        model.add(Conv2DTranspose(8, (5, 5), strides = (2, 2), padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(3, (5, 5), activation = "tanh", padding = "same"))

        assert model.output_shape == (None, 64, 64, 3)

        return model

    def generate(self, noise):
        result = self.model(noise, training = False)
        return result

    def setTrainable(self, trainable):
        self.model.trainable = trainable
