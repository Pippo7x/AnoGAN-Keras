import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, Input, MaxPool2D, UpSampling2D, BatchNormalization, Activation, Dropout, ZeroPadding2D

class DCGAN(object):
    def __init__(self):
        self.image_shape = (64, 64, 3)
        self.noise_dim = 100
        self.checkpoint_path = r"./ckpt"

        #
        self.discriminator = self.build_discriminator(self.image_shape)
        self.discriminator.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

        self.generator = self.build_generator(self.noise_dim)

        self.gan = self.build_gan()
        self.gan.compile(loss = "binary_crossentropy", optimizer = "adam")
        
        #per salvare il modello
        self.checkpoint = tf.train.Checkpoint(gan = self.gan)

    def build_gan(self):
        z = Input(self.noise_dim)
        generated_image = self.generator(z)

        #self.discriminator.trainable = False

        decision = self.discriminator(generated_image)

        model = Model(z, decision)

        return model

    def build_generator(self, noise_dim):
        #
        # Input shape (None, 100)
        # Output shape (None, 64, 64, 3)
        #

        

        model = Sequential(name = "Generator")

        model.add(Dense(4*4*128, input_dim = noise_dim, activation = "relu"))
        model.add(BatchNormalization())
        model.add(Reshape((4,4,128)))

        model.add(Conv2DTranspose(64, (5, 5), strides = (2, 2), padding = "same", activation = "relu"))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(32, (5, 5), strides = (2, 2), padding = "same", activation = "relu"))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(16, (5, 5), strides = (2, 2), padding = "same", activation = "relu"))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(8, (5, 5), strides = (2, 2), padding = "same", activation = "relu"))
        model.add(BatchNormalization())

        model.add(Conv2D(3, (5, 5), activation = "tanh", padding = "same"))

        assert model.output_shape == (None, 64, 64, 3)

        return model

    def build_discriminator(self, image_shape):

        model = Sequential(name = "Discriminator")

        model.add(Conv2D(64, kernel_size = (5, 5), strides = (2, 2), input_shape = image_shape, padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(128, kernel_size = (5, 5), strides = (2, 2), input_shape = image_shape, padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        model.add(Conv2D(128, kernel_size = (5, 5), strides = (2, 2), input_shape = image_shape, padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))

        model.add(Flatten())

        model.add(Dense(1, activation = "sigmoid"))

        return model

    def predict(self, z):
        return self.gan(z)

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            print("Epoch", epoch + 1, end = "  -  ")

            disc_loss = []
            gen_loss = []

            for image_batch, label_batch in dataset:
                noise = tf.random.normal([len(image_batch), self.noise_dim])
                ###TRAIN DISCRIMINATOR

                real_images = image_batch
                fake_images = self.generator(noise, training = False)

                X = tf.concat([real_images, fake_images], 0)
                y = np.array([1] * len(image_batch) + [0] * len(image_batch))

                disc_loss_ = self.discriminator.train_on_batch(X, y)

                ###TRAIN GENERATOR
                self.discriminator.trainable = False
                gen_loss_ = self.gan.train_on_batch(noise, np.array([1] * len(image_batch)))
                self.discriminator.trainable = True 

                disc_loss.append(disc_loss_)
                gen_loss.append(gen_loss_)

            print(f"Generator Loss: {np.mean(gen_loss)}\tDiscriminator Loss: {np.mean(disc_loss)}")

        self.checkpoint.save(file_prefix = self.checkpoint_path + r"\ckpt")
