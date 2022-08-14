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

        ###
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        ###

        #
        self.discriminator = self.build_discriminator(self.image_shape)

        self.generator = self.build_generator(self.noise_dim)

        #self.gan = self.build_gan()
        
        #per salvare il modello
        self.checkpoint = tf.train.Checkpoint(generator = self.generator, discriminator = self.discriminator)

    def build_gan(self):
        z = Input(self.noise_dim)
        generated_image = self.generator(z)

        decision = self.discriminator(generated_image)

        model = Model(z, decision)

        return model

    def build_generator(self, noise_dim):
        #
        # Input shape (None, 100)
        # Output shape (None, 64, 64, 3)
        #
        model = Sequential(name = "Generator")

        model.add(Dense(4*4*128, input_dim = noise_dim, activation = "relu", use_bias = False))
        model.add(BatchNormalization())
        model.add(Reshape((4,4,128)))

        model.add(Conv2DTranspose(64, (5, 5), strides = (2, 2), padding = "same", activation = "relu", use_bias = False))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(32, (5, 5), strides = (2, 2), padding = "same", activation = "relu", use_bias = False))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(16, (5, 5), strides = (2, 2), padding = "same", activation = "relu", use_bias = False))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(8, (5, 5), strides = (2, 2), padding = "same", activation = "relu", use_bias = False))
        model.add(BatchNormalization())

        model.add(Conv2D(3, (5, 5), activation = "tanh", padding = "same", use_bias = False))

        assert model.output_shape == (None, 64, 64, 3)

        return model

    def build_discriminator(self, image_shape):

        model = Sequential(name = "Discriminator")

        model.add(Conv2D(64, kernel_size = (5, 5), strides = (2, 2), input_shape = image_shape, padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, kernel_size = (5, 5), strides = (2, 2), input_shape = image_shape, padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1))

        return model

    def predict(self, z):
        return self.discriminator(self.generator(z))

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def train_step(self, images):
        noise = tf.random.normal([len(images), self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return disc_loss, gen_loss


    def train(self, dataset, epochs):
        disc_loss_history = []
        gen_loss_history = []
        for epoch in range(epochs):
            print("Epoch", epoch + 1, end = "  -  ")

            disc_loss = []
            gen_loss = []
            for image_batch, _ in dataset:
                disc_loss_, gen_loss_ = self.train_step(image_batch)
                disc_loss.append(disc_loss_)
                gen_loss.append(gen_loss_)
        
            print("Generator Loss: {0:.3f}\tDiscriminator Loss: {1:.3f}".format(np.mean(gen_loss), np.mean(disc_loss)))
            disc_loss_history.append(np.mean(disc_loss))
            gen_loss_history.append(np.mean(gen_loss))
        
        #Salvo il modello
        self.save()
        return disc_loss_history, gen_loss_history
        

    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_path))

    def save(self):
        self.checkpoint.save(file_prefix = self.checkpoint_path + r"/ckpt")

