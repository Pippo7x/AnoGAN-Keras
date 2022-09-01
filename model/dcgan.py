import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, Input, BatchNormalization, Dropout
from IPython import display
from tqdm import tqdm

class DCGAN(object):
    def __init__(self, image_shape = (64, 64, 3), noise_dim = 100):
        self.image_shape = image_shape
        self.noise_dim = noise_dim
        self.checkpoint_path = r"./ckpt"

        self.seed = tf.random.normal([1, self.noise_dim])

        ###Training settings
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        ###

        #Creo i componenti della rete
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        
        #per salvare il modello
        self.checkpoint = tf.train.Checkpoint(generator = self.generator, discriminator = self.discriminator)

    def build_generator(self):
        model = Sequential(name = "Generator")

        model.add(Dense(8*8*256, input_dim = self.noise_dim, activation = "relu", use_bias = False))
        model.add(BatchNormalization())
        model.add(Reshape((8,8,256)))

        model.add(Conv2DTranspose(256, (5, 5), strides = (2, 2), padding = "same", activation = "relu", use_bias = False))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(128, (5, 5), strides = (2, 2), padding = "same", activation = "relu", use_bias = False))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(64, (5, 5), strides = (2, 2), padding = "same", activation = "relu", use_bias = False))
        model.add(BatchNormalization())

        model.add(Conv2DTranspose(self.image_shape[2], (5, 5), activation = "tanh", padding = "same", use_bias = False))

        assert model.output_shape[1 :] == self.image_shape

        return model

    def build_discriminator(self):

        model = Sequential(name = "Discriminator")

        model.add(Conv2D(64, kernel_size = (5, 5), strides = (2, 2), input_shape = self.image_shape, padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, kernel_size = (5, 5), strides = (2, 2), padding = "same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Conv2D(256, kernel_size = (5, 5), strides = (2, 2), padding = "same"))
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

    @tf.function
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
            disc_loss = []
            gen_loss = []

            start = time.time()
            for image_batch, _ in tqdm(dataset):
                disc_loss_, gen_loss_ = self.train_step(image_batch)
                disc_loss.append(disc_loss_)
                gen_loss.append(gen_loss_)
            display.clear_output(wait = True)
            print("Epoch", epoch + 1, "  -  ", "Generator Loss: {0:.3f}\tDiscriminator Loss: {1:.3f}\tTime: {2:.1f}s".format(np.mean(gen_loss), np.mean(disc_loss), time.time() - start))
            self.generate_sample()
            disc_loss_history.append(np.mean(disc_loss))
            gen_loss_history.append(np.mean(gen_loss))
        
        #Salvo il modello
        self.save()
        return disc_loss_history, gen_loss_history


    def generate_sample(self):
        generated_image = self.generator(self.seed, training = False)
        generated_image = np.array((generated_image[0] * 127.5) + 127.5, np.int32)

        if self.image_shape[2] == 1:
            plt.imshow(generated_image, cmap = "gray")
        else:
            plt.imshow(generated_image)
        plt.show()

    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_path))

    def save(self):
        self.checkpoint.save(file_prefix = self.checkpoint_path + r"/ckpt")

