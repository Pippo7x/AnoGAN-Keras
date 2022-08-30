import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from .dcgan import DCGAN

class AnoGAN(object):
    def __init__(self, dcgan: DCGAN):
        self.dcgan = dcgan
        self.model = self.anomaly_detector_model(dcgan.generator, dcgan.discriminator)
        
    #Anomaly score: A(x) = (1 - lambda) * R(x) + lambda * D(x)
    def sum_of_residual(self, y_true, y_pred):
        return tf.reduce_sum(tf.abs(y_true - y_pred))

    def extract_feature_layers(self, model):
        feature_layers = Sequential(name = "Feature_Layers")
        feature_layers.add(Input(self.dcgan.image_shape, name = "Input"))
        for i in range(len(model.layers) - 2):
            layer = model.get_layer(index = i)
            feature_layers.add(layer)

        return feature_layers


    def anomaly_detector_model(self, generator, discriminator):
        feature_layers = self.extract_feature_layers(discriminator)
        feature_layers.trainable = False

        generator_copy = tf.keras.models.clone_model(generator)
        generator_copy.set_weights(generator.get_weights())
        generator_copy.trainable = False

        anogan_input = Input(shape = (self.dcgan.noise_dim,))
        generator_input = Dense(self.dcgan.noise_dim)(anogan_input)

        generator_output = generator_copy(generator_input)
        feature_layers_output = feature_layers(generator_output)

        model = Model(inputs = anogan_input, outputs = [generator_output, feature_layers_output])
        model.compile(loss = self.sum_of_residual, loss_weights= [0.9, 0.1], optimizer='adam')

        return model

    def anomaly_detection(self, x):
        z = tf.random.normal([1, self.dcgan.noise_dim])

        feature_layers = self.extract_feature_layers(self.dcgan.discriminator)
        feature_layers_output = feature_layers.predict(x)

        loss = self.model.fit(z, [x, feature_layers_output], epochs = 1000, verbose = 0)
        similar_data, _ = self.model.predict(z)

        return loss.history["loss"][-1], similar_data