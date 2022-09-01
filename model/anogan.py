import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from .dcgan import DCGAN
from IPython import display

class AnoGAN(object):
    def __init__(self, dcgan: DCGAN):
        self.dcgan = dcgan
        ###
        self.feature_layers = self.extract_feature_layers(self.dcgan.discriminator)
        
        self.generator_copy = tf.keras.models.clone_model(self.dcgan.generator)
        self.generator_copy.set_weights(self.dcgan.generator.get_weights())
        self.generator_copy.trainable = False
        ###
        self.model = self.build_anogan()
        
    #Anomaly score: A(x) = (1 - lambda) * R(x) + lambda * D(x)
    def sum_of_residual(self, y_true, y_pred):
        return tf.reduce_sum(tf.abs(y_true - y_pred))

    def extract_feature_layers(self, model):
        feature_layers = Sequential(name = "Feature_Layers")
        feature_layers.add(Input(self.dcgan.image_shape, name = "Input"))
        for i in range(len(model.layers) - 2):
            layer = model.get_layer(index = i)
            feature_layers.add(layer)

        feature_layers.trainable = False
        return feature_layers


    def build_anogan(self):
        anogan_input = Input(shape = (self.dcgan.noise_dim,))
        generator_input = Dense(self.dcgan.noise_dim)(anogan_input)

        generator_output = self.generator_copy(generator_input)
        feature_layers_output = self.feature_layers(generator_output)

        model = Model(inputs = anogan_input, outputs = [generator_output, feature_layers_output])
        model.compile(loss = self.sum_of_residual, loss_weights= [0.9, 0.1], optimizer='adam') #lamda = 0.1

        return model

    def anomaly_detection(self, x):
        z = tf.random.normal([1, self.dcgan.noise_dim])

        feature_layers_output = self.feature_layers.predict(x)

        loss = self.model.fit(z, [x, feature_layers_output], epochs = 500, verbose = 0)
        similar_data, _ = self.model.predict(z)

        return loss.history["loss"][-1], similar_data

    def test(self, dataset):
        #0 -> crack, 1 -> cut, 2 -> good, 3 -> hole, 4 -> print
        results = []

        i = 1
        dataset_length = len(dataset)
        for image, label in dataset:
            anomaly_score, similar_data = self.anomaly_detection(np.array([image]))
            display.clear_output(wait = True)
            print("Test", i, "/", dataset_length, "Label:", label.numpy(),"\tAnomaly Score:", anomaly_score)
            
            self.generate_and_save_images(image, similar_data, anomaly_score, i)

            dict = {
                "anomaly_score": anomaly_score,
                "label": 0 if label.numpy() == 2 else 1
                }
            results.append(dict)

            i = i + 1
            ###END FOR

        good_scores, bad_scores = self.good_bad_split(results)

        threshold = (np.amax(good_scores) + np.amin(bad_scores)) / 2

        y_pred = []
        y_true = []

        for item in results:
            y_true.append(item["label"])
            if item["anomaly_score"] < threshold:
                y_pred.append(0)
            else:
                y_pred.append(1)

        acc = tf.keras.metrics.Accuracy()
        acc.update_state(y_true, y_pred)
        accuracy = acc.result().numpy()

        return results, threshold, accuracy

    def good_bad_split(self, examples):
        good = np.array([])
        bad = np.array([])
        for i in range(len(examples)):
            if(examples[i]["label"] == 0):
                good = np.append(good, examples[i]["anomaly_score"])
            else: 
                bad = np.append(bad, examples[i]["anomaly_score"])

        return good, bad

    
    def generate_and_save_images(self, image, similar_image, anomaly_score, index):
        image = np.array(image * 127.5 + 127.5, np.int32)
        similar_image = np.array(similar_image[0] * 127.5 + 127.5, np.int32)

        plt.figure(figsize = (10, 10), facecolor = "white")

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Test Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(similar_image)
        plt.title("Similar Image")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(abs(similar_image - image))
        plt.title("Anomalies\n{0:.1f}".format(anomaly_score))
        plt.axis("off")

        plt.savefig("results/test_image_{0:03d}.png".format(index))
        plt.show()
