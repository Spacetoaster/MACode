import tensorflow as tf
import tflearn
from PIL import Image
import numpy as np


class QuaternionReader:
    """ Class for reading tf-record files of the liver-images """

    def  __init__(self, filename, image_width=100, image_height=100, image_depth=3, number_of_images=3, batch_size=10, withName=False):
        self.filename = filename
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.number_of_images = number_of_images
        self.batch_size = batch_size
        self.withName = withName

    def getInputList(self):
        """ returns a list of the formats of all the inputs """
        inputList = []
        for input in range(self.number_of_images):
            # RGB-Format for each Input-Image
            inputList.append((self.image_width, self.image_height, self.image_depth))
        return inputList

    # reads raw image and converts to float32
    def decode_image(self, feature):
        image = tf.image.decode_png(feature, channels=self.image_depth)
        image = tf.reshape(image, [self.image_width, self.image_height, self.image_depth])
        # convert to float
        image = tf.cast(image, tf.float32)
        # normalization
        image = tf.image.per_image_standardization(image)

        return image

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        record_features = {}
        record_features['label_qw'] = tf.FixedLenFeature([], tf.float32)
        record_features['label_qx'] = tf.FixedLenFeature([], tf.float32)
        record_features['label_qy'] = tf.FixedLenFeature([], tf.float32)
        record_features['label_qz'] = tf.FixedLenFeature([], tf.float32)
        if self.withName:
            record_features['name'] = tf.FixedLenFeature([], tf.string)
        for i in range(self.number_of_images):
            record_features['image_' + str(i) + '_raw'] = tf.FixedLenFeature([], tf.string)

        features = tf.parse_single_example(
            serialized_example,
            record_features
        )

        # decode images
        images = []
        for i in range(self.number_of_images):
            images.append(self.decode_image(features['image_' + str(i) + '_raw']))

        label = tf.stack([features['label_qw'], features['label_qx'], features['label_qy'], features['label_qz']])
        if self.withName:
            name = features['name']

        if not self.withName:
            return images, label
        else:
            return images, label, name

    def inputs(self, num_epochs=None):
        filename_queue = tf.train.string_input_producer([self.filename], num_epochs=num_epochs)

        images, label = self.read_and_decode(filename_queue)

        image_and_label = [x for x in images] + [label]

        image_and_label_batch = tf.train.batch(
            image_and_label, batch_size=self.batch_size, num_threads=2,
            capacity=200
        )

        images_batch = image_and_label_batch[:-1]
        labels_batch = image_and_label_batch[-1]

        return images_batch, labels_batch

    def inputs_with_name(self, num_epochs=None):
        filename_queue = tf.train.string_input_producer([self.filename], num_epochs=num_epochs)

        images, label, name = self.read_and_decode(filename_queue)

        image_and_label = [x for x in images] + [label] + [name]

        image_and_label_batch = tf.train.batch(
            image_and_label, batch_size=self.batch_size, num_threads=2,
            capacity=200
        )

        images_batch = image_and_label_batch[:-2]
        labels_batch = image_and_label_batch[-2]
        names_batch = image_and_label_batch[-1]

        return images_batch, labels_batch, names_batch