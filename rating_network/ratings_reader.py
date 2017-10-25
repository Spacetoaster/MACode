import tensorflow as tf
import os
import tflearn
from PIL import Image
import numpy as np


class RatingsReader:
    """ Class for reading tf-record files of the liver-images """

    def __init__(self, filename, image_width=100, image_height=100, image_depth=3, batch_size=10, norm_images=True,
                 norm_depth=True, output_classification=False, num_classes=100, shuffle_batch=False, augmentation=False,
                 no_depth=False):
        self.filename = filename
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.number_of_images = 1
        self.batch_size = batch_size
        self.norm_images = norm_images
        self.norm_depth = norm_depth
        self.output_classification = output_classification
        self.num_classes = num_classes
        self.shuffle_batch = shuffle_batch
        self.augmentation = augmentation
        self.no_depth = no_depth
        self.withName = False

    def getInputList(self):
        """ returns a list of the formats of all the inputs """
        inputList = []
        # RGB Image
        inputList.append((self.image_width, self.image_height, self.image_depth))
        # Depth Liver and Depth Tumors
        if not self.no_depth:
            inputList.append((self.image_width, self.image_height, 1))
            inputList.append((self.image_width, self.image_height, 1))
        return inputList

    def decode_image(self, feature):
        """ reads raw image and converts to float32 """
        image = tf.image.decode_png(feature, channels=self.image_depth)
        image = tf.reshape(image, [self.image_width, self.image_height, self.image_depth])
        # convert to float
        image = tf.cast(image, tf.float32)
        # normalization
        if self.norm_images:
            image = tf.image.per_image_standardization(image)
        return image

    def decode_depth_image(self, feature):
        depth_image = tf.decode_raw(feature, out_type=tf.float32)
        depth_image = tf.reshape(depth_image, [self.image_width, self.image_height, 1])
        # normalization of depth image
        if self.norm_depth:
            depth_image = tf.image.per_image_standardization(depth_image)
        return depth_image

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        record_features = {}
        record_features['rating'] = tf.FixedLenFeature([], tf.float32)
        record_features['image_rgb_raw'] = tf.FixedLenFeature([], tf.string)
        record_features['liver_depth'] = tf.FixedLenFeature([], tf.string)
        record_features['tumors_depth'] = tf.FixedLenFeature([], tf.string)
        features = tf.parse_single_example(
            serialized_example,
            record_features
        )

        # decode images
        images = []
        images.append(self.decode_image(features['image_rgb_raw']))
        images.append(self.decode_depth_image(features['liver_depth']))
        images.append(self.decode_depth_image(features['tumors_depth']))

        # data augmentation
        if self.augmentation:
            concat_image = tf.concat(axis=2, values=[images[0], images[1], images[2]])
            rand = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)
            transformed_image = tf.image.random_flip_up_down(concat_image)
            transformed_image = tf.image.random_flip_left_right(transformed_image)
            transformed_image = tf.image.rot90(transformed_image, rand)
            images[0] = tf.slice(transformed_image, [0, 0, 0], [self.image_width, self.image_height, self.image_depth])
            images[1] = tf.slice(transformed_image, [0, 0, 3], [self.image_width, self.image_height, 1])
            images[2] = tf.slice(transformed_image, [0, 0, 4], [self.image_width, self.image_height, 1])

        if self.no_depth:
            images = images[:1]
        # seems like reshape is needed if single label!
        label = tf.reshape(features['rating'], [1])

        if self.output_classification:
            # will output one-hot-encoding of values from 0 to maxBound
            maxBound = self.num_classes - 1
            label = tf.one_hot(
                tf.clip_by_value(tf.cast(tf.floor(tf.multiply(label, self.num_classes)), tf.int32), 0, maxBound),
                self.num_classes)
            label = tf.reshape(label, [self.num_classes])

        return images, label

    def inputs(self, num_epochs=None):
        if not os.path.isdir(self.filename):
            filename_queue = tf.train.string_input_producer([self.filename], num_epochs=num_epochs)
        else:
            filenames = [os.path.join(self.filename, x) for x in os.listdir(self.filename)]
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

        images, label = self.read_and_decode(filename_queue)

        image_and_label = [x for x in images] + [label]

        if self.shuffle_batch:
            image_and_label_batch = tf.train.shuffle_batch(image_and_label, batch_size=self.batch_size, capacity=2000,
                                                           min_after_dequeue=1000)
        else:
            image_and_label_batch = tf.train.batch(image_and_label, batch_size=self.batch_size, capacity=200)

        images_batch = image_and_label_batch[:-1]
        labels_batch = image_and_label_batch[-1]

        return images_batch, labels_batch
