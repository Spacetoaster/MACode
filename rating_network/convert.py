import os
import sys
import tensorflow as tf
from PIL import Image
import StringIO
import argparse
import OpenEXR
import Imath
import random
import numpy as np


class RatingRecordConverter:
    """ Class to convert Image/Label(Rating) pairs to TF-Records from a text-file after rendering """

    def __init__(self, dataDir, ratings_file, split_in_classes=None, shuffle=False, name=None):
        self.dataDir = dataDir
        if ratings_file:
            self.gtFile = ratings_file
        else:
            self.gtFile = os.path.join(dataDir, "ratings.txt")
        self.split_in_classes = split_in_classes
        self.shuffle = shuffle
        self.name = name

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def exrToNumpy(self, exrImagePath):
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        exrFile = OpenEXR.InputFile(exrImagePath)
        dw = exrFile.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        redstr = exrFile.channel('R', pt)
        red = np.fromstring(redstr, dtype=np.float32)
        red.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        return red

    def convert(self):
        gt = open(self.gtFile)
        if not self.shuffle:
            data = [line for line in gt]
        else:
            shuffle_data = [(random.random(), line) for line in gt]
            shuffle_data.sort()
            data = [d[1] for d in shuffle_data]
        gt.close()
        if not self.split_in_classes:
            name = "ratings_record.tfrecords"
            if self.name:
                name = self.name
            writer = tf.python_io.TFRecordWriter(os.path.join(self.dataDir, name))
        else:
            tfrecords_dir = os.path.join(self.dataDir, "tfrecords")
            if not os.path.exists(tfrecords_dir):
                os.mkdir(tfrecords_dir)
            writer = []
            for i in range(self.split_in_classes):
                writer_class = tf.python_io.TFRecordWriter(
                    os.path.join(tfrecords_dir, "ratings_record_class" + str(i) + ".tfrecords"))
                writer.append(writer_class)
        for line in data:
            arrayLine = line.rstrip().split(" ")
            sys.stdout.write("Writing " + arrayLine[0] + "\r")
            record_features = {}
            record_features['rating'] = self._float_feature(float(arrayLine[1]))
            # decode RGB Image
            imagePath = os.path.join(self.dataDir, arrayLine[0])
            output = StringIO.StringIO()
            image = Image.open(imagePath)
            image.save(output, format="PNG")
            record_features['image_rgb_raw'] = self._bytes_feature(output.getvalue())
            # decode depth images
            dirname = os.path.dirname(imagePath)
            depth_liver = self.exrToNumpy(os.path.join(dirname, "liver_0_dl.exr"))
            depth_tumor = self.exrToNumpy(os.path.join(dirname, "liver_0_dt.exr"))
            record_features['liver_depth'] = self._bytes_feature(depth_liver.tobytes())
            record_features['tumors_depth'] = self._bytes_feature(depth_tumor.tobytes())
            # example with features
            example = tf.train.Example(features=tf.train.Features(feature=record_features))
            if not self.split_in_classes:
                writer.write(example.SerializeToString())
            else:
                bin = int(np.clip(np.floor(float(arrayLine[1]) * self.split_in_classes), 0, self.split_in_classes - 1))
                writer[bin].write(example.SerializeToString())
            sys.stdout.flush()
        if not self.split_in_classes:
            writer.close()
        else:
            for w in writer:
                w.close()


def parseArguments():
    parser = argparse.ArgumentParser(description='Convert Images to TF-Records')
    parser.add_argument('--data_dir', type=str, default='./', help='directory where the rendered images and labels are')
    parser.add_argument('--ratings_file', type=str, default=None, help='pass a ratings file other than ./ratings.txt')
    parser.add_argument('--split_in_classes', type=int, default=None,
                        help='will split the dataset into the passed amount')
    parser.add_argument('--shuffle', action='store_true', default=False, help='will shuffle the dataset randomly')
    return parser.parse_args()


def main():
    args = parseArguments()
    converter = RatingRecordConverter(args.data_dir, args.ratings_file, split_in_classes=args.split_in_classes,
                                      shuffle=args.shuffle)
    converter.convert()


if __name__ == '__main__': main()
