import os
import sys
import tensorflow as tf
from PIL import Image
import StringIO
import argparse


class QuaternionRecordConverter:
    """ Class to convert Image/Label(Rotation as Quaternion) pairs to TF-Records from a text-file after rendering """

    def __init__(self, dataDir, num_images, results_file=None, liverName=False):
        self.dataDir = dataDir
        if not results_file:
            self.gtFile = os.path.join(dataDir, "results.txt")
        else:
            self.gtFile = results_file
        self.filename = os.path.join(dataDir, "record.tfrecords")
        self.num_images = num_images
        self.liverName = liverName

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def convert(self):
        gt = open(self.gtFile)
        writer = tf.python_io.TFRecordWriter(self.filename)
        for line in gt:
            arrayLine = line.rstrip().split(" ")
            sys.stdout.write("Writing " + arrayLine[0] + "\r")
            record_features = {}
            record_features['label_qw'] = self._float_feature(float(arrayLine[1]))
            record_features['label_qx'] = self._float_feature(float(arrayLine[2]))
            record_features['label_qy'] = self._float_feature(float(arrayLine[3]))
            record_features['label_qz'] = self._float_feature(float(arrayLine[4]))
            if self.liverName:
                record_features['name'] = self._bytes_feature(arrayLine[0])
            imagesPath = os.path.join(self.dataDir, arrayLine[0])
            for i in range(self.num_images):
                output = StringIO.StringIO()
                image = Image.open(os.path.join(imagesPath, "liver_" + str(i) + ".png"))
                image.save(output, format="PNG")
                record_features['image_' + str(i) + '_raw'] = self._bytes_feature(output.getvalue())
            example = tf.train.Example(features=tf.train.Features(feature=record_features))
            writer.write(example.SerializeToString())
            sys.stdout.flush()
        writer.close()
        gt.close()


def parseArguments():
    parser = argparse.ArgumentParser(description='Convert Images to TF-Records')
    parser.add_argument('--num_images', type=int, default=3, help='number of images in each record')
    parser.add_argument('--results_file', type=str, default=None)
    parser.add_argument('--liverName', dest='liverName', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = parseArguments()
    dataDir = './'
    num_images = args.num_images
    converter = QuaternionRecordConverter(dataDir, num_images, results_file=args.results_file, liverName=args.liverName)
    converter.convert()


if __name__ == '__main__': main()
