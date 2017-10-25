import os
import argparse
from train import MSERegressionTrainerRating
from convert import QuaternionRecordConverter


def convertIfNecessary(path, num_images):
    if not os.path.exists(path):
        print "converting: " + path
        imagesDir = os.path.dirname(path)
        converter = QuaternionRecordConverter(imagesDir, num_images)
        converter.convert()
    else:
        print "tfrecord exists: " + path

def parseArguments():
    parser = argparse.ArgumentParser(description='Train network batch')
    parser.add_argument('parent_folder', type=str, help='path to parent folder of the training data')
    parser.add_argument('save_folder', type=str, help='path to parent folder of trained networks')
    return parser.parse_args()


def addMultiNetwork(dirPrefix, input_format):
    global trainerList, parent_folder, save_folder, num_epochs, num_train, num_test

    trainerList.append(
        MSERegressionTrainerRating(train_data=os.path.join(parent_folder, dirPrefix) + "_train/record.tfrecords",
                                   test_data=os.path.join(parent_folder, dirPrefix) + "_test/record.tfrecords",
                                   num_epochs=num_epochs,
                                   num_train=num_train, num_test=num_test, save_path=os.path.join(save_folder, dirPrefix),
                                   input_format=input_format))


global num_train
global num_test
global num_epochs
global trainerList
global parent_folder
global save_folder


def main():
    global trainerList, parent_folder, save_folder, num_epochs, num_train, num_test
    num_train = 14000
    num_test = 2800
    num_epochs = 25

    args = parseArguments()
    trainerList = []
    parent_folder = args.parent_folder
    save_folder = args.save_folder

    addMultiNetwork("liverAll2Cams", [2, 100, 100, 3])
    addMultiNetwork("liverAll3Cams", [3, 100, 100, 3])
    addMultiNetwork("liverAll6Cams", [6, 100, 100, 3])
    addMultiNetwork("liverAll8Cams", [8, 100, 100, 3])
    addMultiNetwork("liverAll10Cams", [10, 100, 100, 3])
    addMultiNetwork("liverAll16Cams", [16, 100, 100, 3])
    addMultiNetwork("liverAll18Cams", [18, 100, 100, 3])

    for nt in trainerList:
        num_images = nt.input_format[0]
        convertIfNecessary(nt.train_data, num_images)
        convertIfNecessary(nt.test_data, num_images)
        print "training " + nt.train_data
        nt.train()


if __name__ == '__main__': main()
