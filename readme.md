## About:
This repository contains most of the code i wrote during my master thesis.
The objective was to evaluate how optimal views on medical 3d-models can be learned using convolutional neural networks (CNNs). In order to use 3d-models as an input for the models an approach of a multi-view neural network has been used inspired by the approach of Su et al (http://vis-www.cs.umass.edu/mvcnn/).

## Dependencies:
* python (2.7)
* pip (9.01)
	* numpy (1.12.1)
	* numpy-quaternion (0.0.0.dev2017.02.28)
	* tensorflow (1.0.1)
	* tflearn (0.3)
	* OpenEXR (1.2.0)
	* matplotlib (1.5.1)
	* pyquaternion (0.9.0)
	* trianglesolver (1.1)
* blender (2.78)
* OpenEXR
* OpenCV 3

## Filetree:
* `rotation_network/` contains all files which are used for training/prediction/converting for the MVCNN-based approach
* `rotation_network/networks/` contains the implementations of the MVCNN-networks 
* `rating_network/` contains all files for training/prediciton/converting of the rating-network
* `rating_network/networks/` contains the implementations of the rating-networks
* `tfhelper/` contains helper functions for training and evaluation of rotation-quaternions
* `utility/modelgeneration` contains scripts for generation of the 3d-models in blender
* `utility/datageneration` contains scripts for generation/rendering of data used for training
* `utility/imagerating` contains the implementations of the rating function

## Generation of Blender-Models
This can be used to create combinations of models of the liver and the tumors based on the liver-dataset provided by IRCAD (http://ircad.fr/research/3d-ircadb-01/).
The script uses a folder as input, which contains the liver-models as a blender-file and a file containing all the tumor-models.

Example usage:
```
blender --background --python utility/modelgeneration/blender_gen_livers.py -- [liver-folder] [output-folder] [tumor-file] [livercount]
```

## Rendering / Datageneration
Used to generate a dataset by rendering the blender models.

Example usage:
Creates a dataset by rendering models with 3 virtual cameras, while choosing the rotation-axes of the blender-model by using the vertices of an icosphere mesh. There will be 10 rotations per vertex and a resolution of 100x100 pixels is used.
```
blender --background --python utility/datageneration/render.py --blender_files [liverfolder] --rendered_dir [datasetfolder] --num_rotations 10 --icosphere 3 --res_x 100 --res_y 100 --num_cams 3
```

Subsequently the dataset can be converted to a TFRecord-File by using the convert.py script:

Example usage:
```
python convert.py --data_dir [dataset-path] --shuffle True
```

## Rotation-Network (MVCNN)
### Training
Used to train a network after generation of a dataset and converting it to TFRecord-format.

Example usage:
```
python train_rotation.py [Train-Record] [Test-Record] --num_train
[traincount] --num_test [testcount] --num_epochs 25 
--batch_size 50 --learning_rate 0.001 --nntype [networktype] --save_path trainedModel
```
### Prediction
Prediction using a pretrained model.

Example usage:
```
cd trainedModel
python predict_rotation.py [Test-Record] --num_samples [predictcount] 
--nntype [networktype]
```

## Rating-Network
### Training
Example usage:
```
python train_rating.py [Train-Record] [Test-Record] --num_train [traincount] --num_test [testcount] --num_epochs 25 --batch_size 50 --learning_rate 0.001 --nntype [networktype] --save_path trainedModel
```

### Prediction
```
cd trainedModel
python predict_rating.py [Test-Record] --num_samples [predictcount] --nntype [networktype]
```
