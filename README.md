# Handwritten Digit Recognition Using CNNs

[![Project Status: Active – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

The goal of this project is to recognize digits in images using Convolutional Neural Networks (CNNs). The network is trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

<img src="https://github.com/sourabbapusridhar/handwritten-digit-recognition-using-cnns/blob/master/output/mnist-dataset.gif?raw=true" width="1280">

## Dataset
### MNIST Database of Handwritten Digits [[Link](http://yann.lecun.com/exdb/mnist/)]
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. The MNIST database contains 60,000 training images and 10,000 testing images. The images are small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

- **Training Set Images:** http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
- **Training Set Labels:** http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
- **Test Set Images:** http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
- **Test Set Labels:** http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

## Requirements
The code is based on Python3 (>=3.8). There are a few dependencies to run the code. The major libraries are listed as follows:
* PyTorch >= 1.8.0
* Tqdm >= 4.60.0
* Tensorboard >= 2.5.0
* Pandas >= 1.2.0

## Installation Guide
To install the anaconda environment, navigate to the repository folder, and run the following command in the terminal:

```
$conda env create -f environment.yml
```

## Execution Guide
1. To activate the Conda environment, please run the following command in the terminal:

```
$conda activate cnns
```

2. To train the neural network based on the `.json` configuration file, please run the following command in the terminal:

```
$python train.py --config config.json
```

**Note:** The `.json` configuration file can be modified for convenient parameter tuning.

4. To resume the training of the neural network from a previously saved checkpoint, please run the following command in the terminal:

```
$python train.py --resume path/to/checkpoint
```

5. To enable multi-GPU training, the `numberOfGpus` argument in the configuration file can be set to a higher number. If the network is configured to use a smaller number of GPUs than the number of GPUs available in the system, the first `n` devices would be used by default. To specify the indices of the available CUDA devices, please run the following command in the terminal:

```
$python train.py --device 1,2 --config config.json
```

6. To test the trained neural network, please run the file `test.py` by passing the path to the trained checkpoint in the terminal:

```
$python test.py --resume path/to/trained/checkpoint 
```
7. To deactivate the Conda environment, please run the following command in the terminal:

```
$conda deactivate
```

## Visualization Guide
To visualize the training progress, the `tensorboard` argument in the configuration file must be set to `True` before training the neural network. The training progress can be visualized by running the following command in the terminal:

```
$tensorboard --logdir path/to/saved/log
```

The command would open localhost at `http://localhost:6006`

## Customization Guide
The majority of the codebase for this project is taken from the [PyTorch Template repository](https://github.com/victoresque/pytorch-template.git). For additional information on how to customize the project, please refer to the instructions available in the repository [[Link](https://github.com/victoresque/pytorch-template/blob/master/README.md)].

## Clean-up Guide
To remove the anaconda environment, navigate to the repository folder, and run the following command in the terminal:

```
$conda remove --name cnns --all
```

## Acknowledgements
The majority of the codebase for this project is taken from the [PyTorch Template repository](https://github.com/victoresque/pytorch-template.git).

## Authors
* Raghuram Shankar
* Sourab Bapu Sridhar

## License
This project is released under the terms of [MIT License](LICENSE).