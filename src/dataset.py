# Importing dataset and cleaing up dataset for processing
from keras.datasets import mnist

(trainX, trainY), (testX, testY) = mnist.load_data()