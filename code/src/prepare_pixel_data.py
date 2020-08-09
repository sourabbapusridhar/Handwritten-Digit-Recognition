# Implementation to Modify Pixel Data

from keras.datasets import mnist

def prepare_pixel_data(train, test):
    train_normalized = train.astype('float32')
    test_normalized = test.astype('float32')
    train_normalized = train_normalized/255.0
    test_normalized = test_normalized/255.0
    return train_normalized, test_normalized
