import os

import images
from src import dataset, evaluate_model, neural_network, prediction, prepare_pixel_data, run_test_harness

# repo path
repo_path = os.path.dirname(os.path.abspath(__file__))

# save model



# run example

img_num = int(input('Enter image number to test: (0-9) \nor\n46 for multiple digits:\n'))

if img_num < 0 or img_num > 9:
    if img_num != 46:
        raise IOError('Wrong input!')

# add file paths
filename = str(img_num)
img_path = os.path.join(repo_path, "images", filename + ".png")

# run tests
prediction.run_example(img_path)