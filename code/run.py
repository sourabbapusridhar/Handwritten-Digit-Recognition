import os

import images
from src import dataset, evaluate_model, neural_network, prediction, prepare_pixel_data, run_test_harness

# repo path
repo_path = os.path.dirname(os.path.abspath(__file__))

# run example
def run_example():
    img_num = int(input('Enter image number to test: (0-9) \nor\n46 for multiple digits:\n'))

    if img_num < 0 or img_num > 9:
        if img_num != 46:
            raise IOError('Wrong input!')

    # add file paths
    filename = str(img_num)
    img_path = os.path.join(repo_path, "images", filename + ".png")

    # run test
    prediction.run_example(img_path)

def main():
    test_num = int(input('1. Run Evaluate Model\n2. Run Save Model\n3. Run Evaluate Final Model\n4. Run Prediction from user input\nEnter desired test to run:\n'))

    if test_num < 1 or test_num > 4:
        raise IOError('Wrong input!')
    
    if test_num == 1: run_test_harness.run_eval_model()
    if test_num == 2: run_test_harness.run_save_model()
    if test_num == 3: run_test_harness.run_eval_final_model()
    if test_num == 4: run_example()

main()