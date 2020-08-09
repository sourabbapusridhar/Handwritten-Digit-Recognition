import os

import images
from src import prediction

# add file paths
filename = "2"
repo_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(repo_path, "images", filename + ".png")

# run tests
prediction.run_example(img_path)