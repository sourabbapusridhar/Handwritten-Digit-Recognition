from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imageio
 
# Load and prepare the image

def load_image(filename):
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img
 
# Load an image and predict the class

def run_example(img_path):
    img = load_image(img_path)
    model = load_model('final_model.h5')
    digit = model.predict_classes(img)
    print(digit)
