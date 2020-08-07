from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imageio
 
# Load and prepare the image

def load_image(filename):
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    img = img_to_array(img)
    imageio.imwrite('save_3.jpg', img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img
 
# Load an image and predict the class

def run_example():
    img = load_image('2.bmp')
    model = load_model('final_model.h5')
    digit = model.predict_classes(img)
    print(digit)
    # print('Done')
 
run_example()