# Author: Kee Chee Yau
# Last Modified : 9th June 2019
#
# This code is only to import some general functions into the jupyter notebook that will make it less messy

import numpy as np
from IPython.display import Markdown, display
from keras.preprocessing import image

from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image

# print markdown string
def printmd(string):
    display(Markdown(string))
    
    
def load_img_array(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x