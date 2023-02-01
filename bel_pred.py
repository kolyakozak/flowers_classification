import os
import bel_preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from PIL import Image
import numpy as np


model = tf.keras.models.load_model('mobilenet_model.hdf5')

def multiclass_prediction(image):
    classes = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
    y_pred = model.predict(np.array([image]))
    indx = np.argmax(y_pred[0])
    return classes[indx]
