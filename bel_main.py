import bel_pred
import bel_preprocessing

# import cv2
from PIL import Image
import streamlit as st
import numpy as np
import tensorflow as tf
# import time


st.title('FLOWERS CLASSIFICATION')
st.write('\n')


st.write('Upload Image to classify')
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader(' ', type=['png', 'jpg', 'jpeg'])

image = Image.open('test.jpg')
show = st.image(image, use_column_width=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    cv_image = bel_preprocessing.convert_pil_to_cv(image)
    prep_image = bel_preprocessing.preprocess_image(cv_image)
    show.image(image, 'Uploaded Image', use_column_width=True)

st.write('\n')

if st.button('Click Here to Classify'):
    if uploaded_file is None:
        st.write('Please upload an Image to Classify')
    else:
        with st.spinner('Classifying ...'):
            prediction = bel_pred.multiclass_prediction(prep_image)
            value = "{}".format(prediction)
            st.write('**Name of the flower: **',value)