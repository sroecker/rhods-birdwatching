import numpy as np
from requests import post
from classes import default_class_labels
import streamlit as st
import torch
from PIL import Image

DEFAULT_SIZE = (224, 224)

PREDICTION_URL = 'https://birds-birdwatching.apps.rhods-internal.61tk.p1.openshiftapps.com/v2/models/birds/infer'

def reshape_image_onnx(img_arr):
    # change to channels first
    # and add one dimension
    return np.rollaxis(img_arr, 2)[np.newaxis, :]

def serialize_image(img):
    img_arr = np.asarray(img, dtype='int32')
    print(img_arr.shape)
    img_re = reshape_image_onnx(img_arr)
    print(img_re.shape)
    # TODO fix defaults
    payload = {
        'inputs': [
            {
                'name': 'image_1_3_256_256',
                'shape': [1, 3, 256, 256],
                'datatype': 'FP32',
                'data': img_re.flatten().tolist(),
            }
        ]
    }
    return payload


st.set_page_config(page_title='Birdwatching')

st.title('Drag and Drop scoring')

img_data = st.file_uploader(label='Upload bird image for scoring', type=['png', 'jpg', 'jpeg'])
if img_data is not None:
    uploaded_img = Image.open(img_data)
    print(uploaded_img.size)
    uploaded_img = uploaded_img.resize((256, 256), resample=Image.Resampling.BICUBIC)
    print(uploaded_img.size)
    st.image(uploaded_img)
    payload = serialize_image(uploaded_img)
    raw_response = post(PREDICTION_URL, json=payload)
    print(raw_response.text)
    res = raw_response.json()
    pred_class = default_class_labels[np.argmax(res['outputs'][0]['data'])+3]
    st.write(f'Predicted class: {pred_class}')
