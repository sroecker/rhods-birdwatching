import numpy as np
from requests import post
from classes import default_class_labels
import streamlit as st
import torchvision.transforms.functional as fn
from torchvision.transforms import v2
from PIL import Image

DEFAULT_SIZE = (256, 256)

PREDICTION_URL = 'https://birds-birdwatching.apps.rhods-internal.61tk.p1.openshiftapps.com/v2/models/birds/infer'

def serialize_image(img):
    img_arr = fn.pil_to_tensor(img).unsqueeze(0)
    print(img_arr.shape)
    payload = {
        'inputs': [
            {
                'name': 'image_1_3_256_256',
                'shape': [1, 3, DEFAULT_SIZE[0], DEFAULT_SIZE[1]],
                'datatype': 'FP32',
                'data': img_arr.flatten().tolist(),
            }
        ]
    }
    return payload


st.set_page_config(page_title='RHODS Birdwatching demo')

st.title('Drag and Drop scoring')

img_data = st.file_uploader(label='Upload bird image for scoring', type=['png', 'jpg', 'jpeg'])
if img_data is not None:
    uploaded_img = Image.open(img_data)
    print(uploaded_img.size)

    transform = v2.Compose([v2.Resize(size=DEFAULT_SIZE)])
    new_img = transform(uploaded_img)
    print(new_img.size)
    
    st.image(new_img)
    
    payload = serialize_image(new_img)
    raw_response = post(PREDICTION_URL, json=payload)
    print(raw_response.text)
    
    res = raw_response.json()
    prediction = res['outputs'][0]['data']
    pred_class = default_class_labels[np.argmax(prediction)]
    st.write(f'Predicted class: {pred_class}')
