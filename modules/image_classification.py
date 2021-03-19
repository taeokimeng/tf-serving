import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import json
import requests
from config import MODEL_NAME, MODEL_VERSION, MODEL_VERSIONS, TARGET_SIZE, PORT

@st.cache
def image_classifier(loaded_image, port, ver):
    # You should make the size to the expected size
    resized_image = loaded_image.resize(TARGET_SIZE)
    array_image = image.img_to_array(resized_image)
    # 4D (batch_size, width, height, channels)
    input_image = np.expand_dims(array_image, axis=0)
    input_image = preprocess_input(input_image)

    data = json.dumps({"signature_name": "serving_default", "instances": input_image.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(f'http://localhost:{port}/v1/models/{MODEL_NAME}/versions/{ver}:predict',
                                  data=data,
                                  headers=headers)

    predictions = json.loads(json_response.text)['predictions']
    label = decode_predictions(np.array(predictions))

    # Return the top 1 predicted label
    return label[0][0][1]