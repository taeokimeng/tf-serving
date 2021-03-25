import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import importlib
import numpy as np
import json
import requests
from config import MODEL_NAME, MODEL_VERSION, MODEL_VERSIONS, TARGET_SIZE, PORT

@st.cache
def image_classifier(loaded_image, port, modeL_name, import_name, ver, target_size):
    # You should make the size to the expected size
    # resized_image = loaded_image.resize(TARGET_SIZE)
    path_name = f'tensorflow.keras.applications.{import_name}'
    modules = importlib.import_module(path_name)
    resized_image = loaded_image.resize(target_size)
    array_image = image.img_to_array(resized_image)
    # 4D (batch_size, width, height, channels)
    input_image = np.expand_dims(array_image, axis=0)
    # input_image = preprocess_input(input_image)
    input_image = modules.preprocess_input(input_image)

    data = json.dumps({"signature_name": "serving_default", "instances": input_image.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(f'http://localhost:{port}/v1/models/{modeL_name}/versions/{ver}:predict',
                                  data=data,
                                  headers=headers)

    predictions = json.loads(json_response.text)['predictions']
    # label = decode_predictions(np.array(predictions))
    label = modules.decode_predictions(np.array(predictions))

    # Return the top 1 predicted label
    return label[0][0][1]