import streamlit as st
# import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import json
import requests
from config import MODEL_NAME, TARGET_SIZE, PORT
import glob
from pages.multi_pages import _get_state, page_image_classifier, page_tensorflow_serving


# print(tf.__version__)

def main():
    """Main function of the App"""
    st.set_page_config(page_title="TensorFlow Serving Manager", layout="wide")
    state = _get_state()
    pages = {
        "Image Classifier": page_image_classifier,
        "TensorFlow Serving": page_tensorflow_serving,
    }

    st.sidebar.title(":floppy_disk: Page states")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

@st.cache
def warm_up_model(ver):
    # load all images in a directory
    sample_images = []
    for filename in glob.glob('images/*.jpg'):
        img = Image.open(filename)
        sample_images.append(img)

    for img in sample_images:
        # You should make the size to the expected size
        resized_image = img.resize(TARGET_SIZE)
        array_image = image.img_to_array(resized_image)
        # 4D (batch_size, width, height, channels)
        input_image = np.expand_dims(array_image, axis=0)
        input_image = preprocess_input(input_image)

        data = json.dumps({"signature_name": "serving_default", "instances": input_image.tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post(f'http://localhost:{PORT}/v1/models/{MODEL_NAME}/versions/{ver}:predict',
                                      data=data,
                                      headers=headers)

        predictions = json.loads(json_response.text)['predictions']

    print("warm-up completed")

if __name__ == '__main__':
    main()