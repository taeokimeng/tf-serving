import streamlit as st
from random import randint
from config import MODEL_VERSIONS, MODELS
from PIL import Image
from modules.image_classification import image_classifier

def upload_images(state):
    uploaded_images = st.file_uploader("Please choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key=state.widget_key)
    target_size = (300, 200)
    if st.button("Clear uploaded files"):
        state.widget_key = str(randint(1000, 100000000))
    if len(uploaded_images) > 0:
        images_list = []
        names_list = []
        for img in uploaded_images:
            try:
                image_pil = Image.open(img)
            except Exception: # Check invalid image
                st.error("Error: Invalid image")
            else:
                image_pil = image_pil.resize(target_size, Image.ANTIALIAS)
                images_list.append(image_pil)
                names_list.append(img.name)
        return images_list, names_list
    else:
        return None, None

def display_image_classifier(state):
    st.title(":mag: Image Classifier")
    host_port = st.text_input("Host port:", "8505")
    images_pil, file_names = upload_images(state)
    # Image has been selected
    if images_pil is not None:
        if len(images_pil) < 4:
            cols = list(st.beta_columns(len(images_pil)))
        else:
            cols = list(st.beta_columns(3))

        model = st.selectbox("Choose a model", list(MODELS.keys()))
        version = st.radio("Select the version", MODELS[model]["Versions"])
        if st.button("Request prediction"):
            for i, img in enumerate(images_pil):
                idx = i % 3
                # start = time.time()
                cols[idx].subheader(image_classifier(img, host_port, MODELS[model]["Name"], MODELS[model]["ImportName"], version, MODELS[model]["TargetSize"])) # state.host_port
                # print((time.time()-start))
                cols[idx].image(img, caption=file_names[i], width=192, use_column_width='auto')
            st.write("**Classification has been done!**")

