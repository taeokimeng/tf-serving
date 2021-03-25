from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import json
import requests
from config import MODEL_NAME, MODEL_VERSION


if MODEL_NAME == 'vgg16':
    model = VGG16(weights='imagenet')
    target_size = (224, 224)

elif MODEL_NAME == 'inceptionv3':
    model = InceptionV3(weights='imagenet')
    target_size = (299, 299)

img_path = 'images/lion.jpg'
img = image.load_img(img_path, target_size=target_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

model.save(f'models/{MODEL_NAME}/{MODEL_VERSION}')

# data = json.dumps({"instances": x.tolist()})
# headers = {"content-type": "application/json"}
# json_response = requests.post(f'http://localhost:8505/v1/models/{MODEL_NAME}/versions/{MODEL_VERSION}:predict',
#                               data=data,
#                               headers=headers)
#
# predictions = json.loads(json_response.text)
# print('Predicted:', decode_predictions(preds, top=3)[0])