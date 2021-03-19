# import tensorflow as tf
# from tensorflow_serving.apis import classification_pb2
# from tensorflow_serving.apis import inference_pb2
# from tensorflow_serving.apis import model_pb2
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_log_pb2
# from tensorflow_serving.apis import regression_pb2
#
# def main():
#     with tf.python_io.TFRecordWriter("tf_serving_warmup_requests") as writer:
#         # replace <request> with one of:
#         # predict_pb2.PredictRequest(..)
#         # classification_pb2.ClassificationRequest(..)
#         # regression_pb2.RegressionRequest(..)
#         # inference_pb2.MultiInferenceRequest(..)
#         log = prediction_log_pb2.PredictionLog(
#             predict_log=prediction_log_pb2.PredictLog(request=<request>))
#         writer.write(log.SerializeToString())
#
# if __name__ == "__main__":
#     main()



# """Generate Warmup requests."""
# import tensorflow as tf
# import requests
# import base64
# from tensorflow.python.framework import tensor_util
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_log_pb2
#
# IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
# NUM_RECORDS = 100
#
# def get_image_bytes():
#     image_content = requests.get(IMAGE_URL, stream=True)
#     image_content.raise_for_status()
#     return image_content.content
#
# def main():
#     """Generate TFRecords for warming up."""
#     with tf.io.TFRecordWriter("models/vgg16/1/assets.extra/tf_serving_warmup_requests") as writer:
#         image_bytes = get_image_bytes()
#         predict_request = predict_pb2.PredictRequest()
#         predict_request.model_spec.name = 'vgg16'
#         predict_request.model_spec.signature_name = 'serving_default'
#         predict_request.inputs['input_1'].CopyFrom(
#             tensor_util.make_tensor_proto([image_bytes]))
#         log = prediction_log_pb2.PredictionLog(
#             predict_log=prediction_log_pb2.PredictLog(request=predict_request))
#         for r in range(NUM_RECORDS):
#             writer.write(log.SerializeToString())
# 
# if __name__ == "__main__":
#     main()


import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from config import MODEL_NAME, MODEL_VERSION

IMAGE_PATH = './images/lion.jpg'
NUM_RECORDS = 100

X_new = tf.io.read_file(IMAGE_PATH)
X_new = tf.image.decode_image(X_new)
X_new = tf.image.resize(X_new, (224, 224))
X_new = tf.cast(X_new, tf.float32)/255.0

def main():
    """Generate TFRecords for warming up."""

    with tf.io.TFRecordWriter(f"models/{MODEL_NAME}/{MODEL_VERSION}/assets.extra/tf_serving_warmup_requests") as writer:
        predict_request = predict_pb2.PredictRequest()
        predict_request.model_spec.name = f'{MODEL_NAME}'
        predict_request.model_spec.signature_name = 'serving_default'
        predict_request.inputs["input_1"].CopyFrom(tf.make_tensor_proto(tf.reshape(X_new, (-1, 224, 224, 3))))
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=predict_request))
        for r in range(NUM_RECORDS):
            writer.write(log.SerializeToString())

if __name__ == "__main__":
    main()

