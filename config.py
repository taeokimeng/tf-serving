MODEL_NAME = "inceptionv3"
MODEL_VERSION = 1
MODEL_VERSIONS = (1, 2)
TARGET_SIZE = (224, 224)
MODELS = {
    "VGG16": {
        "Name": "vgg16",
        "ImportName": "vgg16",
        "TargetSize": (224, 224),
        "Versions": (1, 2)
    },
    "InceptionV3": {
        "Name": "inceptionv3",
        "ImportName": "inception_v3",
        "TargetSize": (299, 299),
        "Versions": (1, )
    }
}
PORT = "8501"