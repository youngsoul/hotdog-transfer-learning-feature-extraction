from tensorflow.keras.applications import VGG16, ResNet50V2, VGG19
from tensorflow.keras.layers import Input


MODELS = [
    {
        "base_model": VGG16(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3))),
        "name": "vgg16",
        "feature_shape": 7 * 7 * 512
    },
    # {
    #     "base_model": VGG19(weights="imagenet", include_top=False,
    #                         input_tensor=Input(shape=(224, 224, 3))),
    #     "name": "vgg19",
    #     "feature_shape": 7 * 7 * 512
    # },
    # {
    #     "base_model": ResNet50V2(weights="imagenet", include_top=False,
    #                              input_tensor=Input(shape=(224, 224, 3))),
    #     "name": "resnet50v2",
    #     "feature_shape": 7 * 7 * 2048
    #
    # }
]
