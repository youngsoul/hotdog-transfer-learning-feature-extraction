import pickle
import config
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import cnn_models
import numpy as np
from imutils import paths
from tensorflow.keras.applications import imagenet_utils
import cv2
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


def display_prediction_errors(error_images):
    for error_image in error_images:
        image = cv2.imread(error_image[0])
        image = cv2.resize(image, (400, 400))

        cv2.putText(image, f"Actual: {error_image[1]}", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f"Predicted: {error_image[2]}", (3, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Prediction Error", image)
        cv2.waitKey(0)


if __name__ == '__main__':

    model_name = 'vgg16'
    cnn_feature_extractor_model = cnn_models.MODELS[0]['base_model']
    feature_shape = cnn_models.MODELS[0]['feature_shape']

    # get image paths for validation images
    validation_images_path = os.path.sep.join([config.BASE_DATA_PATH, config.VAL])
    imagePaths = list(paths.list_images(validation_images_path))

    images = []  # images to predict on
    labels = []

    # load images - we are loading all of the validation images but
    # in practice we would likely only load a limited set
    for imagePath in imagePaths:
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by:
        # 1 - expanding the dimensions because the model expects and array of array of image values
        #           and what image currently is, is a single array
        image = np.expand_dims(image, axis=0)
        # 2 - subracting the mean RGB pixel intensity from the ImageNet dataset
        image = imagenet_utils.preprocess_input(image)

        images.append(image)

        label = imagePath.split('/')[-2]
        labels.append(label)

    # pass the images through the network nad use the outputs as
    # our actual features, then reshape the features into a flattened volume
    batchImages = np.vstack(images)

    # recall our base_model has the front FCN layer REMOVED so we are getting the output
    # of the convolutional network.
    image_features = cnn_feature_extractor_model.predict(batchImages, batch_size=config.BATCH_SIZE)

    # reshape features into an array of array
    image_features = image_features.reshape((image_features.shape[0], feature_shape))

    # load the label encoder from disk
    le = pickle.loads(open(config.LE_FILE.format(model_name), "rb").read())

    f = open(config.MODEL_PATH, 'rb')
    model = pickle.loads(f.read())
    logging.debug(model)

    # PREDICT
    predicted_labels = []
    image_errors = []
    for i, feature in enumerate(image_features):
        prediction = model.predict(np.expand_dims(feature, axis=0))
        pred_label = le.classes_[int(prediction[0])]
        predicted_labels.append(pred_label)
        if pred_label != labels[i]:
            image_errors.append((imagePaths[i], labels[i], pred_label))
            logging.debug(f"Image: {imagePaths[i]} Actual: {labels[i]}  Predicted: {pred_label}")

    logging.debug(f"Model: {model_name}")
    logging.debug(f"Accuracy: {(len(imagePaths) - len(image_errors)) / len(imagePaths):.2f}")
    logging.debug(f"Total Images: {len(imagePaths)}")
    logging.debug(f"Total Errors: {len(image_errors)}")

    display_prediction_errors(image_errors)
