from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import config
from imutils import paths
import numpy as np
import pickle
import random
import os
import logging
from pathlib import Path

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


def extract_features_from_model(base_model, feature_shape):

    label_encoder = None

    # loop over the data splits
    for split in (config.TRAIN, config.TEST, config.VAL):
        # grab all image paths in the current split
        p = os.path.sep.join([config.BASE_DATA_PATH, split])
        imagePaths = list(paths.list_images(p))

        logger.debug(f"Processing data split: {p} with {len(imagePaths)} images")

        # randomly shuffle the image paths and then extract the class labels
        # from the file paths.  It is more efficient to shuffle the classes
        # now, instead of during the training;
        random.shuffle(imagePaths)
        # get the labels in the same order as the random image paths
        # path/dataset/training/nonfood/0_123.jpb
        # index of -2 references 'nonfood'
        labels = [imagePath.split(os.path.sep)[-2] for imagePath in imagePaths]

        # if the label encoder is None, create it
        if label_encoder is None:
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)

        logger.debug(f"Class Labels:  {label_encoder.classes_}")

        # open the output CSV file for writing
        Path(config.BASE_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        csvPath = os.path.sep.join([config.BASE_OUTPUT_PATH, f"{split}.csv"])
        csv = open(csvPath, "w")

        # loop over the images in batches that match the batchsize
        # for the model
        # will feed the image through the model in batches
        # to get the resulting vector
        for (b, i) in enumerate(range(0, len(imagePaths), config.BATCH_SIZE)):
            # extract the batch of images and labels, then initialize the
            # list of actual images that will be passed through the network
            # for feature extraction
            logger.info(f"Processing batch {b+1}/{int(np.ceil(len(imagePaths)/float(config.BATCH_SIZE)))}")
            batchPaths = imagePaths[i:i+config.BATCH_SIZE]
            batchLabels = label_encoder.transform(labels[i:i+config.BATCH_SIZE])
            batchImages = []

            for imagePath in batchPaths:
                # load the input image using the keras helpt utility
                # while ensuring the image is resized to 224x224 pixels
                image = load_img(imagePath, target_size=(224,224))
                image = img_to_array(image)

                # preprocess the image by:
                # 1 - expanding the dimensions because the model expects and array of array of image values
                #           and what image currently is, is a single array
                image = np.expand_dims(image, axis=0)
                # 2 - subracting the mean RGB pixel intensity from the ImageNet dataset
                image = imagenet_utils.preprocess_input(image)

                # add the image to the batch collection
                batchImages.append(image)

            # at this point we are ready to pass the image through the model network to extract the
            # features.  which in this case is an array/vector of size:  7*7*512

            # pass the images through the network nad use the outputs as
            # our actual features, then reshape the features into a flattened volume
            batchImages = np.vstack(batchImages)
            # recall our base_model has the front FCN layer REMOVED so we are getting the output
            # of the convolutional network.
            features = base_model.predict(batchImages, batch_size=config.BATCH_SIZE)
            # reshape features into an array of array
            features = features.reshape((features.shape[0], feature_shape))

            # loop over the class labels and extracted features
            for (label, vec) in zip(batchLabels, features):
                # construct a row that exists of the class label and extracted features
                vec = ",".join([str(v) for v in vec])
                csv.write(f"{label},{vec}\n")

        # close file
        csv.close()

    f = open(config.LE_FILE, "wb")
    f.write(pickle.dumps(label_encoder))
    f.close()





def main():
    # Use model.summary() to see the last layer shape (None, 7, 7, 2048)  or (None, 7, 7, 512)
    import cnn_models

    for model in cnn_models.MODELS:
        logger.info(f"\tExtract feature for model: {model['name']}")
        print(f"{model['base_model'].summary()}")
        extract_features_from_model(model['base_model'], model['feature_shape'])



if __name__ == '__main__':
    logger.debug("Running Feature Extraction.....")
    import time
    s = time.time()
    main()
    e = time.time()
    logger.debug(f"DONE! Feature Extraction took: {(e-s)} seconds")