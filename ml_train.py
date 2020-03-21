# USAGE
# python ml_train.py

# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import config
import numpy as np
import pickle
import os
import time
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)



def load_data_split(splitPath):
    # initialize the data and labels
    data = []
    labels = []

    # loop over the rows in the data split file
    for row in open(splitPath):
        # extract the class label and features from the row
        row = row.strip().split(",")
        label = row[0]
        features = np.array(row[1:], dtype="float")

        # update the data and label lists
        data.append(features)
        labels.append(label)

    # convert the data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    # return a tuple of the data and labels
    return (data, labels)


def train_ml_model(model_name):
    # derive the paths to the training and testing CSV files
    trainingPath = os.path.sep.join([config.BASE_OUTPUT_PATH, f"{config.TRAIN}.csv"])
    testingPath = os.path.sep.join([config.BASE_OUTPUT_PATH,f"{config.TEST}.csv"])

    # load the data from disk
    logging.debug("[INFO] loading data...")
    (trainX, trainY) = load_data_split(trainingPath)
    (testX, testY) = load_data_split(testingPath)

    # load the label encoder from disk
    le = pickle.loads(open(config.LE_FILE.format(model_name), "rb").read())

    # train the model
    logging.debug("[INFO] training model...")
    model = LogisticRegression(solver="lbfgs", multi_class="auto")
    model.fit(trainX, trainY)

    # evaluate the model
    logging.debug("[INFO] evaluating...")
    preds = model.predict(testX)
    logging.debug(classification_report(testY, preds, target_names=le.classes_))
    logging.debug(f"Accuracy: {accuracy_score(testY, preds)}")

    # serialize the model to disk
    logging.debug("[INFO] saving model...")
    f = open(config.MODEL_PATH, "wb")
    f.write(pickle.dumps(model))
    f.close()


if __name__ == '__main__':
    model_names = ['vgg16']
    for model_name in model_names:
        s = time.time()
        train_ml_model(model_name)
        e = time.time()
        logging.debug(f"Training Model: {model_name} took: {(e-s)} seconds")
