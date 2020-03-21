# Transfer Learning Configuration File
import os

# initialize the base path to the *new* or processed directory contain our images
# after computeing the training and test split
BASE_DATA_PATH = os.path.sep.join(['/Volumes', 'MacBackup', 'hotdog_derived_dataset'])
BASE_OUTPUT_PATH = os.path.sep.join(['/Volumes', 'MacBackup', 'hotdog_transfer_learning_output'])

# define the names of the training, testing, and validation directories
TRAIN = "train"
TEST = "test"
VAL = "holdout"

# initialize the list of class labels
# the reason non_food is in index 0 is because of the way the files are provided and name encoded.
# all non-food images are of the form:  0_<num>.jpg all food images are of the form: 1_<num>.jpg
CLASSES=['hotdog', 'not_hotdog']

BATCH_SIZE = 32

# initialize the label encoder file path and the output directory to where the
# extracted features ( in CSV file format ) will be stored.
LE_FILE = os.path.sep.join([BASE_OUTPUT_PATH, "label_encoder.cpickle"])

# set the path to the serialized  model after training
MODEL_PATH = os.path.sep.join([BASE_OUTPUT_PATH, "model.cpickle"])

