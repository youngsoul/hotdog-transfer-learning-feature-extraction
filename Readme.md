# HotDog, Not-HotDog Transfer Learning Feature Extraction

This repo will example of Deep Transfer Learning using feature extraction for the Silicon Valley "HotDog, Not-HotDog" image classifier

The transfer learning employed in this example is call `feature extraction`.

We will use the VGG16 CNN model, remove the fully connected network layer and use the output of the convolutional layers to create features to train a LogisticRegression model.



## Dataset

I had previously curated a dataset that was a combination of a Kaggle dataset and images from ImageNet.

You can find that dataset on my [Github repo here](https://github.com/youngsoul/hotdog-not-hotdog-dataset).

### Training HotDog Pictures

![hotdogs](./doc_images/hotdog_collage.png)


### Training Not-HotDog Pictures

![nothotdogs](./doc_images/not_hotdog_collage.png)


### Holdout Validation Pictures

![val](./doc_images/holdout_validation_collage.png)


## Configuration

`config.py`

The config file contains path configurations to the dataset mentioned above, and paths to where the output of the feature extraction and model training.


## Feature Extraction

`extract_features.py`

The feature extraction script will create csv files for each of the train/test/holdout image directories.

```text
.
├── holdout.csv
├── test.csv
└── train.csv

```


## Model Training

`ml_train.py`

The model training script will use the csv files produced from the feature extraction step to train a LogisticRegression model.

When it completes the output directory will have a pickle file for the label encoder and the trained model.
```text
.
├── holdout.csv
├── label_encoder.cpickle
├── model.cpickle
├── test.csv
└── train.csv

```

Training metrics using the train and test files is shown below:

```text
DEBUG:root:    precision    recall  f1-score   support

     hot_dog       0.94      0.99      0.97       200
 not_hot_dog       0.99      0.94      0.96       200

    accuracy                           0.96       400
   macro avg       0.97      0.97      0.96       400
weighted avg       0.97      0.96      0.96       400

DEBUG:root:Accuracy: 0.965
DEBUG:root:[INFO] saving model...
DEBUG:root:Training Model: vgg16 took: 25.940191984176636 seconds

```

## Validation using the extracted validation features

`ml_validate.py`

This script will use holdout.csv file to test the performance of the model on the validation images after feature extraction.

```text
DEBUG:root:    precision    recall  f1-score   support

     hot_dog       0.94      0.99      0.97       100
 not_hot_dog       0.99      0.94      0.96       100

    accuracy                           0.96       200
   macro avg       0.97      0.96      0.96       200
weighted avg       0.97      0.96      0.96       200

DEBUG:root:Accuracy: 0.965
DEBUG:root:Training Model: vgg16 took: 0.8644349575042725 seconds

```

## Validate on actual images

`ml_predict.py`

This script will start with the images used for validation, and run the raw image data through the VGG16 CNN for feature extraction and then through the saved LogisticRegression model.

It will also display the images that were incorrectly classified.

### Metrics
```text
DEBUG:root:Model: vgg16
DEBUG:root:Accuracy: 0.96
DEBUG:root:Total Images: 200
DEBUG:root:Total Errors: 7

```

## Results

out of the 200 Holdout Validation pictures, there were 7 mis-classified pictures.

Below are the pictures that were misclassififed and how they were misclassified.

![PredictionResults](./doc_images/2020-03-21_10-55-14%20(1).gif)
