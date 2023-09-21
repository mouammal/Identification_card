# Card Recognition System

This repository contains code for a Card Recognition System. The system uses machine learning models to recognize and classify playing cards. The code is written in Python and uses various libraries and models for image processing and classification.

## Setup

Before running the code, make sure to set up the following:

1. **Data**: Ensure that the card images dataset is available. The default path is set to "./data," but you can adjust it to your dataset location.

2. **Dependencies**: Install the necessary Python libraries using `pip install numpy matplotlib opencv-python scikit-learn seaborn`.

3. **Note**: Some parts of the code require significant computational resources, especially hyperparameter optimization, which may take a long time to run.

## Data Preprocessing

The code includes data preprocessing steps to prepare the card images for classification:

- Shuffle the list of card image filenames.
- Extract labels for each card image based on the filename.

## Image Features

The code extracts image features using grayscale conversion and SIFT (Scale-Invariant Feature Transform) feature extraction. This helps in training and evaluating machine learning models.

## Machine Learning Models

### 1. K-Nearest Neighbors (KNN)

- Classifier: KNeighborsClassifier
- Hyperparameters: `n_neighbors=1`, `p=1`
- Accuracy on testing set: 0.389

### 2. Random Forest

- Classifier: RandomForestClassifier
- Hyperparameters: `criterion='entropy'`, `n_estimators=300`
- Accuracy on testing set: 0.562

### 3. Support Vector Classifier (SVC)

- Classifier: SVC (Kernel: RBF)
- Accuracy on testing set: 0.551

### 4. Multi-Layer Perceptron (MLP)

- Classifier: MLPClassifier
- Hyperparameters: Varies (optimized using GridSearchCV)
- Accuracy on testing set: 0.577

### Hyperparameter Optimization

The code includes hyperparameter optimization for the MLP model using GridSearchCV. The best parameters found are:

- `activation`: 'relu'
- `alpha`: 0.05
- `hidden_layer_sizes`: (150, 100, 50)
- `learning_rate`: 'adaptive'
- `max_iter`: 150
- `solver`: 'adam'

### K-Nearest Neighbors (KNN) Hyperparameters

The code also includes a visualization of KNN accuracy variations with different values of K (number of neighbors) using both Euclidean and Manhattan distances.

### Support Vector Classifier (SVC) with SIFT Features

The code optimizes the SVC model with SIFT feature extractions and a reduced dataset of 1000 images. The accuracy achieved on the testing set is 0.735.

## Results

The code generates a bar plot displaying the accuracy of different classification models, including KNN, Random Forest, SVC, MLP, and SVC with SIFT features.

## Conclusion

This code provides a comprehensive pipeline for training, evaluating, and optimizing machine learning models for playing card recognition. It demonstrates the use of various classifiers and preprocessing techniques to achieve accurate classification results.
