# Fashion MNIST Classification using CNN

This repository contains implementations of a Convolutional Neural Network (CNN) for classifying the Fashion MNIST dataset. The CNN is implemented in both Python and R using the Keras library.

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow and Keras libraries for Python
- R with the keras package installed
- Fashion MNIST dataset (automatically downloaded by the code)

### Instructions for Python
1. Install the required packages using:
   ```bash
   pip install tensorflow matplotlib
   ```
2. Run the Python script:
   ```bash
   python fashion_mnist_cnn.py
   ```

### Instructions for R
1. Install the required R packages using:
   ```R
   install.packages("keras")
   library(keras)
   install_keras()
   ```
2. Run the R script:
   ```R
   source("fashion_mnist_cnn.R")
   ```

## Output
Both scripts will:
1. Train a CNN model to classify the Fashion MNIST dataset.
2. Save the trained model as `fashion_mnist_cnn.h5`.
3. Display predictions for two sample images from the dataset.

## Notes
- Ensure you have sufficient compute resources for training the model.
- The model architecture includes six layers: three convolutional layers, two pooling layers, and two dense layers (one for output).

## Contact
For any questions, please contact Elvis9819.