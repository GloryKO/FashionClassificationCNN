# FashionClassificationCNN
Fashion MNIST Image Classification
This is a machine learning project using TensorFlow and Keras to classify images from the Fashion MNIST dataset.

Dataset
The Fashion MNIST dataset consists of 60,000 grayscale images of clothing items belonging to 10 different classes. The images are 28x28 pixels in size and are pre-split into 50,000 training images and 10,000 testing images.

Model
For this project, I trained a convolutional neural network (CNN) using the Keras API from TensorFlow. The model consists of several layers of convolution, pooling, and dropout, followed by a dense output layer with 10 nodes (one for each class).

Evaluation
The trained model achieved an accuracy of 93% on the test set, which is quite good for this type of image classification task.

Files
train.py: Python script for training the model on the Fashion MNIST dataset
evaluate.py: Python script for evaluating the trained model on the test set
model.h5: Keras model saved in HDF5 format for future use
requirements.txt: List of Python dependencies for running the scripts
Usage
To train the model, run python train.py in the terminal. This will load the Fashion MNIST dataset, train the CNN model, and save the trained model to a file named model.h5.

To evaluate the trained model, run python evaluate.py in the terminal. This will load the saved model from the model.h5 file, evaluate it on the test set, and print out the accuracy score.

Conclusion
This project demonstrates the power of deep learning and convolutional neural networks for image classification tasks. With further tuning and optimization, it may be possible to achieve even higher accuracy on this dataset.
