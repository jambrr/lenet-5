
# LeNet-5 MNIST Classifier (Keras)

This is a Python implementation of the classic **LeNet-5** Convolutional Neural Network using **TensorFlow/Keras**, trained and evaluated on the **MNIST dataset**. Created for educational purposes, in order to understand the architecture. The script supports training, evaluation, and prediction via command-line arguments.

## Requirements

* Python 3.7+
* TensorFlow / Keras
* Matplotlib
* NumPy

Install dependencies (if not already installed):

```bash
pip install tensorflow matplotlib
```

## Usage

This script uses `argparse` to control the mode of operation:

### 1. Train the model

```bash
python lenet.py --train
```

* Loads the MNIST dataset
* Builds and trains the LeNet-5 model for 10 epochs
* Saves the model to `lenet.keras`

### 2. Evaluate the trained model

```bash
python lenet.py --evaluate True
```

* Loads the saved model from `lenet.keras`
* Evaluates it on the MNIST test set

### 3. Predict an image by index

```bash
python lenet.py --predict 7
```

* Loads the saved model
* Displays the 7th image from the test set
* Outputs the predicted digit

## LeNet-5 Architecture

The model is composed of:

1. `Conv2D(6 filters, 5x5 kernel, tanh)`
2. `AveragePooling2D(2x2)`
3. `Conv2D(16 filters, 5x5 kernel, tanh)`
4. `AveragePooling2D(2x2)`
5. `Conv2D(120 filters, 5x5 kernel, tanh)`
6. `Flatten`
7. `Dense(84, tanh)`
8. `Dense(10, softmax)`

Loss: `categorical_crossentropy`
Optimizer: `SGD`
Metrics: `accuracy`

## Dataset Info

* Dataset: MNIST (handwritten digits 0–9)
* Shape: 28×28 grayscale images
* Classes: 10

## Output

* Trained model saved to: `lenet.keras`
* Evaluation output: accuracy and loss on test set
* Predictions shown via matplotlib

## Notes

* You must **run `--train` at least once** to create the `lenet.keras` file before using `--evaluate` or `--predict`.
* Model input shape is automatically reshaped to `(28, 28, 1)` if needed.

