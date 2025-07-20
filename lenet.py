from tensorflow import keras 
from keras import layers
from keras.utils import to_categorical
from keras.models import load_model 
import matplotlib.pyplot as plt
import argparse

def create_lenet5(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), activation='tanh', input_shape=input_shape)),
    model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))),

    model.add(layers.Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), activation='tanh')),
    model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))),

    model.add(layers.Conv2D(filters=120, kernel_size=(5,5), strides=(1,1), padding='same', activation='tanh')),

    model.add(layers.Flatten()),
    model.add(layers.Dense(84, activation='tanh')),
    model.add(layers.Dense(10, activation='softmax')),

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 

    parser.add_argument("--train", action="store_true", help='Train a model.')
    parser.add_argument("--evaluate", help='Train a model.')
    parser.add_argument("--predict", type=int, help='Train a model.')

    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print("Dataset Information")
    print(f"Train shape: {x_train[0].shape}")
    print(f"Test shape: {y_train[0].shape}")
    print(f"Training count: {len(x_train)}")
    print(f"Testing count: {len(y_train)}")

    input_shape = (28, 28, 1)

    lenet = create_lenet5()

    lenet.summary()
    lenet.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Train the model
    if args.train:
        lenet.fit(
            x=x_train,
            y=y_train,
            epochs=10,
            batch_size=128,
        )

        lenet.save('./lenet.keras')

    # Evaluate the model
    if args.evaluate:
        print("Loading Model...")
        lenet = load_model('lenet.keras')

        loss, acc = lenet.evaluate(
            x=x_test,
            y=y_test
        )

        print(f"Model loss: {loss}")
        print(f"Model acc: {acc}")

    if args.predict:
        print("Loading Model...")
        lenet = load_model('lenet.keras')

        image_index = args.predict
        print(f"Predicting image: {image_index}")

        # Display the number 
        plt.imshow(x_test[image_index].reshape(28,28), cmap='Greys')
        pred = lenet.predict(x_test[image_index].reshape(1, 28, 28, 1))

        print(f"Model prediction: {pred.argmax()}")
        plt.show()

