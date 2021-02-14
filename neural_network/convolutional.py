from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

EPOCHS = 3
BATCH_SIZE = 64

def create_neural_network(input_shape, output_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(output_shape, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_neural_network(model, training_set, test_set):
    model.fit(training_set[0], training_set[1], validation_data=test_set, epochs=EPOCHS, batch_size=BATCH_SIZE)
    return model.evaluate(test_set[0], test_set[1], verbose=1)
