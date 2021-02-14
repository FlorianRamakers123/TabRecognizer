import dataset.generator as dsg
import dataset.loader as dsl
import os
import neural_network.convolutional as conv
import concurrent.futures as cf


def train(string_index):
    train_set = dsl.load_train_data(string_index)
    test_set = dsl.load_test_data(string_index)

    model = conv.create_neural_network(dsg.get_input_shape(), dsg.get_output_shape())
    model.summary()
    print(conv.train_neural_network(model, train_set, test_set))

def main():

    # Create the train en test set if it does not exist.
    if not os.listdir(dsg.TRAINING_SET_PATH) or not os.listdir(dsg.TEST_SET_PATH):
        dsg.create_input_data()

    train(0)
    # Train a convolutional neural network for each guitar string
    #with cf.ThreadPoolExecutor(max_workers=dsg.STRING_COUNT) as executor:
    #    executor.map(train, range(dsg.STRING_COUNT))

if __name__ == "__main__":
    main()