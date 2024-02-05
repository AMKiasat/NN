from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from amk_nn import train, test

if __name__ == '__main__':
    digits = load_digits()
    train_data, test_data, train_labels, test_labels = train_test_split(digits.data, digits.target, test_size=0.2,
                                                                        random_state=43)
    # print(digits.data.shape, train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
    layer_n = 2
    hiddenL_neuron_n = []
    train(train_data, train_labels, layer_n, hiddenL_neuron_n, epoch=50, learning_rate=0.01, activation_function=8)

    predictions = test(test_data)
    correct = 0
    for i in range(len(test_labels)):
        if predictions[i] == test_labels[i]:
            correct += 1
            # print(predictions[i])
    print("Number of correct answers: ", correct)
    print("Number of tests: ", len(test_labels))
    print("Accuracy: ", correct / len(test_labels))
