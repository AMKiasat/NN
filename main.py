import numpy as np
import math
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from amk_nn import train, test
import matplotlib.pyplot as plt


def split_train_test(data, test_ratio=0.2, random_seed=None):
    """Split the data into train and test sets."""
    if random_seed:
        np.random.seed(random_seed)

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data[train_indices], data[test_indices]


if __name__ == '__main__':
    digits = load_digits()
    train_data, test_data, train_labels, test_labels = train_test_split(digits.data, digits.target, test_size=0.2,
                                                                        random_state=42)
    # print(digits.data.shape, train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
    layer_n = 4
    hiddenL_neuron_n = [10, 10]
    train(train_data, train_labels, layer_n, hiddenL_neuron_n, epoch=20, learning_rate=0.01)
    predictions = test(test_data)
    correct = 0
    for i in range(len(test_labels)):
        if predictions[i] == test_labels[i]:
            correct += 1
    print(correct, len(test_labels), correct / len(test_labels))
