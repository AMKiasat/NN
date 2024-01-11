import numpy as np
import matplotlib.pyplot as plt


def gaussian(x):
    """ It returns exp(-x^2) where the values lie between '0' and '1' """
    return np.exp(-1 * np.square(x))


def sigmoid(x):
    """ It returns 1/(1+exp(-x)) where the values lie between '0' and '1' """
    return 1 / (1 + np.exp(-x))


def step(x):
    """ It returns '0' if the input is less than zero otherwise it returns '1' """
    return np.heaviside(x, 1)


def sign(x):
    """ It returns '-1' if the input is less than zero & returns '0' if the input is zero otherwise it returns '1' """
    return np.sign(x)


def saturated_linear(x):
    """ It returns '-1' if the input is less than -1 & returns '1' if the input is greater than 1 otherwise it
    returns the given input."""
    x1 = []
    for i in x:
        if i < -1:
            x1.append(-1)
        elif i > 1:
            x1.append(1)
        else:
            x1.append(i)

    return x1


def linear(x):
    """ y = f(x) It returns the input as it is"""
    return x


def tanh(x):
    """ It returns the value (1-exp(-2x))/(1+exp(-2x)) where the values lie between '-1' and '1'."""
    return np.tanh(x)


def ReLU(x):
    """ It returns zero if the input is less than zero otherwise it returns the given input. """
    x1 = []
    for i in x:
        if i < 0:
            x1.append(0)
        else:
            x1.append(i)

    return x1


def softmax(x):
    """ Compute softmax values for each sets of scores in x. """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# def feed_forward(n, w, b):
#     for i in range(n.size() - 1):
#         n[i + 1] = w[i].dot(n[i]) + b[i]


if __name__ == '__main__':
    """User must set these:   """
    epoch = 20
    layer_num = 3
    input_neuron_num = 8
    hiddenL_neuron_num = [4, 3]  # 8 4 3
    neurons = [np.random.randint(-10, 10, input_neuron_num)]

    for i in hiddenL_neuron_num:
        neurons.append(np.zeros(i).T)
    print(neurons)

    """Making random wights and biases"""
    wi = []
    bi = []
    for i in range(layer_num - 1):
        if i == 0:
            tmp = np.random.rand(hiddenL_neuron_num[0], input_neuron_num)
            temp = np.random.rand(hiddenL_neuron_num[0], 1)
        else:
            tmp = np.random.rand(hiddenL_neuron_num[i], hiddenL_neuron_num[i - 1])
            temp = np.random.rand(hiddenL_neuron_num[i], 1)
        wi.append(tmp)
        bi.append(temp)
        # print(tmp)
        # print(temp)
    # print(wi)
    # print(bi)

    # for i in range(epoch):
    #     feed_forward(neurons, wi, bi, activation_function)

    # x = np.linspace(-10, 10)
    # plt.plot(x, gaussian(x))
    # plt.axis('tight')
    # plt.title('Activation Function :binaryStep')
    # plt.show()
