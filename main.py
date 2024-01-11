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
        if i[0] < -1:
            x1.append([-1])
        elif i[0] > 1:
            x1.append([1])
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
        # print(i)
        if i[0] < 0:
            x1.append([0])
        else:
            x1.append(i)

    return x1


def softmax(x):
    """ Compute softmax values for each sets of scores in x. """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def feed_forward(n, w, b, af):
    for i in range(n.__len__() - 1):
        tmp = w[i].dot(n[i]) + b[i]
        # print(tmp, "\n")

        if af == 0:
            n[i + 1] = gaussian(tmp)
        elif af == 1:
            n[i + 1] = sigmoid(tmp)
        elif af == 2:
            n[i + 1] = step(tmp)
        elif af == 3:
            n[i + 1] = sign(tmp)
        elif af == 4:
            n[i + 1] = saturated_linear(tmp)
        elif af == 5:
            n[i + 1] = linear(tmp)
        elif af == 6:
            n[i + 1] = tanh(tmp)
        elif af == 7:
            n[i + 1] = ReLU(tmp)
        elif af == 8:
            n[i + 1] = softmax(tmp)


if __name__ == '__main__':
    """User must set these:   """
    epoch = 20
    layer_num = 3
    input_neuron_num = 8
    hiddenL_neuron_num = [4, 3]  # 8 4 3
    neurons = [np.random.randint(-10, 10, size=(input_neuron_num, 1))]
    activation_function = 2

    for i in hiddenL_neuron_num:
        neurons.append(np.zeros(i).T)

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
        # print(temp)
    # print(wi)
    # print(bi)

    for i in range(epoch):
        feed_forward(neurons, wi, bi, activation_function)
        # for i in neurons:
        #     print(i, "\n")

    # x = np.linspace(-10, 10)
    # plt.plot(x, gaussian(x))
    # plt.axis('tight')
    # plt.title('Activation Function :binaryStep')
    # plt.show()
