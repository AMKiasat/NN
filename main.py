import numpy as np
import matplotlib.pyplot as plt


def gaussian(x):
    """ It returns exp(-x^2) where the values lie between '0' and '1' """
    return np.exp(-1 * np.square(x))


def gaussian_derivative(x, sigma):
    return (-x / sigma ** 2) * np.exp(-x ** 2 / (2 * sigma ** 2))


def sigmoid(x):
    """ It returns 1/(1+exp(-x)) where the values lie between '0' and '1' """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def step(x):
    """ It returns '0' if the input is less than zero otherwise it returns '1' """
    return np.heaviside(x, 1)


def step_derivative(x):
    x1 = []
    for k in x:
        if k[0] == 0:
            x1.append([1])
        else:
            x1.append([0])
    return x1


def sign(x):
    """ It returns '-1' if the input is less than zero & returns '0' if the input is zero otherwise it returns '1' """
    return np.sign(x)


def sign_derivative(x):
    x1 = []
    for k in x:
        if k[0] == 0:
            x1.append([1])
        else:
            x1.append([0])
    return x1


def saturated_linear(x):
    """ It returns '-1' if the input is less than -1 & returns '1' if the input is greater than 1 otherwise it
    returns the given input."""
    x1 = []
    for k in x:
        if k[0] < -1:
            x1.append([-1])
        elif k[0] > 1:
            x1.append([1])
        else:
            x1.append(k)

    return x1


def saturated_linear_derivative(x):
    x1 = []
    for k in x:
        if k[0] < -1:
            x1.append([0])
        elif k[0] > 1:
            x1.append([0])
        else:
            x1.append([1])

    return x1


def linear(x):
    """ y = f(x) It returns the input as it is"""
    return x


def linear_derivative(x):
    x1 = []
    for _ in x:
        x1.append([1])
    return x1


def tanh(x):
    """ It returns the value (1-exp(-2x))/(1+exp(-2x)) where the values lie between '-1' and '1'."""
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    """ It returns zero if the input is less than zero otherwise it returns the given input. """
    x1 = []
    for k in x:
        # print(i)
        if k[0] < 0:
            x1.append([0])
        else:
            x1.append(k)

    return x1


def relu_derivative(x):
    x1 = []
    for k in x:
        # print(i)
        if k[0] < 0:
            x1.append([0])
        else:
            x1.append([1])

    return x1


def softmax(x):
    """ Compute softmax values for each sets of scores in x. """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)


def feed_forward(n, w, b, af):
    for k in range(1):
        temp = w[k].dot(n[k]) + b[k]

        if af == 0:
            n[k + 1] = gaussian(temp)
        elif af == 1:
            n[k + 1] = sigmoid(temp)
        elif af == 2:
            n[k + 1] = step(temp)
        elif af == 3:
            n[k + 1] = sign(temp)
        elif af == 4:
            n[k + 1] = saturated_linear(temp)
        elif af == 5:
            n[k + 1] = linear(temp)
        elif af == 6:
            n[k + 1] = tanh(temp)
        elif af == 7:
            n[k + 1] = relu(temp)
        elif af == 8:
            n[k + 1] = softmax(temp)


# def back_propagate(n, w, b, af, label):
#     loss = label - n[-1]
#     if af == 0:
#         n[i + 1] = gaussian(tmp)


if __name__ == '__main__':
    """User must set these:   """
    epoch = 1
    learning_rate = 0.5
    layer_num = 3
    input_neuron_num = 8
    hiddenL_neuron_num = [4, 3]  # 8 4 3
    neurons = [np.random.randint(-10, 10, size=(input_neuron_num, 1))]
    output_label = [0, 1, 0]
    activation_function = 2

    for i in hiddenL_neuron_num:
        neurons.append(np.zeros(i).T)

    """Making random wights and biases"""
    wi = []
    bi = []
    for i in range(layer_num - 1):
        if i == 0:
            tmp = np.random.rand(hiddenL_neuron_num[0], input_neuron_num)
        else:
            tmp = np.random.rand(hiddenL_neuron_num[i], hiddenL_neuron_num[i - 1])
        wi.append(tmp)
        bias = np.random.rand(1, 1)
        bi.append(bias)
    # print(wi)
    # print(bi)

    for i in range(epoch):
        feed_forward(neurons, wi, bi, activation_function)
        # for i in neurons:
        #     print(i, "\n")
        # back_propagate(neurons, wi, bi, activation_function, output_label)

    # x = np.linspace(-10, 10)
    # plt.plot(x, gaussian(x))
    # plt.axis('tight')
    # plt.title('Activation Function :binaryStep')
    # plt.show()
