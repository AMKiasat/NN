import numpy as np
import math
import pickle
import matplotlib.pyplot as plt


def gaussian(x):
    """ It returns exp(-x^2) where the values lie between '0' and '1' """
    return np.exp(-1 * np.square(x))


def gaussian_derivative(x):
    return -2 * x * np.exp(-1 * np.square(x))


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
    for k in range(len(n) - 1):
        temp = w[k].dot(n[k]) + b[k]

        if af == 0:
            neuro = gaussian(temp)
        elif af == 1:
            neuro = sigmoid(temp)
        elif af == 2:
            neuro = step(temp)
        elif af == 3:
            neuro = sign(temp)
        elif af == 4:
            neuro = saturated_linear(temp)
        elif af == 5:
            neuro = linear(temp)
        elif af == 6:
            neuro = tanh(temp)
        elif af == 7:
            neuro = relu(temp)
        elif af == 8:
            neuro = softmax(temp)
        if math.isnan(neuro[0][0]):
            return
        n[k + 1] = neuro


def back_propagate(n, w, b, af, label, lr):
    loss = label - n[-1]
    # loss = loss.T
    # print(label, "\n")
    # print(n[-1], "\n")
    # print(loss, "\n")
    if af == 0:
        delta = loss * gaussian_derivative(n[-1])
    elif af == 1:
        delta = loss * sigmoid_derivative(n[-1])
    elif af == 2:
        delta = loss * step_derivative(n[-1])
    elif af == 3:
        delta = loss * sign_derivative(n[-1])
    elif af == 4:
        delta = loss * saturated_linear_derivative(n[-1])
    elif af == 5:
        delta = loss * linear_derivative(n[-1])
    elif af == 6:
        delta = loss * tanh_derivative(n[-1])
    elif af == 7:
        delta = loss * relu_derivative(n[-1])
    elif af == 8:
        delta = loss * softmax_derivative(n[-1])

    next_w = w[-1].copy()
    # print(b[-1], '\n')
    b[-1] += np.sum(delta, axis=0, keepdims=True) * lr
    # print(delta.flatten())
    # print(b[-1], "\n")
    for k in range(len(w[-1].T)):
        # print(w[-1].T[k])
        w[-1].T[k] += (lr * delta * n[-2][k]).flatten()
        # print(n[-2][k])
        # print((delta * n[-2][k]).flatten())
        # print((lr * delta * n[-2][k]).flatten())
        # print(next_w.T[k])
        # print(w[-1].T[k], "\n")

    for k in reversed(range(len(w) - 1)):
        loss = next_w.T.dot(delta)
        # loss = loss.T
        # print(label, "\n")
        # print(n[-1], "\n")
        # print(loss, "\n")
        if af == 0:
            delta = loss * gaussian_derivative(n[k + 1])
        elif af == 1:
            delta = loss * sigmoid_derivative(n[k + 1])
        elif af == 2:
            delta = loss * step_derivative(n[k + 1])
        elif af == 3:
            delta = loss * sign_derivative(n[k + 1])
        elif af == 4:
            delta = loss * saturated_linear_derivative(n[k + 1])
        elif af == 5:
            delta = loss * linear_derivative(n[k + 1])
        elif af == 6:
            delta = loss * tanh_derivative(n[k + 1])
        elif af == 7:
            delta = loss * relu_derivative(n[k + 1])
        elif af == 8:
            delta = loss * softmax_derivative(n[k + 1])
        next_w = w[k].copy()
        # print(b, '\n')
        b[k] += np.sum(delta, axis=0, keepdims=True) * lr
        # print(delta.flatten())
        # print(b[k], "\n")
        for j in range(len(w[k].T)):
            # print(w[k].T[j])
            w[k].T[j] += (lr * delta * n[k][j]).flatten()
            # print(n[k][j])
            # print((delta * n[k][j]).flatten())
            # print((lr * delta * n[k][j]).flatten())
            # print(next_w.T[j])
            # print(w[k].T[j], "\n")


def train(data, label, layer_num, hiddenL_neuron_num, activation_function=1, epoch=20, learning_rate=0.1):
    input_neuron_num = data.shape[1]
    output_neuron_num = label.max() + 1
    label01 = []

    for i in label:
        tmp = []
        for j in range(output_neuron_num):
            if i == j:
                tmp.append([1])
            else:
                tmp.append([0])
        label01.append(tmp)

    """ Making random wights and biases """
    wi = []
    bi = []
    for i in range(layer_num - 1):
        if i == 0:
            tmp = np.random.rand(hiddenL_neuron_num[0], input_neuron_num)
        elif i == layer_num - 2:
            tmp = np.random.rand(output_neuron_num, hiddenL_neuron_num[i - 1])
        else:
            tmp = np.random.rand(hiddenL_neuron_num[i], hiddenL_neuron_num[i - 1])
        wi.append(tmp)
        bias = np.random.rand(1, 1)
        bi.append(bias)

    """ Training with epoch """
    for i in range(epoch):
        for j in range(len(data)):
            tmp = np.array(data[j])
            neurons = [tmp[:, np.newaxis]]
            for k in hiddenL_neuron_num:
                neurons.append(np.zeros(k).T)
            neurons.append(np.zeros(output_neuron_num).T)
            feed_forward(neurons, wi, bi, activation_function)
            back_propagate(neurons, wi, bi, activation_function, label01[j], learning_rate)

    with open('wights.pkl', 'wb') as file:
        pickle.dump(len(wi), file)
        for array in wi:
            np.save(file, array)
    with open('biases.pkl', 'wb') as file:
        pickle.dump(len(bi), file)
        for array in bi:
            np.save(file, array)
    with open('activation_function.txt', 'w') as file:
        file.write(str(activation_function))


def test(data):
    with open('wights.pkl', 'rb') as file:
        wights = pickle.load(file)
        wi = [np.load(file) for _ in range(wights)]
    with open('biases.pkl', 'rb') as file:
        biases = pickle.load(file)
        bi = [np.load(file) for _ in range(biases)]
    with open('activation_function.txt', 'r') as file:
        activation_function = int(file.read())

    output_label = []
    for j in range(len(data)):
        tmp = np.array(data[j])
        neurons = [tmp[:, np.newaxis]]
        for k in wi:
            # print(k.shape)
            neurons.append(np.zeros(k.shape[0]).T)
        feed_forward(neurons, wi, bi, activation_function)
        output_label.append(np.argmax(neurons[-1]))
    return output_label
