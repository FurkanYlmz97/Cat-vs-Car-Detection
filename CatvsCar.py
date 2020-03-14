__author__ = "Furkan Yilmaz"

import h5py as h5
import numpy as np
from matplotlib import pyplot as plt


def hyperbolic_tangent(x):
    return np.tanh(x)


def derivative_of_hyperbolic_tangent(x):
    return 1 - np.tanh(x) * np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_of_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def TwoHiddenLayerTrain(neuron_number_1, neuron_number_2, batch_number, epoch_number, learning_rate, trainims,
                        trainlbls,
                        testims, testlbs, momentum_rate):
    mse_cum = []
    acc_cum = []

    test_mse_cum = []
    test_acc_cum = []

    # Initialize weights randomly small
    w_1 = np.random.normal(0, 0.01, (neuron_number_1, 32 * 32 + 1))
    w_2 = np.random.normal(0, 0.01, (neuron_number_1, 32 * 32 + 1))
    w_3 = np.random.normal(0, 0.01, neuron_number_2 + 1)

    # epochs starts
    for m in range(epoch_number):

        gradient_w_1 = np.zeros((neuron_number_1, 32 * 32 + 1))
        gradient_w_2 = np.zeros((neuron_number_2, neuron_number_1 + 1))
        gradient_w_3 = np.zeros(neuron_number_2 + 1)
        momentum_w_1 = np.zeros((neuron_number_1, 32 * 32 + 1))
        momentum_w_2 = np.zeros((neuron_number_2, neuron_number_1 + 1))
        momentum_w_3 = np.zeros(neuron_number_2 + 1)
        mse = []
        test_mse = []
        acc = []
        test_acc = []

        for i in range(1900):

            x = (trainims[i] / 255).flatten()  # input
            x = x - np.mean(x)
            x = x / np.std(x)
            x = np.append(x, 1)  # bias
            d = trainlbls[i]  # desired output
            if d == 0:  # adjust the desired suitable with the hyperbolic_tangent activation function
                d = -1

            # Hidden layer 1 equations
            o_hidden1 = hyperbolic_tangent(np.matmul(x, w_1.transpose()))
            o_hidden1 = np.append(o_hidden1, 1)  # bias
            f_hidden1 = np.diag(derivative_of_hyperbolic_tangent(np.matmul(x, w_1.transpose())))

            # Hidden layer 2 equations
            o_hidden2 = hyperbolic_tangent(np.matmul(o_hidden1, w_2.transpose()))
            o_hidden2 = np.append(o_hidden2, 1)  # bias
            f_hidden2 = np.diag(derivative_of_hyperbolic_tangent(np.matmul(o_hidden1, w_2.transpose())))

            # Output layer equations
            o = hyperbolic_tangent(np.matmul(o_hidden2, w_3))
            f = derivative_of_hyperbolic_tangent(np.matmul(o_hidden2, w_3))

            # Find training mse and acc for every step
            mse.append(1 / 2 * (d - o) * (d - o))
            if (o > 0 and d == 1) or (o < 0 and d == -1):
                acc.append(1.0)
            else:
                acc.append(0.0)

            # small deltas
            delta_o = np.dot(f, d - o)
            delta_h2 = np.matmul(f_hidden2, np.transpose(w_3[0:neuron_number_2])) * delta_o
            delta_h1 = np.matmul(np.matmul(f_hidden1, np.transpose(w_2[:, 0:neuron_number_1])), delta_h2)

            # Calculating gradient of the output layer's weights
            gradient_w_3 = gradient_w_3 + np.dot(delta_o, o_hidden2)

            # Calculating gradient of the hidden layer's weights
            gradient_w_2 = gradient_w_2 + np.outer(delta_h2, o_hidden1)
            gradient_w_1 = gradient_w_1 + np.outer(delta_h1, x)

            # Update the weights according to gradients and reinitialize gradients
            if (i + 1) % batch_number == 0:
                w_1 = w_1 + learning_rate * (gradient_w_1 / batch_number - momentum_rate * momentum_w_1)
                w_2 = w_2 + learning_rate * (gradient_w_2 / batch_number - momentum_rate * momentum_w_2)
                w_3 = w_3 + learning_rate * (gradient_w_3 / batch_number - momentum_rate * momentum_w_3)

                momentum_w_1 = gradient_w_1 / batch_number + momentum_rate * momentum_w_1
                momentum_w_2 = gradient_w_2 / batch_number + momentum_rate * momentum_w_2
                momentum_w_3 = gradient_w_3 / batch_number + momentum_rate * momentum_w_3
                gradient_w_1 = np.zeros((neuron_number_1, 32 * 32 + 1))
                gradient_w_2 = np.zeros((neuron_number_2, neuron_number_1 + 1))
                gradient_w_3 = np.zeros(neuron_number_2 + 1)

        for i in range(1000):
            # Test mse and accuracy will be found
            x = (testims[i] / 255).flatten()  # input
            x = x - np.mean(x)
            x = x / np.std(x)
            x = np.append(x, 1)  # bias
            d = testlbs[i]  # desired output
            if d == 0:  # adjust the desired suitable with the hyperbolic_tangent activation function
                d = -1

            # Hidden layer 1 equations
            o_hidden1 = hyperbolic_tangent(np.matmul(x, w_1.transpose()))
            o_hidden1 = np.append(o_hidden1, 1)  # bias

            # Hidden layer 2 equations
            o_hidden2 = hyperbolic_tangent(np.matmul(o_hidden1, w_2.transpose()))
            o_hidden2 = np.append(o_hidden2, 1)  # bias

            # Output layer equations
            o = hyperbolic_tangent(np.matmul(o_hidden2, w_3))

            # Find training mse and acc for every step
            test_mse.append((1 / 2) * (d - o) * (d - o))
            if (o > 0 and d == 1) or (o < 0 and d == -1):
                test_acc.append(1.0)
            else:
                test_acc.append(0.0)

        # Calculate total accuracy and mse
        mse_cum.append(np.mean(mse))
        acc_cum.append((1 - np.mean(acc)) * 100)
        test_mse_cum.append(np.mean(test_mse))
        test_acc_cum.append((1 - np.mean(test_acc)) * 100)

    # Plot the error and the accuracy respect to epoch
    plt.plot(mse_cum, label="Training MSE")
    plt.plot(test_mse_cum, label="Test MSE")
    plt.title("MSE Error with 2 Hidden Layer NN")
    plt.xlabel("Epoch Number")
    plt.ylabel("MSE Error")
    plt.legend()
    plt.show()

    plt.plot(acc_cum, label="Training Accuracy Error")
    plt.plot(test_acc_cum, label="Test Accuracy Error")
    plt.title("Accuracy Error with 2 Hidden Layer NN")
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy Error %")
    plt.legend()
    plt.show()


def OneHiddenLayerTrain(neuron_number, batch_number, epoch_number, learning_rate, trainims, trainlbls, testims,
                        testlbs):
    mse_cum = []
    acc_cum = []

    test_mse_cum = []
    test_acc_cum = []

    # Initialize weights randomly small
    w_1 = np.random.normal(0, 0.01, (neuron_number, 32 * 32 + 1))
    w_2 = np.random.normal(0, 0.01, (neuron_number + 1))

    # epochs starts
    for m in range(epoch_number):
        gradient_w_1 = np.zeros((neuron_number, 32 * 32 + 1))
        gradient_w_2 = np.zeros(neuron_number + 1)
        mse = []
        acc = []
        test_mse = []
        test_acc = []

        for i in range(1900):

            x = ((trainims[i] / 255)).flatten()  # input
            x = x - np.mean(x)
            x = x / np.std(x)
            x = np.append(x, -1)  # bias
            d = trainlbls[i]  # desired output
            if d == 0:  # adjust the desired suitable with the hyperbolic_tangent activation function
                d = -1

            # Hidden layer equations
            o_hidden = hyperbolic_tangent(np.matmul(x, w_1.transpose()))
            o_hidden = np.append(o_hidden, -1)  # bias
            f_hidden = np.diag(derivative_of_hyperbolic_tangent(np.matmul(x, w_1.transpose())))

            # Output layer equations
            o = hyperbolic_tangent(np.matmul(o_hidden, w_2.transpose()))
            f = derivative_of_hyperbolic_tangent(np.matmul(o_hidden, w_2.transpose()))

            # Find training mse and acc for every step
            mse.append(1 / 2 * (d - o) * (d - o))
            if (o > 0 and d == 1) or (o < 0 and d == -1):
                acc.append(1.0)
            else:
                acc.append(0.0)

            # Calculating gradient of the output layer's weights
            delta_o = f * (d - o)
            gradient_w_2 = gradient_w_2 + delta_o * o_hidden

            # Calculating gradient of the hidden layer's weights
            delta_1 = np.matmul(f_hidden, np.transpose(w_2[0:neuron_number])) * delta_o
            gradient_w_1 = gradient_w_1 + learning_rate * np.outer(delta_1, x)

            # Update the weights according to gradients and reinitialize gradients
            if (i + 1) % batch_number == 0:
                w_1 = w_1 + gradient_w_1 / batch_number
                w_2 = w_2 + gradient_w_2 / batch_number
                gradient_w_2 = np.zeros(neuron_number + 1)
                gradient_w_1 = np.zeros((neuron_number, 32 * 32 + 1))

        for i in range(1000):
            # Test mse and accuracy will be found
            x = (testims[i] / 255).flatten()  # input
            x = x - np.mean(x)
            x = x / np.std(x)
            x = np.append(x, -1)  # bias
            d = testlbs[i]  # desired output
            if d == 0:  # adjust the desired suitable with the hyperbolic_tangent activation function
                d = -1

            # Hidden layer equations
            o_hidden = hyperbolic_tangent(np.matmul(x, w_1.transpose()))
            o_hidden = np.append(o_hidden, -1)  # bias

            # Output layer equations
            o = hyperbolic_tangent(np.matmul(o_hidden, w_2.transpose()))

            # Find test mse and acc for every step
            test_mse.append(1 / 2 * (d - o) * (d - o))
            if (o > 0 and d == 1) or (o < 0 and d == -1):
                test_acc.append(1.0)
            else:
                test_acc.append(0.0)

        # Calculate total accuracy and mse
        mse_cum.append(np.mean(mse))
        acc_cum.append((1 - np.mean(acc)) * 100)
        test_mse_cum.append(np.mean(test_mse))
        test_acc_cum.append((1 - np.mean(test_acc)) * 100)

    # Plot the error and the accuracy respect to epoch
    plt.plot(mse_cum, label="Training MSE")
    plt.plot(test_mse_cum, label="Test MSE")
    plt.title("MSE Error with 1 Hidden Layer NN")
    plt.xlabel("Epoch Number")
    plt.ylabel("MSE Error")
    plt.legend()
    plt.show()

    plt.plot(acc_cum, label="Training Accuracy")
    plt.plot(test_acc_cum, label="Test Accuracy")
    plt.title("Accuracy Error with 1 Hidden Layer NN")
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy Error %")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    file = h5.File("assign2_data1.h5", "r")
    testims = np.array(file["testims"])  # shape -> 1000 - 32 - 32
    testlbls = np.array(file["testlbls"])  # shape -> 1000 1-> Car 0-> Cat
    trainims = np.array(file["trainims"])  # shape -> 1900 - 32 - 32
    trainlbls = np.array(file["trainlbls"])  # shape -> 1900 1-> Car 0-> Cat

    # (neuron_number, batch_number, epoch_number, learning_rate, trainims, trainlbls, testims, testlbs, momentum rate):
    OneHiddenLayerTrain(32, 20, 1000, 0.2, trainims, trainlbls, testims, testlbls)
    TwoHiddenLayerTrain(32, 32, 20, 500, 0.2, trainims, trainlbls, testims, testlbls, 0.2)
    plt.show()
