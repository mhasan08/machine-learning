import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

import scipy.misc

from data_import import import_data

print("****logistic_regression****")

# import data
train_set_x, train_set_y, test_set_x, test_set_y, classes = import_data()

# number of samples
train_samples = train_set_x[0]
test_samples = train_set_y[0]

# flatten, normalize => make feature vector
feature_vector_x_train = train_set_x.reshape(train_set_x.shape[0], -1).T/255.
feature_vector_x_test = test_set_x.reshape(test_set_x.shape[0], -1).T/255.


def activation_function(x):
    return 1 / (1 + np.exp(-x))


def for_back_propagation(X, Y, w, b):
    m = X.shape[1]
    A = activation_function(np.dot(w.T, X) + b)
    cost = -1 / m * (np.sum((Y * np.log(A)) + (1 - Y) * np.log(1 - A)))

    dw = 1 / m * (np.dot(X, (A - Y).T))
    db = 1 / m * (np.sum(A - Y))

    cost = np.squeeze(cost)

    grads = {"dw": dw, "db": db}

    return grads, cost


def compute(X, Y, w, b, num_iterations, learning_rate, print_cost):
    costs = []

    for i in range(num_iterations):
        grads, cost = for_back_propagation(X, Y, w, b)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 50 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}

    return params, costs


def predict(X, w, b):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = activation_function(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[0, i] <= 0.60:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
        pass

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, w, b, print_cost):

    parameters, costs = compute(X_train, Y_train, w, b, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(X_test, w, b)
    Y_prediction_train = predict(X_train, w, b)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    trained_parameters = {"costs": costs, "Y_prediction_test": Y_prediction_test,
                          "Y_prediction_train": Y_prediction_train, "w": w,
                          "b": b, "learning_rate": learning_rate, "num_iterations": num_iterations}

    return trained_parameters


def logistic_regression():

    index = random.randint(1, 242)
    plt.imshow(train_set_x[index])
    print("Y: " + str(train_set_y[:, index]) + ", Random Sample: '" + classes[np.squeeze(train_set_y[:, index])]+"' image")
    plt.show()

    number_of_channels = 3
    number_of_h_pixels = 128
    number_of_w_pixels = 128

    # initialize parameters
    w = np.zeros([number_of_w_pixels * number_of_h_pixels * number_of_channels, 1])
    b = 0

    # hyper-parameters
    num_iterations = 10000
    learning_rate = 0.001

    learned_parameters = model(feature_vector_x_train, train_set_y, feature_vector_x_test, test_set_y, num_iterations, learning_rate,
                               w, b, print_cost=True)

    costs = np.squeeze(learned_parameters['costs'])
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (steps of 50)')
    plt.title("Learning rate =" + str(learned_parameters["learning_rate"]))
    plt.show()

    # prediction
    image_to_be_predicted = "cat_in_iran.jpg"
    #image_to_be_predicted = "polar_bear.jpg"

    image_name = "images/" + image_to_be_predicted
    image = np.array(plt.imread(image_name))

    plt.imshow(image)
    plt.show()

    reshaped_image = scipy.misc.imresize(image, size=(number_of_w_pixels, number_of_h_pixels)).reshape((1, number_of_w_pixels * number_of_h_pixels * number_of_channels)).T

    predicted_image = predict(reshaped_image, learned_parameters["w"], learned_parameters["b"])

    print("Y: [" + str(int(np.squeeze(predicted_image))) + "], Prediction: \"" + classes[
        int(np.squeeze(predicted_image)),] + "\" image")


if __name__ == "__main__":
    logistic_regression()