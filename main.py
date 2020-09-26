import os
import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures

from sklearn import svm
import sklearn

def read_in_images_new(directory):
    vehicle_dir = directory + "/vehicle"
    non_dir = directory + "/non"

    win_size = 64
    win_size_tuple = (win_size, win_size)
    cell_size = 8
    cell_size_tuple = (cell_size, cell_size)
    block_size = (cell_size*2, cell_size*2)
    block_stride = (cell_size, cell_size)
    nbins = 9
    feature_size = int(9 * (4 + ((((win_size/cell_size)-2)*4)*2) + ((((win_size/cell_size)-2) * ((win_size/cell_size)-2))*4)))

    hog = cv2.HOGDescriptor(win_size_tuple, block_size, block_stride, cell_size_tuple, nbins)

    sum = 0
    for filename in os.listdir(vehicle_dir):
        sum += 1
    for filename in os.listdir(non_dir):
        sum += 1

    x_array = np.zeros((sum, feature_size))
    y_array = np.zeros((sum, 2))

    sum = 0
    for filename in os.listdir(vehicle_dir):
        full_path = vehicle_dir + "/" + filename
        img = cv2.imread(full_path, 0)
        out = hog.compute(img)
        out = np.transpose(out)
        out = np.array(out)
        x_array[sum] = out
        y_array[sum] = np.array([0, 1])
        sum += 1
        if sum % 100 == 0:
            print("Loaded " + str(sum) + " images")

    for filename in os.listdir(non_dir):
        full_path = non_dir + "/" + filename
        img = cv2.imread(full_path, 0)
        out = hog.compute(img)
        out = np.transpose(out)
        out = np.array(out)
        x_array[sum] = out
        y_array[sum] = np.array([1, 0])
        sum += 1
        if sum % 100 == 0:
            print("Loaded " + str(sum) + " images")

    return feature_size, x_array, y_array

def detectEdges(img):
    edges = cv2.Canny(img, 100, 200)
    plt.imshow(edges, cmap='gray')
    plt.show()

def read_in_weights(filename):
    file = open(filename, 'r')
    string = file.readline()
    file_list = string.split(sep=" ")
    output_W_size = (int(file_list[0]), int(file_list[1]))
    W_output = cp.zeros(output_W_size)
    string = file.readline()
    file_list = string.split(sep=" ")
    for x in range(0, W_output.shape[0]):
        for y in range(0, W_output.shape[1]):
            W_output[x][y] = float(file_list[x * W_output.shape[1] + y])

    string = file.readline()
    file_list = string.split(sep=" ")
    hidden_W_size = (int(file_list[0]), int(file_list[1]))
    W_hidden = cp.zeros(hidden_W_size)
    string = file.readline()
    file_list = string.split(sep=" ")
    for x in range(0, W_hidden.shape[0]):
        for y in range(0, W_hidden.shape[1]):
            W_hidden[x][y] = float(file_list[x * W_output.shape[1] + y])

    string = file.readline()
    file_list = string.split(sep=" ")
    output_B_size = (int(file_list[0]), int(file_list[1]))
    B_output = cp.zeros(output_B_size)
    string = file.readline()
    file_list = string.split(sep=" ")
    for x in range(0, B_output.shape[0]):
        for y in range(0, B_output.shape[1]):
            B_output[x][y] = float(file_list[x * W_output.shape[1] + y])

    string = file.readline()
    file_list = string.split(sep=" ")
    hidden_B_size = (int(file_list[0]), int(file_list[1]))
    B_hidden = cp.zeros(hidden_B_size)
    string = file.readline()
    file_list = string.split(sep=" ")
    for x in range(0, B_hidden.shape[0]):
        for y in range(0, B_hidden.shape[1]):
            B_hidden[x][y] = float(file_list[x * W_output.shape[1] + y])

    file.close()
    return W_output, W_hidden, B_output, B_hidden

def sigmoid(input):
    return 1 / (1 + cp.exp(-input))

def make_prediction(img, weights):

    win_size = 64
    win_size_tuple = (win_size, win_size)
    cell_size = 8
    cell_size_tuple = (cell_size, cell_size)
    block_size = (cell_size * 2, cell_size * 2)
    block_stride = (cell_size, cell_size)
    nbins = 9
    feature_size = int(9 * (4 + ((((win_size / cell_size) - 2) * 4) * 2) + (
                (((win_size / cell_size) - 2) * ((win_size / cell_size) - 2)) * 4)))

    img = cv2.resize(img, win_size_tuple)

    hog = cv2.HOGDescriptor(win_size_tuple, block_size, block_stride, cell_size_tuple, nbins)
    out = hog.compute(img)
    out = cp.transpose(out)
    out = cp.array(out)

    net_hidden = cp.dot(out, weights[1]) + weights[3]
    out_hidden = sigmoid(net_hidden)

    net_output = cp.dot(out_hidden, weights[0]) + weights[2]
    out_output = sigmoid(net_output)

    prediction = cp.argmax(out_output)


    return prediction

def make_prediction_new(img, model):

    win_size = 64
    win_size_tuple = (win_size, win_size)
    cell_size = 8
    cell_size_tuple = (cell_size, cell_size)
    block_size = (cell_size * 2, cell_size * 2)
    block_stride = (cell_size, cell_size)
    nbins = 9
    feature_size = int(9 * (4 + ((((win_size / cell_size) - 2) * 4) * 2) + (
                (((win_size / cell_size) - 2) * ((win_size / cell_size) - 2)) * 4)))

    img = cv2.resize(img, win_size_tuple)

    hog = cv2.HOGDescriptor(win_size_tuple, block_size, block_stride, cell_size_tuple, nbins)
    out = hog.compute(img)
    out = np.transpose(out)
    out = np.array(out)

    prediction = model.predict(out)
    #prediction = np.argmax(prediction)

    return prediction

def detect_car_in_image(img, weights):
    window_size = (200, 200)
    step = 15

    draw_on = img.copy()
    draw_on = cv2.cvtColor(draw_on, cv2.COLOR_GRAY2BGR)

    window = np.zeros(window_size, dtype=np.uint8)
    for x in range(0, img.shape[0]-window_size[0], step):
        for y in range(0, img.shape[1]-window_size[1], step):
            window = np.zeros(window_size, dtype=np.uint8)
            for i in range(0, window_size[0]):
                for j in range(0, window_size[1]):
                    window[i][j] = img[x+i][y+j]
            if make_prediction(window, weights) == 1:
                cv2.rectangle(draw_on, (y, x), (y+window_size[1], x+window_size[0]), color=(255, 0, 0))

    plt.imshow(draw_on)
    plt.show()

def detect_car_in_image_new(img, model):
    window_size = (100, 100)
    step = 25

    draw_on = img.copy()
    draw_on = cv2.cvtColor(draw_on, cv2.COLOR_GRAY2BGR)

    window = np.zeros(window_size, dtype=np.uint8)
    for x in range(0, img.shape[0]-window_size[0], step):
        for y in range(0, img.shape[1]-window_size[1], step):
            window = np.zeros(window_size, dtype=np.uint8)
            for i in range(0, window_size[0]):
                for j in range(0, window_size[1]):
                    window[i][j] = img[x+i][y+j]
            if make_prediction_new(window, model) == 1:
                cv2.rectangle(draw_on, (y, x), (y+window_size[1], x+window_size[0]), color=(255, 0, 0))

    plt.imshow(draw_on)
    plt.show()

if __name__ == "__main__":

    # W_output, W_hidden, B_output, B_hidden = read_in_weights("weights.txt")
    # weights = [W_output, W_hidden, B_output, B_hidden]

    test_image = cv2.imread("dashcam.jpg", 0)
    # detect_car_in_image(test_image, weights)

    feature_size, x_train, y_train = read_in_images_new("imgs")
    feature_size_test, x_test, y_test = read_in_images_new("test_imgs")

    """inputs = keras.Input(shape=(feature_size,), name="imgs")
    x = layers.Dense(32, activation="relu", name="dense_1")(inputs)
    outputs = layers.Dense(2, activation="sigmoid", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)"""

    x_train, y_train = sklearn.utils.shuffle(x_train, y_train)

    clf = svm.SVC()

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    clf.fit(x_train, y_train)

    x_val = x_train[-1000:]
    y_val = y_train[-1000:]
    x_train = x_train[:-1000]
    y_train = y_train[:-1000]

    """model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=2,
        validation_data=(x_val, y_val),
    )

    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)"""

    detect_car_in_image_new(test_image, clf)