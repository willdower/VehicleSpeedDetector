import cv2
import cupy as cp
import os
import random


def read_in_images(directory):
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
    list = []

    sum = 0
    for filename in os.listdir(vehicle_dir):
        full_path = vehicle_dir + "/" + filename
        img = cv2.imread(full_path, 0)
        out = hog.compute(img)
        out = cp.transpose(out)
        out = cp.array(out)
        list.append((out, cp.array((0, 1))))
        sum += 1
        if sum % 100 == 0:
            print("Loaded " + str(sum) + " images")

    for filename in os.listdir(non_dir):
        full_path = non_dir + "/" + filename
        img = cv2.imread(full_path, 0)
        out = hog.compute(img)
        out = cp.transpose(out)
        out = cp.array(out)
        list.append((out, cp.array((1, 0))))
        sum += 1
        if sum % 100 == 0:
            print("Loaded " + str(sum) + " images")

    return feature_size, list


def sigmoid(input):
    return 1 / (1 + cp.exp(-input))


def randomly_initialize_w(n_input_neurons, n_hidden_neurons, n_output_neurons):
    W_1 = cp.random.normal(loc=0, scale=1, size=(n_input_neurons, n_hidden_neurons))
    W_2 = cp.random.normal(loc=0, scale=1, size=(n_hidden_neurons, n_output_neurons))
    B_hidden = cp.random.normal(loc=0, scale=1, size=(1, n_hidden_neurons))
    B_output = cp.random.normal(loc=0, scale=1, size=(1, n_output_neurons))
    return W_1, W_2, B_hidden, B_output


def forward_pass(W_output, W_hidden, B_hidden, B_output, x):
    net_hidden = cp.dot(x, W_hidden) + B_hidden
    out_hidden = sigmoid(net_hidden)

    net_output = cp.dot(out_hidden, W_output) + B_output
    out_output = sigmoid(net_output)

    return out_hidden, out_output


def backward_pass(x, y, output, hidden_output, W_output):
    output_error = -(y - output)  # Calculate error
    output_over_net = output*(1 - output)  # Derivative of sigmoid function
    sigmoid_on_error = cp.multiply(output_error, output_over_net)  # Calculate the sigmoid function's affect on error

    W_output = cp.transpose(W_output)
    hidden_error = cp.dot(sigmoid_on_error, W_output)  # Calculate the affect of output weights on hidden weights' error
    hidden_over_net = hidden_output*(1 - hidden_output)  # Derivative of sigmoid function
    sigmoid_on_hidden_error = cp.multiply(hidden_error, hidden_over_net)  # Calculate the sigmoid function's affect on error

    # Correctly arrange matrices for calculations
    x = cp.atleast_2d(x)
    hidden_output = cp.atleast_2d(hidden_output)
    x_transpose = cp.transpose(x)
    hidden_output_transpose = cp.transpose(hidden_output)
    sigmoid_on_hidden_error = sigmoid_on_hidden_error.reshape(1, sigmoid_on_hidden_error.size)
    sigmoid_on_error = sigmoid_on_error.reshape(1, sigmoid_on_error.size)

    # Calculate weight changes
    W_hidden_c = cp.dot(x_transpose, sigmoid_on_hidden_error)
    W_output_c = cp.dot(hidden_output_transpose, sigmoid_on_error)

    # Calculate bias changes
    B_hidden_c = sigmoid_on_hidden_error
    B_output_c = sigmoid_on_error

    return W_output_c, W_hidden_c, B_hidden_c, B_output_c


def predict_if_car(W_output, W_hidden, B_hidden, B_output, x):
    hidden, out = forward_pass(W_output, W_hidden, B_hidden, B_output, x)
    prediction = cp.argmax(out)
    return prediction


if __name__ == "__main__":
    feature_size, images = read_in_images("imgs")
    random.shuffle(images)

    n_input_neurons = feature_size
    n_hidden_neurons = 64
    n_output_neurons = 2
    learning_rate = 1



    W_hidden, W_output, B_hidden, B_output = randomly_initialize_w(n_input_neurons, n_hidden_neurons, n_output_neurons)

    sum = 0
    total = len(images)

    for tuple in images:
        x = tuple[0]
        y = tuple[1]
        out_hidden, out_output = forward_pass(W_output, W_hidden, B_hidden, B_output, x)
        W_output_c, W_hidden_c, B_hidden_c, B_output_c = backward_pass(x, y, out_output, out_hidden, W_output)

        W_output = W_output - (learning_rate * W_output_c)
        W_hidden = W_hidden - (learning_rate * W_hidden_c)
        B_hidden = B_hidden - (learning_rate * B_hidden_c)
        B_output = B_output - (learning_rate * B_output_c)

        sum += 1
        if sum % 100 == 0:
            print("Trained on " + str(sum) + " images, " + str(total-sum) + " remaining")

    print("\nBeginning testing...")
    random.shuffle(images)
    correct = 0
    incorrect = 0
    for tuple in images:
        x = tuple[0]
        y = tuple[1]
        out_hidden, out_output = forward_pass(W_output, W_hidden, B_hidden, B_output, x)

        prediction = cp.argmax(out_output)

        if prediction == 1:
            # Predicted vehicle
            comparison = y == cp.array((0, 1))
            if comparison.all():
                correct += 1
            else:
                incorrect += 1
        else:
            # Predicted non-vehicle
            comparison = y == cp.array((1, 0))
            if comparison.all():
                correct += 1
            else:
                incorrect += 1

    print("\n")
    print(str(correct) + " correctly identified")
    print(str(incorrect) + " incorrectly identified")
    percentage = (correct/(correct+incorrect))*100
    print(str(percentage) + "% accuracy")

    file = open("weights.txt", 'w')
    string = str(W_output.shape[0]) + " " + str(W_output.shape[1]) + "\n"
    file.write(string)
    for x in range(0, W_output.shape[0]):
        for y in range(0, W_output.shape[1]):
            string = str(W_output[x][y]) + " "
            file.write(string)
    file.write("\n")

    string = str(W_hidden.shape[0]) + " " + str(W_hidden.shape[1]) + "\n"
    file.write(string)
    for x in range(0, W_hidden.shape[0]):
        for y in range(0, W_hidden.shape[1]):
            string = str(W_hidden[x][y]) + " "
            file.write(string)
    file.write("\n")

    string = str(B_output.shape[0]) + " " + str(B_output.shape[1]) + "\n"
    file.write(string)
    for x in range(0, B_output.shape[0]):
        for y in range(0, B_output.shape[1]):
            string = str(B_output[x][y]) + " "
            file.write(string)
    file.write("\n")

    string = str(B_hidden.shape[0]) + " " + str(B_hidden.shape[1]) + "\n"
    file.write(string)
    for x in range(0, B_hidden.shape[0]):
        for y in range(0, B_hidden.shape[1]):
            string = str(B_hidden[x][y]) + " "
            file.write(string)
    file.write("\n")

    file.close()