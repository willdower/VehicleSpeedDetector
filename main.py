import cupy as cp
from matplotlib import pyplot as plt
import cv2

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

if __name__ == "__main__":

    W_output, W_hidden, B_output, B_hidden = read_in_weights("weights.txt")