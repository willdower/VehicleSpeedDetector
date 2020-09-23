import numpy as np
from matplotlib import pyplot as plt
import cv2

def detectEdges(img):
    edges = cv2.Canny(img, 100, 200)
    plt.imshow(edges, cmap='gray')
    plt.show()

if __name__ == "__main__":
    img = cv2.imread("test.png")
    detectEdges(img)