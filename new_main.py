import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

img = cv2.imread('carside.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
original_gray = np.copy(gray)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img_keypoints = cv2.drawKeypoints(gray, kp, img)
plt.imshow(img_keypoints)
plt.show()

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

img2 = cv2.imread('bluecar1.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
original_gray2 = np.copy(gray2)
kp2, des2 = sift.detectAndCompute(gray2, None)


start_bf = time.perf_counter()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des, des2)

matches = sorted(matches, key = lambda x:x.distance)
end_bf = time.perf_counter()

img3 = img
img3 = cv2.drawMatches(original_gray, kp, original_gray2, kp2, matches[:10], flags=2, outImg=img3)

plt.imshow(img3)
plt.show()
print("Time: " + str(end_bf-start_bf))

new_bf = cv2.BFMatcher()
start_knn_bf = time.perf_counter()
new_matches = new_bf.knnMatch(des, des2, k=2)
good = []
for m, n in new_matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
end_knn_bf = time.perf_counter()

img4 = img
img4 = cv2.drawMatchesKnn(original_gray, kp, original_gray2, kp2, good, flags=2, outImg=img4)

plt.imshow(img4)
plt.show()
print("Time: " + str(end_knn_bf - start_knn_bf))

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

start_flann = time.perf_counter()
flann = cv2.FlannBasedMatcher(index_params, search_params)
flann_matches = flann.knnMatch(des, des2, k=2)
matches_mask = [[0, 0] for i in range(len(flann_matches))]


for i, (m, n) in enumerate(flann_matches):
    if m.distance < 0.7*n.distance:
        matches_mask[i]=[1, 0]
end_flann = time.perf_counter()

draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask=matches_mask, flags = 0)
img5 = img
img5 = cv2.drawMatchesKnn(original_gray, kp, original_gray2, kp2, flann_matches, None, **draw_params)
plt.imshow(img5)
plt.show()
print("Time: " + str(end_flann - start_flann))