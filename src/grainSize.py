import cv2
import numpy as np
from skimage.color import label2rgb

def filterByArea(image, area):
    cc = cv2.connectedComponentsWithStats(edges, connectivity=8, stats=cv2.CC_STAT_AREA)
    badIdx = np.where(cc[2][:, 4] < area)
    for idx in badIdx[0]:
        edges[np.where(cc[1]==idx)] = 0

def connectLines(img, numIter, kernelSize):
    # for i in range(numIter):
    kernel = np.ones((kernelSize, kernelSize))
    img = cv2.dilate(img, kernel, iterations=numIter)
    img = cv2.erode(img, kernel, iterations=numIter)
    return img

in_between = cv2.imread('generation.png')
gau_blurred = cv2.GaussianBlur(in_between, (3, 3), 1.0)
sharpened = cv2.addWeighted(in_between, 1.5, gau_blurred, -0.5, 0)
Z = in_between.reshape((-1,3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((in_between.shape))
cv2.imshow('res2',res2)
cv2.waitKey(0)

slic = cv2.ximgproc.createSuperpixelLSC(sharpened, region_size=25, ratio = 0.015)
slic.iterate(50)
slic.enforceLabelConnectivity(5)
mask_slic = slic.getLabelContourMask()
label_slic = slic.getLabels()
number_slic = slic.getNumberOfSuperpixels()
mask_inv_slic = cv2.bitwise_not(mask_slic)
img_slic = cv2.bitwise_and(sharpened, sharpened, mask =  mask_inv_slic)
cv2.imshow("img_slic",img_slic)
cv2.waitKey(0)



slickLabel = slic.getLabels()
numPix = slic.getNumberOfSuperpixels()
meanImg = np.copy(sharpened)
for i in range(numPix):
    meanImg[slickLabel==i] = np.mean(meanImg[slickLabel==i], axis=0)
cv2.imshow("mean",meanImg)
cv2.imwrite("mean.png", meanImg)
cv2.waitKey(0)




blured = cv2.bilateralFilter(sharpened, 7, 50, 50)
# blured = cv2.GaussianBlur(in_between, (5, 5), 1.5)
edges = cv2.Canny(blured, 20, 10, apertureSize=3)
edgesLoG = cv2.GaussianBlur(sharpened,(3,3),0) 
edgesLoG = cv2.Laplacian(edgesLoG, cv2.CV_64F)

edgesLoG = np.abs(edgesLoG)
# edgesLoG = edgesLoG/edgesLoG.max()
edgesLoG = np.max(edgesLoG, axis=2)
_, edgesLoG = cv2.threshold(edgesLoG, 8, 255, cv2.THRESH_BINARY)

dialated = cv2.dilate(edges, np.ones((3, 3)))
filterByArea(dialated, 4)

connected = connectLines(edges, 1, 3)
inverted = cv2.bitwise_not(connected)
# inverted = cv2.bitwise_not(edges)

labels = cv2.connectedComponents(inverted, connectivity=4)

colors = []
colors.append([0, 0, 0])
import random

for i in range(labels[0] - 1):
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    rgb = [r,g,b]
    colors.append(rgb)
pcolor = label2rgb(labels[1], colors = colors)


# params = cv2.SimpleBlobDetector_Params()
# params.minThreshold = 0
# params.maxThreshold = 256
# params.filterByArea = True
# params.minArea = 30
# params.filterByCircularity = False
# params.minCircularity = 0.1
# params.filterByConvexity = False
# params.minConvexity = 0.5
# params.filterByInertia =False
# params.minInertiaRatio = 0.5

# detector = cv2.SimpleBlobDetector_create(params)
#keypoints = detector.detect(inverted)

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#im_with_keypoints = cv2.drawKeypoints(in_between, keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow('sharpened', sharpened)
cv2.imshow('edges', edges)
cv2.imshow('LoG',edgesLoG)
cv2.imshow('dialated', connected)
cv2.imshow('inverted', inverted)
cv2.imshow('graines', pcolor.astype(np.uint8))

cv2.waitKey(0)
