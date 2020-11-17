import cv2
import numpy as np
from skimage.color import label2rgb
import random
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
import argparse
import os
def readImages(filenames):
    imgs = []
    for file in filenames:
        img = cv2.imread(file)
        print(img.shape)
        imgs.append(img)
    return np.stack(imgs, axis=0)
def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel

def filterByArea(img, area):
    cc = cv2.connectedComponentsWithStats(img, connectivity=8, stats=cv2.CC_STAT_AREA)
    badIdx = np.where(cc[2][:, 4] < area)
    for idx in badIdx[0]:
        img[np.where(cc[1]==idx)] = 0

def deBranch(img):
    __, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    newContours = []
    for i in range(len(contours)):
        cont = np.zeros(img.shape, dtype=np.uint8)
        contFilled = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours(cont, contours, i, 255, 1)
        cv2.drawContours(contFilled, contours, i, 255, -1)
        cont = cv2.bitwise_or(cont, contFilled)
        # cv2.imshow("Contour", cont)
        __, smallContour, __ = cv2.findContours(cont, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        newContours.extend(smallContour)
    result = np.zeros(img.shape, dtype=np.uint8)
    cv2.drawContours(result, newContours, -1, 255, 1)
    return result
    # cv2.imshow("Debranched", vis)

"""
img: original image, 
skel: skeletonized image 
"""
def mergeSmallArea(img, skel, thArea, thDist):
    contours, hierarchy = cv2.findContours(skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    newcon = contours[0:2]
    
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        newcon.append(contours[i])
        if area < thArea:
            parentIdx = hierarchy[0][i, 3]
            if parentIdx != -1:
                newcon.pop(-1)
                """
                maskParent = np.zeros(skel.shape, dtype=np.uint8)
                mask = np.zeros(skel.shape, dtype=np.uint8)
                cv2.drawContours(maskParent, contours, parentIdx, 255, 1)
                cv2.drawContours(mask, contours, i, 255, 1)
                mean = cv2.mean(img, mask)
                meanParent = cv2.mean(img, maskParent)
                l2dist = cv2.norm(np.array(mean) - np.array(meanParent))
                # TODO: Merge small area

                # print("L2 Distance: ", l2dist)
                """
    newone = np.zeros(skel.shape, dtype=np.uint8)
    cv2.drawContours(newone, newcon, -1, 255, 1)
    cv2.imshow('notmerged',skel)
    cv2.imshow('merged',newone)
    return newone

def visualizeDetectedGrains(cc):
    numLabels = cc[0]
    labels = cc[1]
    colors = []

    for i in range(numLabels):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        rgb = [r,g,b]
        colors.append(rgb)
    pcolor = label2rgb(labels, colors = colors)
    return pcolor.astype(np.uint8)

def cleanUpGrains(cc, img):
    labels = cc[1]
    
    pcolor = label2rgb(labels, image=img, bg_label=0, kind='avg')
    label3Ch = np.stack([labels]*3, axis=2)
    
    pcolor[label3Ch==0] = img[label3Ch==0]

    return pcolor.astype(np.uint8)

"""
img: a inverted image for connected components
"""
def grainSizeHist(img, minn, maxx, nbins):
  
    cc = cv2.connectedComponentsWithStats(img, connectivity=4, stats=cv2.CC_STAT_AREA)
    pcolor = visualizeDetectedGrains(cc)
    cv2.imshow("Segmentation", pcolor)
    cv2.imwrite("seg.png", pcolor)
    area = cc[2][:, 4]
    area = area[area!=np.max(area)]
    hist = np.histogram(area, bins=nbins, range=(minn, maxx))
    percentArea = area / np.sum(area)
    bins = np.linspace(minn, maxx, num=nbins)
    labelsHist = np.digitize(area, bins)
    y = np.zeros((nbins, ))
    for i in range(nbins):
        y[i] = np.sum(percentArea[labelsHist==i])
        
    return (hist, y, bins)
def histmatch(source,target):
    result = match_histograms(source, target, multichannel=True)
    
def segmentation(filename, origFileName):
    regenImg = cv2.imread(filename)
    origImg = cv2.imread(origFileName)
    result = histmatch(regenImg, origImg)
    

    # gau_blurred = cv2.GaussianBlur(regenImg, (3, 3), 1.5)
    # sharpened = cv2.addWeighted(regenImg, 1.5, gau_blurred, -0.5, 0)
    # TODO: Add histogram matching
    processed = regenImg
    # gray = cv2.cvtColor(regenImg, cv2.COLOR_BGR2GRAY)
    scharrx = cv2.Scharr(processed, cv2.CV_32F, 1, 0)
    scharry = cv2.Scharr(processed, cv2.CV_32F, 0, 1)

    mag = np.sqrt(scharrx ** 2 + scharry ** 2)
    rescaledMag = cv2.convertScaleAbs(mag)
    magGray = np.max(mag, axis=2)
    # magGray = np.mean(mag, axis=2)
    rescaledMagGray = cv2.convertScaleAbs(magGray)

    __, threshImg = cv2.threshold(rescaledMagGray, 254, 255, cv2.THRESH_BINARY)
    filterByArea(threshImg, 200)
    invThresh = cv2.bitwise_not(threshImg)
    filterByArea(invThresh, 10)
    threshImg = cv2.bitwise_not(invThresh)

    skel = cv2.ximgproc.thinning(threshImg)
    filterByArea(skel, 200)

    deBranched = deBranch(skel)
    mergeSmallArea(processed, skel, 300, 50)

    inv = cv2.bitwise_not(deBranched)
    skel3Ch = cv2.cvtColor(deBranched, cv2.COLOR_GRAY2BGR)

    vis = cv2.bitwise_or(skel3Ch, regenImg)
    inv3ch = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    boundary = cv2.bitwise_and(inv3ch, regenImg)

    cc = cv2.connectedComponentsWithStats(inv, connectivity=4, stats=cv2.CC_STAT_AREA)
    cleanedGrains = cleanUpGrains(cc, regenImg)
    return cleanedGrains
    cv2.imwrite("cleaned.png", cleanedGrains)

    # cv2.imshow("Processed", processed)
    # cv2.imshow("Magnitude", rescaledMag)
    # cv2.imshow("Magnitude Gray Rescaled", rescaledMagGray)
    # cv2.imshow("Threshold", threshImg)
    # cv2.imshow("Inverted Threshold", invThresh)

    # cv2.imshow("Filtered Skeletonized", skel)
    # cv2.imshow("Visualize", vis)
    # cv2.imshow("Inverted", inv)
    # cv2.imshow("Debranched", deBranched)


    # cv2.imwrite("skel.png", skel)
    # cv2.imwrite("inv.png", inv)
    # cv2.imwrite("vis.png", vis)
    # cv2.imwrite("boundary.png", boundary)
    (count, area, edges) = grainSizeHist(inv, 0, 4000, 40)
    # cv2.waitKey(0)
    plt.figure()
    plt.bar(edges, area, width=80, align='edge')
    plt.ylim(0, 0.15)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Texture Synthesis witn CNN.')
    parser.add_argument('--generation_input', '-i', type=str, nargs='+', help='Path(s) to the generated image.')
    parser.add_argument('--source_dir', '-d', type=str, help='Path(s) to the directory of the source images.')
    parser.add_argument('--output_dir', '-o', type=str, help='Output path.')
    args = parser.parse_args()

    generation_input = args.generation_input
    output_dir = args.output_dir
    source_dir = args.source_dir
    cleaned = segmentation(imgPath, sourcePath)
    for imgPath in generation_input:
        baseName = os.path.basename(imgPath)
        savePath = os.path.join(output_dir, baseName)
        sourcePath = os.path.join(source_dir, baseName)
        #cleaned = segmentation(imgPath, sourcePath)
        cv2.imwrite(savePath, cleaned)
        