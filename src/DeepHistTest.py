import numpy as np
import keras
import skimage.io
import skimage.exposure
from TextureSynthesis import readImages, deprocess_image
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def deep_hist_loss(img, n_ch, nbins, rangee, bandwidth):
    loss = 0
    hist_3ch = []

    for i in range(n_ch):
        
        img_1ch = img[:, :, i].flatten()
        hist_1ch = histogram_deep_hist(img_1ch, nbins, rangee, bandwidth)
        hist_3ch.append(hist_1ch)

    return hist_3ch

def get_hist_3ch(img, n_ch, nbins, rangee):
    hist_3ch = []
    
    for i in range(n_ch):
        img_1ch = img[:, :, i].flatten()
        hist, __ = np.histogram(img_1ch, nbins, rangee)
        hist_3ch.append(hist / np.sum(hist))
    return hist_3ch

def sk_hist_3ch(img, n_ch):
    hist_3ch = []

    for i in range(n_ch):
        img_1ch = img[:, :, i].flatten()
        hist, __ = skimage.exposure.histogram(img, normalize=True)
        hist_3ch.append(hist)
    return hist_3ch

def histogram_deep_hist(data, nbins, rangee, bandwidth):

    binNum = np.arange(nbins, dtype='float32')
    binLenth = (rangee[1] - rangee[0]) / nbins

    func = deep_hist_bin_density(data, binLenth, rangee[0], bandwidth)
    vfunc = np.vectorize(func)
    histogram = vfunc(binNum)
    return histogram

def deep_hist_bin_density(data, L, minn, b):
    def fn(k):
        left = k * L + minn
        right = (k + 1) * L + minn
        return np.sum(sigmoid((data - left) / b) - sigmoid( (data - right) / b)) / data.shape[0]
    return fn

def plot_3ch_hist(hist, nbins):
    fig,a =  plt.subplots(3)
    
    binNum = np.arange(nbins, dtype='float32')
    a[0].bar(binNum, hist[0])
    a[0].set_ylim((0, 0.015))
    a[1].bar(binNum, hist[1])
    a[1].set_ylim((0, 0.015))
    a[2].bar(binNum, hist[2])
    a[2].set_ylim((0, 0.015))
    plt.show()
if __name__ == '__main__':
    H = 448
    W = 448

    IMG_FILE = ".\data\small\sim1_full.png"
    # img = skimage.io.imread(IMG_FILE)[np.newaxis, :]
    img = skimage.io.imread(IMG_FILE)
    # img_proced = imagenet_utils.preprocess_input(img)

    deep_hist =  deep_hist_loss(img, 3, 256, [0.0, 255.0], 0.01)

    ori_hist = sk_hist_3ch(img, 3)

    plot_3ch_hist(deep_hist, 256)
    plot_3ch_hist(ori_hist, 256)


