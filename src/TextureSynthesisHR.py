import time
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from skimage.io import imread, imsave
import tensorflow as tf
from keras import Model
from vgg19_avg import build_VGG19_avg
from keras.applications import imagenet_utils
import os
import keras
from keras.layers import Lambda
from keras.models import Model
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K
from scipy.ndimage import interpolation
from scipy import linalg
from scipy.optimize import minimize
from PIL import Image
import re
from keras.utils import conv_utils
from enum import Enum
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def deprocess_image(x, H, W):

    x = x.reshape((W, H, 3))

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR -> RGB
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination, size):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def style_loss_constant(gram_style, combination, size):
    S = gram_style
    C = gram_matrix(combination)
    channels = 3
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x, H, W):
    a = K.square(x[:, :H-1, :W-1, :] - x[:, 1:, :W-1, :])
    b = K.square(x[:, :H-1, :W-1, :] - x[:, :H-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

class TextureSynthesis:
    
    def eval_loss_and_grads(self, x):
        x = x.reshape((1, self.H, self.W, 3))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        return loss_value, grad_values

    def __init__(self, H=448, W=448, C=3, S_weight=1.0, V_weight = 1.0, style_path=None, style_gram_mat=None, style_image=None, vgg19=None):

        self.H = H
        self.W = W

        feature_layers = ['block1_pool', 'block2_pool',
        'block3_pool', 'block4_pool',
        'block1_conv1']

        if style_gram_mat is not None:
            
            if vgg19 is None:
                input_tensor = keras.Input(shape=(H, W, 3))
                self.model = build_VGG19_avg(input_tensor)
            else:
                input_tensor = vgg19.input
                self.model = vgg19

            combination_image = input_tensor
            self.layers = dict([(layer.name, layer.output) for layer in self.model.layers])
            idx = 0
            self.loss = K.constant(0)
            for layer_name in feature_layers:
                layer_features = self.layers[layer_name]
                combination_features = layer_features[0, :, :, :]
                gram_size = combination_features.shape[2]
                style_gram_mat_tensor = K.reshape(K.variable(style_gram_mat[idx:idx + gram_size ** 2]), ((gram_size, gram_size))) 
                idx += gram_size ** 2
                sl = style_loss_constant(style_gram_mat_tensor, combination_features, H * W)
                self.loss += (S_weight / len(feature_layers)) * sl
            self.loss = self.loss + V_weight * total_variation_loss(combination_image, H, W)
            grads = K.gradients(self.loss, combination_image)
            outputs = [self.loss]
            outputs += grads
            self.f_outputs = K.function([combination_image], outputs)
        else:
            if style_path:
                style_image = imread(style_path)[np.newaxis, :]
            style_image = imagenet_utils.preprocess_input(style_image)
            style_image = K.variable(style_image)
            combination_image = K.placeholder((1, H, W, 3))

            input_tensor = K.concatenate([style_image, combination_image], axis=0)

            if vgg19 is None:
                self.model = build_VGG19_avg(input_tensor)
            else:
                self.model = vgg19(input_tensor)

            self.layers = dict([(layer.name, layer.output) for layer in self.model.layers])
            self.loss = K.constant(0)
            for layer_name in feature_layers:
                layer_features = self.layers[layer_name]
                style_features = layer_features[0, :, :, :]
                combination_features = layer_features[1, :, :, :]
                sl = style_loss(style_features, combination_features, H * W)
                self.loss += (S_weight / len(feature_layers)) * sl
            self.loss = self.loss + V_weight * total_variation_loss(combination_image, H, W)
            grads = K.gradients(self.loss, combination_image)
            outputs = [self.loss]
            outputs += grads
            self.f_outputs = K.function([combination_image], outputs)
    

class Evaluator(object):

    def __init__(self, textureSynthesis):
        self.loss_value = None
        self.grads_values = None
        self.ts = textureSynthesis

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.ts.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def train(evaluator, H, W, iterations=10):
    
    x = np.random.uniform(0, 1, (1, W, H, 3)) - 0.5
    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                        fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

    return x

def generateImageFromStyle(H=448, W=448, C=3, S_weight=1.0, V_weight=1.0, style_image=None, iters=20, style_path=None):

    ts = TextureSynthesis(H=H, W=W, C=3, S_weight=S_weight, V_weight=V_weight, style_image=style_image, style_path=style_path)
    evaluator = Evaluator(ts)
    generated = train(evaluator, H, W ,iters)
    generated = deprocess_image(generated, H, W)
    return generated

def generateImageFromGramMatrix(gram_mat, H=448, W=448, C=3, S_weight=1.0, V_weight=1.0, iters=20, vgg19=None):
    ts = TextureSynthesis(H=H, W=W, C=3, S_weight=S_weight, V_weight=V_weight, style_gram_mat=gram_mat, vgg19=vgg19)
    evaluator = Evaluator(ts)
    generated = train(evaluator,H, W, iters)
    generated = deprocess_image(generated, H, W)
    return generated


if __name__ == '__main__':
    H = 448
    W = 448
    N_ITERS = 40

    TOTAL_VARIATION_WEIGHT = 0.1
    STYLE_WEIGHT = 1.0


    STYLE_PATH = 'data/small/300.3_full.png'
    generated = generateImageFromStyle(H=H, W=W, C=3, S_weight=STYLE_WEIGHT, V_weight=TOTAL_VARIATION_WEIGHT, style_path=STYLE_PATH, iters=N_ITERS)
    imsave('generated.png', generated)