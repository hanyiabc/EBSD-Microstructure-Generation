import time
import numpy as np
from numpy.core.numeric import outer
from tensorflow import keras
from scipy.optimize import fmin_l_bfgs_b
from skimage.io import imread, imsave
import tensorflow as tf
import keras
from keras import backend as K
from keras import Model
from vgg19_avg import build_VGG19_avg
from keras.applications import imagenet_utils
import argparse
import os

tf.compat.v1.disable_eager_execution()

def readImages(filenames):
    imgs = []
    for file in filenames:
        img = imread(file)
        imgs.append(img)
    return np.stack(imgs, axis=0)

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

def emd(true, pred):
    return K.sum(K.square(true - pred))

def entropy(distr_joint):
    ret = -1 * K.sum(distr_joint * K.log(distr_joint))
    return ret

def mi(distr_true, distr_pred, distr_joint):
    distr_gram = K.expand_dims(distr_pred, axis=0) * K.transpose(K.expand_dims(distr_true, axis=0))
    ret = K.sum(distr_joint * K.log(distr_joint / distr_gram))
    return ret

def mi_loss(distr_true, distr_pred, distr_joint):
    ret = 1 - mi(distr_true, distr_pred, distr_joint) / entropy(distr_joint)
    return ret

def offset_range(x):
    ret = x + K.variable([103.939, 116.779, 123.68])
    return ret
    

def deep_hist_loss(target, source, n_ch, w_emd, w_mi, nbins, rangee, bandwidth):

    source = offset_range(source)
    target = offset_range(target)

    loss = K.variable(0)

    binNum = K.arange(nbins, dtype='float32')
    binLenth = (rangee[1] - rangee[0]) / nbins

    for i in range(n_ch):
        source_1ch = K.flatten(source[:, :, i])
        target_1ch = K.flatten(target[:, :, i])

        activations_source = K.map_fn(rect_func(source_1ch, binLenth, rangee[0], bandwidth), binNum)
        activations_target = K.map_fn(rect_func(target_1ch, binLenth, rangee[0], bandwidth), binNum)

        hist_source = K.map_fn(deep_hist_bin_density, activations_source)
        hist_target = K.map_fn(deep_hist_bin_density, activations_target)
        hist_joint = deep_hist_joint_bin_density(activations_source, activations_target)
        
        loss = loss + emd(hist_target, hist_source) / 3 * w_emd
        loss = loss + mi_loss(hist_target, hist_source, hist_joint) / 3 * w_mi


    return loss, hist_source, hist_target

 

def rect_func(data, L, minn, b):
    def fn(k):
        left = k * L + minn
        right = (k + 1) * L + minn
        pi = K.sigmoid((data - left) / b) - K.sigmoid( (data - right) / b)
        return pi
    return fn

def deep_hist_joint_bin_density(activation_src, activation_tgt):
    return K.dot(activation_src, K.transpose(activation_tgt)) / activation_src.shape[1]

def deep_hist_bin_density(activations):
    return K.sum(activations) / K.cast(K.size(activations), dtype='float32')

class TextureSynthesis:
    
    def eval_loss_and_grads(self, x):
        x = x.reshape((1, self.H, self.W, 3))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        # hist_loss = outs[2]
        # hist = outs[3]
        # print("Histogram Loss: ", hist_loss)
        # print(hist)
        return loss_value, grad_values

    def __init__(self, H=448, W=448, C=3, S_weight=1.0, V_weight = 1.0, H_weight = 1.0, style_path=None, style_gram_mat=None, style_image=None, vgg19=None):

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

            x0 = input_tensor
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
            self.loss = self.loss + V_weight * total_variation_loss(x0, H, W)
            grads = K.gradients(self.loss, x0)
            outputs = [self.loss]
            outputs += grads
            self.f_outputs = K.function([x0], outputs)
        else:
            if style_path:
                style_image = imread(style_path)[np.newaxis, :]
            style_image = imagenet_utils.preprocess_input(style_image)
            self.style_image = K.variable(style_image)
            x0 = K.placeholder((1, H, W, 3))

            input_tensor = K.concatenate([self.style_image, x0], axis=0)

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
            self.loss = self.loss + V_weight * total_variation_loss(x0, H, W)
            # hist_loss, source_1ch, target_1ch =  deep_hist_loss(self.style_image, x0, 3, 1.0, 1.0, 256, [-1.0, 255.0], 0.05)
            # hist_loss = H_weight * hist_loss
            # self.loss = self.loss + hist_loss
            grads = K.gradients(self.loss, x0)
            outputs = [self.loss]
            outputs += grads
            self.f_outputs = K.function([x0], outputs)
    

class Evaluator(object):

    def __init__(self, textureSynthesis):
        self.loss_value = None
        self.grads_values = None
        self.ts = textureSynthesis
        self.i = 1
        self.losses = []
        self.mid_imgs = []
    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.ts.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        if self.i % 200 == 0 or self.i == 1:
            print("Iteration %d: loss=%.2f" % (self.i, self.loss_value))
            img = deprocess_image(x, self.ts.H, self.ts.W)
            self.mid_imgs.append(img)
            self.losses.append(self.loss_value)
            # fname = "data/iters/regen" + "_at_iteration_%d.png" % self.i
            # keras.preprocessing.image.save_img(fname, img)
        self.i+=1
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
        
    def reset(self):
        self.losses.clear()
        self.mid_imgs.clear()
        self.i=1



def train(evaluator, H, W, C, iterations=500, get_mid_res=False):
    
    x = np.random.uniform(0, 255, (1, W, H, C)) - 128

    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=iterations, factr=1e4, pgtol=1e-8)
    img = deprocess_image(x, H, W)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration completed in %ds' % (end_time - start_time, ))

    if get_mid_res is True:
        return img, evaluator.losses, evaluator.mid_imgs
    return img

def generateImageFromStyle(H=448, W=448, C=3, S_weight=1.0, V_weight=1.0, H_weight=1.0, style_image=None, style_path=None, iters=500,  get_mid_res=False):

    ts = TextureSynthesis(H=H, W=W, C=3, S_weight=S_weight, V_weight=V_weight, style_image=style_image, style_path=style_path)
    evaluator = Evaluator(ts)
    generated = train(evaluator, H, W, C ,iters, get_mid_res=get_mid_res)
    
    return generated

def generateImageFromGramMatrix(gram_mat, H=448, W=448, C=3, S_weight=1.0, V_weight=1.0, iters=500, vgg19=None):
    ts = TextureSynthesis(H=H, W=W, C=3, S_weight=S_weight, V_weight=V_weight, style_gram_mat=gram_mat, vgg19=vgg19)
    evaluator = Evaluator(ts)
    generated = train(evaluator,H, W, C, iters)
    return generated

def generateMultiImagesFromStyle(style_paths, output_dir=None, H=448, W=448, C=3, S_weight=1.0, V_weight=1.0, H_weight=1.0, iters=500):
    
    ts = TextureSynthesis(H=H, W=W, C=3, S_weight=S_weight, V_weight=V_weight, H_weight=H_weight, style_image=np.zeros((1, W, H, C)))
    evaluator = Evaluator(ts)

    images = readImages(style_paths)
    images = imagenet_utils.preprocess_input(images)

    generatedImgs = []
    for i in range(len(style_paths)):
        evaluator.reset()
        print("Working on: ", style_paths[i])
        K.set_value(ts.style_image, images[np.newaxis, i, :, :, :])
        generated = train(evaluator,H, W, C, iters)
        generatedImgs.append(generated)
        if output_dir is not None:
            styleImgName = os.path.basename(style_paths[i])
            savePath = os.path.join(output_dir, styleImgName)
            print("Saving generated image to", savePath)
            imsave(savePath, generated)
    return generatedImgs



if __name__ == '__main__':
    H = 448
    W = 448
    parser = argparse.ArgumentParser(description='Texture Synthesis witn CNN.')
    parser.add_argument('--style', '-s', type=str, nargs='+', help='Path to the style reference image.')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='Output path.')
    parser.add_argument('--style_weight', '-sw', type=float, default=1e-6, help='Style weight.')
    parser.add_argument('--tv_weight', '-tw', type=float, default=1e-6, help='TV MIN weight.')
    parser.add_argument('--hist_weight', '-hw', type=float, default=0, help='Histogram weight.')
    parser.add_argument('--num_iter', '-n', type=int, default=40, help='Number of iterations.')

    args = parser.parse_args()


    style_path = args.style
    style_weight = args.style_weight
    output_dir = args.output_dir
    tv_weight = args.tv_weight
    hist_weight = args.hist_weight
    n_iter = args.num_iter

    generateMultiImagesFromStyle(style_path, output_dir, H=H, W=W, C=3, S_weight=style_weight, V_weight=tv_weight, H_weight=hist_weight, iters=n_iter)