import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19, imagenet_utils
import time
from scipy.optimize import fmin_l_bfgs_b
from skimage.io import imread, imsave
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from vgg19_avg import build_VGG19_avg
import argparse
import os



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
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, combination, size):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def style_loss_constant(gram_style, combination, size):
    S = gram_style
    C = gram_matrix(combination)
    channels = 3
    return tf.reduce_sum(tf.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x, H, W):
    a = tf.square(x[:, :H-1, :W-1, :] - x[:, 1:, :W-1, :])
    b = tf.square(x[:, :H-1, :W-1, :] - x[:, :H-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

def emd(true, pred):
    return tf.reduce_sum(tf.square(true - pred))

def entropy(distr_joint):
    ret = -1 * tf.reduce_sum(distr_joint * safelog(distr_joint))
    return ret

def safelog(x):
    return tf.math.log(x + 1e-5)

def mi(distr_true, distr_pred, distr_joint):
    distr_gram = tf.expand_dims(distr_pred, axis=0) * tf.transpose(tf.expand_dims(distr_true, axis=0))
    ret = tf.reduce_sum(distr_joint * (safelog(distr_joint) - safelog(distr_gram)))
    return ret

def mi_loss(distr_true, distr_pred, distr_joint):
    ret = 1 - mi(distr_true, distr_pred, distr_joint) / entropy(distr_joint)
    return ret

def offset_range(x):
    ret = x + np.array([103.939, 116.779, 123.68], dtype=np.float32)
    return ret
    

def deep_hist_loss(target, source, n_ch, w_emd, w_mi, nbins, rangee, bandwidth):

    source = offset_range(source)
    target = offset_range(target)

    loss = tf.zeros(shape=())

    binNum = tf.range(nbins, dtype='float32')
    binLenth = (rangee[1] - rangee[0]) / nbins

    for i in range(n_ch):
        source_1ch = tf.reshape(source[:, :, i], [-1])
        target_1ch = tf.reshape(target[:, :, i], [-1])

        activations_source = tf.map_fn(rect_func(source_1ch, binLenth, rangee[0], bandwidth), binNum)
        activations_target = tf.map_fn(rect_func(target_1ch, binLenth, rangee[0], bandwidth), binNum)

        hist_source = tf.map_fn(deep_hist_bin_density, activations_source)
        hist_target = tf.map_fn(deep_hist_bin_density, activations_target)
        hist_joint = deep_hist_joint_bin_density(activations_source, activations_target)
        
        loss = loss + emd(hist_target, hist_source) / 3 * w_emd
        loss = loss + mi_loss(hist_target, hist_source, hist_joint) / 3 * w_mi


    return loss, hist_source, hist_target

 

def rect_func(data, L, minn, b):
    def fn(k):
        left = k * L + minn
        right = (k + 1) * L + minn
        pi = tf.math.sigmoid((data - left) / b) - tf.math.sigmoid( (data - right) / b)
        return pi
    return fn

def deep_hist_joint_bin_density(activation_src, activation_tgt):
    return tf.matmul(activation_src, tf.transpose(activation_tgt)) / activation_src.shape[1]

def deep_hist_bin_density(activations):
    return tf.reduce_sum(activations) / tf.cast(tf.size(activations), dtype='float32')


class TextureSynthesis:
    
    def train(self, get_mid_res=False):
        optimizer = keras.optimizers.Adam(
            keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
            )
        )
        loss = []
        mid_imgs = []
        for i in range(1, self.n_iter + 1):
            self.loss, self.grads = self.compute_loss_and_grads()
            if i % 200 == 0 or i == 1:
                print("Iteration %d: loss=%.2f" % (i, self.loss))
                img = deprocess_image(self.x0.numpy(), self.H, self.W)
                loss.append(self.loss)
                mid_imgs.append(img)
                # fname = "data/iters/regen" + "_at_iteration_%d.png" % i
                # keras.preprocessing.image.save_img(fname, img)
            optimizer.apply_gradients([(self.grads, self.x0)])
        img = deprocess_image(self.x0.numpy(), self.H, self.W)
        if get_mid_res is True:
            return img, loss, mid_imgs

        return img

    @tf.function
    def compute_loss_and_grads(self):
        with tf.GradientTape() as tape:
            self.loss = self.compute_loss_img()
        self.grads = tape.gradient(self.loss, self.x0)
        return self.loss, self.grads
    def compute_loss_img(self):
        feature_layers = ['block1_pool', 'block2_pool',
        'block3_pool', 'block4_pool',
        'block1_conv1']
        self.loss = tf.zeros(shape=())

        self.input_tensor = tf.concat([self.style_image, self.x0], axis=0)
        features = self.feature_extractor(self.input_tensor)

        for layer_name in feature_layers:
            layer_features = features[layer_name]
            style_features = layer_features[0, :, :, :]
            combination_features = layer_features[1, :, :, :]
            sl = style_loss(style_features, combination_features, self.H * self.W)
            self.loss += (self.S_weight / len(feature_layers)) * sl

        self.loss = self.loss + self.V_weight * total_variation_loss(self.x0, self.H, self.W)

        # hist_loss, source_1ch, target_1ch =  deep_hist_loss(self.style_image, self.x0, 3, 1.0, 1.0, 256, [-1.0, 255.0], 0.05)
        # hist_loss = self.H_weight * hist_loss
        # self.loss = self.loss + hist_loss
        
        return self.loss

    def __init__(self, H=448, W=448, C=3, S_weight=1.0, V_weight = 1.0, H_weight = 1.0, style_path=None, style_gram_mat=None, style_image=None, vgg19=None, n_iter=5):

        self.H = H
        self.W = W
        self.S_weight = S_weight
        self.V_weight = V_weight
        self.H_weight = H_weight
        self.n_iter = n_iter

        if style_gram_mat is not None:
            
            if vgg19 is None:
                input_tensor = keras.Input(shape=(H, W, 3))
                self.model = build_VGG19_avg(input_tensor)
            else:
                input_tensor = vgg19.input
                self.model = vgg19

            # x0 = input_tensor
            # self.layers = dict([(layer.name, layer.output) for layer in self.model.layers])
            # idx = 0
            # self.loss = K.constant(0)

            # for layer_name in feature_layers:
            #     layer_features = self.layers[layer_name]
            #     combination_features = layer_features[0, :, :, :]
            #     gram_size = combination_features.shape[2]
            #     style_gram_mat_tensor = K.reshape(K.variable(style_gram_mat[idx:idx + gram_size ** 2]), ((gram_size, gram_size))) 
            #     idx += gram_size ** 2
            #     sl = style_loss_constant(style_gram_mat_tensor, combination_features, H * W)
            #     self.loss += (S_weight / len(feature_layers)) * sl
            # self.loss = self.loss + V_weight * total_variation_loss(x0, H, W)
            # grads = K.gradients(self.loss, x0)
            # outputs = [self.loss]
            # outputs += grads
            # self.f_outputs = K.function([x0], outputs)
        else:
            if style_path:
                style_image = imread(style_path)[np.newaxis, :].astype(np.float32)
            self.style_image = imagenet_utils.preprocess_input(style_image).astype(np.float32)
            # self.style_image = tf.Variable(style_image.astype(np.float32))
            self.x0 = np.random.uniform(0, 255, (1, W, H, C)) - 128
            self.x0 = tf.Variable(self.x0.astype(np.float32))

            if vgg19 is None:
                self.model = build_VGG19_avg(input_shape=(W, H, C))
            else:
                self.model = vgg19
            
            self.layers = dict([(layer.name, layer.output) for layer in self.model.layers])
            self.feature_extractor = keras.Model(inputs=self.model.inputs, outputs=self.layers)



def generateImageFromStyle(H=448, W=448, C=3, S_weight=1.0, V_weight=1.0, H_weight=1.0, style_image=None, style_path=None, iters=20, get_mid_res=False):

    ts = TextureSynthesis(H=H, W=W, C=3, S_weight=S_weight, V_weight=V_weight, H_weight=H_weight, style_image=style_image, style_path=style_path, n_iter=iters)

    generated = ts.train(get_mid_res=get_mid_res)
    return generated

# def generateImageFromGramMatrix(gram_mat, H=448, W=448, C=3, S_weight=1.0, V_weight=1.0, iters=20, vgg19=None):
#     ts = TextureSynthesis(H=H, W=W, C=3, S_weight=S_weight, V_weight=V_weight, style_gram_mat=gram_mat, vgg19=vgg19, n_iter=iters)
#     generated = ts.train()
#     return generated

def generateMultiImagesFromStyle(style_paths, output_dir, H=448, W=448, C=3, S_weight=1.0, V_weight=1.0, H_weight=1.0, iters=20):
    
    ts = TextureSynthesis(H=H, W=W, C=3, S_weight=S_weight, V_weight=V_weight, H_weight=H_weight, style_image=np.zeros((1, W, H, C)), n_iter=iters)
    images = readImages(style_paths)
    images = imagenet_utils.preprocess_input(images)

    numImgs = len(style_paths)
    generatedImgs = []
    for i in range(numImgs):
        ts.style_image = images[np.newaxis, i, :, :, :].astype(np.float32)
        generated = ts.train()
        generatedImgs.append(generatedImgs)
        styleImgName = os.path.basename(style_paths[i])
        savePath = os.path.join(output_dir, styleImgName)
        imsave(savePath, generated)
    return generatedImgs

if __name__ == '__main__':
    H = 448
    W = 448
    parser = argparse.ArgumentParser(description='Texture Synthesis witn CNN.')
    parser.add_argument('--style', '-s', type=str, nargs='+', help='Path to the style reference image.')
    parser.add_argument('--output_dir', '-o', type=str, help='Output path.')
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