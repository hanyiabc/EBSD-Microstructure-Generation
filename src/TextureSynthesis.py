import time
import numpy as np
from tensorflow import keras
from scipy.optimize import fmin_l_bfgs_b
from skimage.io import imread, imsave
import tensorflow as tf
import keras
from keras import backend as K
from keras import Model
from vgg19_avg import build_VGG19_avg
from keras.applications import imagenet_utils
import tensorflow_probability as tfp
import argparse
import os

tf.compat.v1.disable_eager_execution()
tfd = tfp.distributions

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
# def mi(true, pred):

def offset_range(x):
    ret = x + K.variable([103.939, 116.779, 123.68])
    return ret
    

def deep_hist_loss(source, target, n_ch, w_emd, w_mi, nbins, rangee, bandwidth):

    source = offset_range(source)
    target = offset_range(target)

    loss = K.variable(0)
    for i in range(n_ch):
        source_1ch = K.flatten(source[:, :, i])
        target_1ch = K.flatten(target[:, :, i])
        hist_source = histogram_deep_hist(source_1ch, nbins, rangee, bandwidth)
        hist_target = histogram_deep_hist(target_1ch, nbins, rangee, bandwidth)

        loss = loss + emd(hist_source, hist_target) / 3
    return loss, source_1ch, target_1ch
        

def histogram_deep_hist(data, nbins, rangee, bandwidth):
    histogram = K.variable(np.zeros((nbins, 1)))
    binNum = K.arange(nbins, dtype='float32')
    binLenth = (rangee[1] - rangee[0]) / nbins
    histogram = K.map_fn(deep_hist_bin_density(data, binLenth, rangee[0], bandwidth), binNum)
    return histogram

def deep_hist_bin_density(data, L, minn, b):
    def fn(k):
        left = k * L + minn
        right = (k + 1) * L + minn
        return K.sum(K.sigmoid((data - left) / b) - K.sigmoid( (data - right) / b)) / K.cast(K.size(data), dtype='float32')
    return fn
    

def histogram_loss(style, target):
    edges = np.linspace(-200, 200, 10, endpoint=True)
    # edges = edges.tolist()
    edges = tf.Variable(edges)
    hist0 = tfp.stats.histogram(style, edges)
    hist1 = tfp.stats.histogram(target, edges)
    hist0 /= K.sum(hist0)
    hist1 /= K.sum(hist1)
    
    return keras.losses.KLD(hist0, hist1)


def histogram_loss_2(x, generated):

    # r_range = [255 - 103.939 + 20, -103.939 - 20]
    # g_range = [255 - 116.779 + 20, -116.779 - 20]
    # b_range = [255 - 123.68 + 20, -123.68 - 20]

    r_range = [-128, 128]
    g_range = [-128, 128]
    b_range = [-128, 128]

    histB = tf.histogram_fixed_width( K.flatten(x[:, :, 0]), b_range, dtype=tf.dtypes.int32)
    histG = tf.histogram_fixed_width( K.flatten(x[:, :, 1]) , g_range, dtype=tf.dtypes.int32)
    histR = tf.histogram_fixed_width( K.flatten(x[:, :, 2]),  r_range, dtype=tf.dtypes.int32)

    histBGen = tf.histogram_fixed_width( K.flatten(generated[:, :, 0]),  b_range, dtype=tf.dtypes.int32 )
    histGGen = tf.histogram_fixed_width( K.flatten(generated[:, :, 1]),  g_range, dtype=tf.dtypes.int32 )
    histRGen = tf.histogram_fixed_width( K.flatten(generated[:, :, 2]),  r_range, dtype=tf.dtypes.int32 )

    histB = tf.cast(histB, tf.float32)
    histG = tf.cast(histG, tf.float32)
    histR = tf.cast(histR, tf.float32)
    histBGen = tf.cast(histBGen, tf.float32)
    histGGen = tf.cast(histGGen, tf.float32)
    histRGen = tf.cast(histRGen, tf.float32)

    histB /= K.sum(histB)
    histG /= K.sum(histG)
    histR /= K.sum(histR)
    histBGen /= K.sum(histBGen)
    histGGen /= K.sum(histGGen)
    histRGen /= K.sum(histRGen)

    lossB = keras.losses.mean_squared_error(histB, histBGen)
    lossG = keras.losses.mean_squared_error(histG, histGGen)
    lossR = keras.losses.mean_squared_error(histR, histRGen)

    total = (lossB + lossG + lossR) / 3

    # histLossTotal = K.sum(K.square(histR - histRGen)) + K.sum(K.square(histG - histGGen)) + K.sum(K.square(histB - histBGen))
    return total

class TextureSynthesis:
    
    def eval_loss_and_grads(self, x):
        x = x.reshape((1, self.H, self.W, 3))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        hist_loss = outs[2]
        hist = outs[3]

        print("Histogram Loss: ", hist_loss)
        print("Histogram: ", hist)

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
            hist_loss, source_1ch, target_1ch =  deep_hist_loss(x0, self.style_image, 3, 1.0, 1.0, 256, [0.0, 255.0], 0.1)
            hist_loss = H_weight * hist_loss
            self.loss = self.loss + hist_loss
            grads = K.gradients(self.loss, x0)
            outputs = [self.loss]
            outputs += grads
            outputs += [hist_loss]
            outputs += [source_1ch]
            self.f_outputs = K.function([x0], outputs)
    

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
    
    x = np.random.uniform(0, 2, (1, W, H, 3)) - 1.0
    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                        fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

    return x

def generateImageFromStyle(H=448, W=448, C=3, S_weight=1.0, V_weight=1.0, H_weight=1.0, style_image=None, style_path=None, iters=20):

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

def generateMultiImagesFromStyle(style_paths, output_dir, H=448, W=448, C=3, S_weight=1.0, V_weight=1.0, H_weight=1.0, iters=20):
    
    ts = TextureSynthesis(H=H, W=W, C=3, S_weight=S_weight, V_weight=V_weight, H_weight=H_weight, style_image=np.zeros((1, W, H, C)))
    evaluator = Evaluator(ts)

    images = readImages(style_paths)
    images = imagenet_utils.preprocess_input(images)

    numImgs = len(style_paths)
    generatedImgs = []
    for i in range(numImgs):
        K.set_value(ts.style_image, images[np.newaxis, i, :, :, :])
        generated = train(evaluator,H, W, iters)
        generated = deprocess_image(generated, H, W)
        generatedImgs.append(generatedImgs)
        styleImgName = os.path.basename(style_paths[i])
        savePath = os.path.join(output_dir, styleImgName)
        imsave(savePath, generated)



if __name__ == '__main__':
    H = 448
    W = 448
    parser = argparse.ArgumentParser(description='Texture Synthesis witn CNN.')
    parser.add_argument('--style', '-s', type=str, nargs='+', help='Path to the style reference image.')
    parser.add_argument('--output_dir', '-o', type=str, help='Output path.')
    parser.add_argument('--style_weight', '-sw', type=float, default=1.0, help='Style weight.')
    parser.add_argument('--tv_weight', '-tw', type=float, default=0.05, help='TV MIN weight.')
    parser.add_argument('--hist_weight', '-hw', type=float, default=1.0e9, help='Histogram weight.')
    parser.add_argument('--num_iter', '-n', type=int, default=40, help='Number of iterations.')

    args = parser.parse_args()


    style_path = args.style
    style_weight = args.style_weight
    output_dir = args.output_dir
    tv_weight = args.tv_weight
    hist_weight = args.hist_weight
    n_iter = args.num_iter

    generateMultiImagesFromStyle(style_path, output_dir, H=H, W=W, C=3, S_weight=style_weight, V_weight=tv_weight, H_weight=hist_weight, iters=n_iter)