from time import sleep

import os
import PIL
import scipy.misc
import math
import chi
import tensortools as tt
from tensortools import Function
import numpy as np
import gym
import tensorflow as tf
from tensorflow.contrib import layers
from .util import pp, to_json, show_all_variables

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN:
    """
    An implementation of
        Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
        https://arxiv.org/abs/1511.06434
    """
    def __init__(self, cnn: tt.Model, batch_size=64, sample_num=64, logdir=None):
        self.logdir = logdir

        def act(x: [so]):
            convnet = cnn()
            #TODO: finish this class

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def test_dcgan():
    pp.pprint(flags.FLAGS.__flags)

    @model(optimizer=tf.train.AdamOptimizer(0.0002, beta1=0.5))
    def generator(z, y=None):
        if y=None:
            s_h, s_w = FLAGS.output_height, FLAGS.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h, 4), conv_out_size_same(s_w, 4)
            s_h8, s_w8 = conv_out_size_same(s_h, 8), conv_out_size_same(s_w, 8)
            s_h16, s_w16 = conv_out_size_same(s_h, 16), conv_out_size_same(s_w, 16)

            z_ = #TODO: project
            x = layers.conv2d_transpose(z_, )
            c = layers.conv2d_transpose(x, )
            c = layers.conv2d_transpose(c, )
            c = layers.conv2d_transpose(c, )
#weights: cf deconv2 in ops
            return a





