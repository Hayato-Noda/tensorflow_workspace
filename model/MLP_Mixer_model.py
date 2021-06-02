#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange, Reduce

class MLPBlock(layers.Layer):
    def __init__(self, mixing, d_ff):
        super(MLPBlock, self).__init__()
        if mixing != 'token' and mixing != 'channel':
            raise ValueError("undefiend mixing")
        self.mixing = mixing
        self.d_ff = d_ff
        
    def build(self, input_shape):
        d_out = input_shape[-1] if self.mixing=='channel' else input_shape[-2]
        self.dense_1 = layers.Dense(self.d_ff, use_bias=False)
        self.dense_2 = layers.Dense(d_out, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        '''
        inputsは(batches, tokens, channels)
        Denseは最後の軸に作用
        '''
        x = self.norm(inputs)
        x = x if self.mixing!='token' else tf.transpose(x, perm=[0, 2, 1])
        x =self.dense_1(x)
        x = tf.keras.activations.gelu(x, approximate=True)
        x = self.dense_2(x)
        x = x if self.mixing!='token' else tf.transpose(x, perm=[0, 2, 1])
        return x + inputs

class MixerBlock(layers.Layer):
    def __init__(self, d_s, d_c):
        super(MixerBlock, self).__init__()
        self.mlp_1 = MLPBlock(mixing='token', d_ff=d_s)
        self.mlp_2 = MLPBlock(mixing='channel', d_ff=d_c)

    def call(self, inputs):
        x = self.mlp_1(inputs)
        x = self.mlp_2(x)
        return x

#StemがDenseModel
def MLP_Mixer1_build(**config):
    shapes = (config['image_size'], config['image_size'], config['channels'])
    hidden_dim  = config['hidden_dim']
    patch_size = config['patch_size']
    num_blocks = config['depth']
    tokens_mlp_dim = config['tokens_mlp_dim']
    channels_mlp_dim = config['channels_mlp_dim']
    num_classes = config['num_classes']
    size = (config['image_size']//config['patch_size'])  #4

    inputs = layers.Input(shape=shapes)
    x = Rearrange('b (h x) (w y) c -> b (h w) (x y c)', x=size, y=size)(inputs) #(B, 64, 48)
    x = layers.Dense(hidden_dim, use_bias=False)(x)

    for i in range(num_blocks):
      x = MixerBlock(tokens_mlp_dim, channels_mlp_dim)(x)
    
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)



#StemがConvModel
def MLP_Mixer2_build(**config):
    shapes = (config['image_size'], config['image_size'], config['channels'])
    hidden_dim  = config['hidden_dim']
    patch_size = config['patch_size']
    num_blocks = config['depth']
    tokens_mlp_dim = config['tokens_mlp_dim']
    channels_mlp_dim = config['channels_mlp_dim']
    num_classes = config['num_classes']
    size = config['image_size']//config['patch_size']

    inputs = layers.Input(shape=shapes)
    x = layers.Conv2D(hidden_dim, (size, size), strides=(size, size))(inputs)
    x = Rearrange('n h w c -> n (h w) c')(x)

    for _ in range(num_blocks):
      x = MixerBlock(tokens_mlp_dim, channels_mlp_dim)(x)
    
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)

