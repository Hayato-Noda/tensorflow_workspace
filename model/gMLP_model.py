#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras import initializers
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange, Reduce


class gMLPBlock(layers.Layer):
    def __init__(self, d_ff, **kwargs):
        super(gMLPBlock, self).__init__(**kwargs)
        self.d_ff = d_ff
        
    def build(self, input_shape):
        d_out = input_shape[-1]
        n_token = input_shape[-2]
        self.dense_1 = layers.Dense(self.d_ff, use_bias=False)
        self.dense_2 = layers.Dense(d_out, use_bias=False)
        self.dense_3 = layers.Dense(n_token, kernel_initializer=initializers.RandomNormal(stddev=0.001),
                                    bias_initializer=initializers.Ones())
        self.norm_1 = layers.LayerNormalization()
        self.norm_2 = layers.LayerNormalization()

    def call(self, inputs):
        '''
        inputsは(batches, tokens, channels)
        Denseは最後の軸に作用
        '''
        x = self.norm_1(inputs)
        x =self.dense_1(x)
        x = tf.keras.activations.gelu(x, approximate=True)
        x = self.spatial_gating_unit(x)
        x = self.dense_2(x)

        return x + inputs

    def spatial_gating_unit(self, inputs):
        '''
        inputsは(batches, tokens, channels/2)二つへ分割
        channelsが奇数の時は実装を改良
        vを(tokens, tokens)でアフィン変換。
        これがゲート値なので、重みはほぼ0、バイアスは1で初期化
        '''
        u, v = tf.split(inputs,2,axis=-1)
        v = self.norm_2(v)
        v = tf.transpose(v, perm=[0, 2, 1])
        v = self.dense_3(v)
        v = tf.transpose(v, perm=[0, 2, 1])

        return u * v
    
    def get_config(self):
        config = {
            "d_ff" : self.d_ff,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_gMLP(**config):
    shapes = (config['image_size'], config['image_size'], config['channels'])
    hidden_dim  = config['hidden_dim']
    num_blocks = config['depth']
    d_ff = config['d_ff']
    num_classes = config['num_classes']
    size = config['image_size']//config['patch_size']

    inputs = layers.Input(shape=shapes)
    x = Rearrange('b (h x) (w y) c -> b (h w) (x y c)', x=size, y=size)(inputs) #(B, 64, 48)
    x = layers.Dense(hidden_dim, use_bias=False)(x)

    for _ in range(num_blocks):
      x = gMLPBlock(d_ff)(x)
    
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)

