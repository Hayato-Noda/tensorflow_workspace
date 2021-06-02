#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras import initializers
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange, Reduce


#gMLPBlockのmodelと同様だが、inputとoutputが４次元になっているためそれに伴う部分だけ変更
class gMLPBlock(layers.Layer):
    '''
    inputsは(batches, windows, tokens, channels)
    outputsは(batches, windows, tokens, channels)
    '''
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
        x = self.norm_1(inputs)
        x =self.dense_1(x)
        x = tf.keras.activations.gelu(x, approximate=True)
        x = self.spatial_gating_unit(x)
        x = self.dense_2(x)

        return x + inputs

    def spatial_gating_unit(self, inputs):
        u, v = tf.split(inputs,2,axis=-1)
        v = self.norm_2(v)
        v = tf.transpose(v, perm=[0, 1, 3, 2])
        v = self.dense_3(v)
        v = tf.transpose(v, perm=[0, 1, 3, 2])

        return u * v
    
    def get_config(self):
        config = {
            "d_ff" : self.d_ff,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Tokens2Img(layers.Layer):
    def __init__(self):
        super(Tokens2Img, self).__init__()

    def call(self, inputs):
        _, h, w, c =inputs.shape
        p = int(np.sqrt(h*w))
        x = tf.reshape(inputs, [-1,p,p,c])
        return x

def build_nesT(**config):
    shapes = (config['image_size'], config['image_size'], config['channels'])
    hidden_dim  = config['hidden_dim']
    num_classes = config['num_classes']
    size_blocks = config['size_blocks'] #4
    patch_size = config['patch_size'] #4
    num_blocks = config['depth']
    d_ff = config['d_ff']
    aggregation_option = 2
    
    inputs = layers.Input(shape=shapes)
    x = Rearrange('n (h x) (w y) c -> n (x y) h w c', x=size_blocks, y=size_blocks)(inputs) #IMG2Block [N,32,32,3] -> [N,16,8,8,3]
    x = Rearrange('n b (h x) (w y) c -> n b h w (x y c)', h=patch_size, w=patch_size)(x)  #Block2Patch [N,16,8,8,3] ->[N,16,4,4,12]
    x = layers.Dense(hidden_dim, use_bias=False)(x) #Dense [N,16,4,4,12] ->[N,16,4,4,128]

    for i in range(num_blocks):
      x = Rearrange('n b h w c -> n b (h w) c')(x)  #Flatten [N,16,4,4,128] ->[N,16,16,128]
      x = gMLPBlock(d_ff)(x)

      if aggregation_option==1:
        conv1 = layers.Conv2D(hidden_dim,(3,3),padding='same')
        x = [[conv1(block) for block in blocks] for blocks in x]
        x = Tokens2Img()(x) # tokensの二重リストをimgへ戻す
      elif aggregation_option==2:
        x = Tokens2Img()(x)
        x = layers.Conv2D(hidden_dim,(3,3),padding='same')(x)
        x = layers.LayerNormalization()(x)
      x = layers.MaxPool2D((3,3),strides=2,padding='same')(x) #Flatten [N,16,16,128] ->[N,8,8,128]

      size_blocks = size_blocks//2

      if i!=num_blocks-1:
        x = Rearrange('n (h x) (w y) c -> n (x y) h w c', x=size_blocks, y=size_blocks)(x) #IMG2Block [N,8,8,128] -> [N,4,4,4,128]

    x = layers.GlobalAveragePooling2D()(x)
#    x = layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)

