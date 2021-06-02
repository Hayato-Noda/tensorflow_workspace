#!/usr/bin/env python
# coding: utf-8
import numpy as np
from einops import rearrange, repeat
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def Visual_weight(i,model, **config):
  norm = mcolors.DivergingNorm(vcenter=0.0)

  weights = model.layers[i].get_weights()[0].reshape((config['patch_size'], 
  config['patch_size'], config['tokens_mlp_dim']))
  num_h = int(np.sqrt(config['hidden_dim']))
  num_w = config['hidden_dim']//num_h
  full_img = np.zeros(((config['patch_size']+2)*num_h-2, (config['patch_size']+2)*num_w-2))

  for i in range(num_h):
    for j in range(num_w):
      idx = config['patch_size']*i+j
      full_img[(config['patch_size']+2)*i:(config['patch_size']+2)*i+config['patch_size'],
      (config['patch_size']+2)*j:(config['patch_size']+2)*j+config['patch_size']] = weights[:, :, idx]

  plt.figure(figsize=(10,10))
  plt.imshow(full_img, cmap='bwr', norm=norm)
  plt.axis("off")

def Visualize_patch(x_train, **config):
  patch_size = config['patch_size'] #p=8
  img_size = x_train.shape[1]
  size = img_size//patch_size # patchのサイズ
  steps = (img_size-size)//size+1 # 一辺の分割回数、今のストライドなら=P
  full_img = np.ones(((size+2)*steps-2, (size+2)*steps-2, 3))
  batch_idx = -1

  img_array = rearrange(x_train, 'b (h x) (w y) c -> b (h w) x y c', x=size, y=size) #[B,32,32,3]->[B,64,4,4,3]

  for i in range(steps):
    for j in range(steps):
      idx = i*steps+j
      full_img[(size+2)*i:(size+2)*i+size, (size+2)*j:(size+2)*j+size, :] = img_array[batch_idx, idx]

  plt.figure(figsize=(8,8))
  plt.imshow(full_img)


def Visualize_gMLP(i, model, **config):
    P = config['patch_size']
    norm = mcolors.DivergingNorm(vcenter=0.0)
    weights = model.layers[i].get_weights()[2].reshape((P, P, P, P))
    full_img = np.zeros(((P+2)*P-2, (P+2)*P-2))
    
    for i in range(P):
        for j in range(P):
            full_img[(P+2)*i:(P+2)*i+P, (P+2)*j:(P+2)*j+P] = weights[:, :, i, j]

    plt.figure(figsize=(10,10))
    plt.imshow(full_img, cmap='seismic', norm=norm)
    plt.axis("off")
