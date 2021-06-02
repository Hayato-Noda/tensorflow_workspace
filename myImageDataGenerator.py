#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class PatchImageDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def flow(self, *args, **kwargs):
        batches = super().flow(*args, **kwargs)
        while True:
            batch_x, batch_y = next(batches)
            yield (batch_x, batch_y)
            
    def flow_from_directory(self, *args, **kwargs):
        batches = super().flow_from_directory(*args, **kwargs)
        while True:
            batch_x, batch_y = next(batches)
            yield (batch_x, batch_y)


class get_dataset():
  def __init__(self, **config):
    super().__init__()
    self.class_num = config['num_classes']
    self.batch_size = config['batch_size']
    self.dataname = config['dataset']

  def _cifar10(self):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')/255.
    y_train = to_categorical(y_train, self.class_num)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    return x_train, x_val, y_train, y_val

  def _cifar100(self):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')/255.
    y_train = to_categorical(y_train, self.class_num)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    return x_train, x_val, y_train, y_val
    
  def setting(self):
    if self.dataname == 'CIFAR10':
      x_train, x_val, y_train, y_val = self._cifar10()
    else:
       x_train, x_val, y_train, y_val = self._cifar100()
       
    return x_train, x_val, y_train, y_val

