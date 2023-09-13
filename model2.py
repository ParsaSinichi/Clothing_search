

import os
import cv2
import json
import time
import csv


import numpy as np
import tensorflow as tf

from keras.applications.vgg19 import VGG19
from keras.layers import Dropout, Flatten, Dense, Input, MaxPool2D, GlobalAveragePooling2D, Lambda, Conv2D, concatenate, ZeroPadding2D, Layer, MaxPooling2D , Flatten
from keras import backend as K
from keras.models import Model, load_model



def model_architecture():
    vgg_model = VGG19( include_top=False, input_shape=(224,224,3))
    convnet_output = GlobalAveragePooling2D()(vgg_model.output)
    convnet_output = Dense(4096, activation='relu')(convnet_output)
    convnet_output = Dropout(0.5)(convnet_output)
    convnet_output = Dense(4096, activation='relu')(convnet_output)
    convnet_output = Dropout(0.5)(convnet_output)
    convnet_output = Lambda(lambda  x: K.l2_normalize(x,axis=1))(convnet_output)
      
    s1_inp = Input(shape=(224,224,3))    
    s1 = MaxPool2D(pool_size=(4,4),strides = (4,4),padding='valid')(s1_inp)
    s1 = ZeroPadding2D(padding=(4, 4), data_format=None)(s1)
    s1 = Conv2D(96, kernel_size=(8, 8),strides=(4,4), padding='valid')(s1)
    s1 = ZeroPadding2D(padding=(2, 2), data_format=None)(s1)
    s1 = MaxPool2D(pool_size=(7,7),strides = (4,4),padding='valid')(s1)
    s1 = Flatten()(s1)

    s2_inp = Input(shape=(224,224,3))    
    s2 = MaxPool2D(pool_size=(8,8),strides = (8,8),padding='valid')(s2_inp)
    s2 = ZeroPadding2D(padding=(4, 4), data_format=None)(s2)
    s2 = Conv2D(96, kernel_size=(8, 8),strides=(4,4), padding='valid')(s2)
    s2 = ZeroPadding2D(padding=(1, 1), data_format=None)(s2)
    s2 = MaxPool2D(pool_size=(3,3),strides = (2,2),padding='valid')(s2)
    s2 = Flatten()(s2)
    
    merge_one = concatenate([s1, s2])
    merge_one_norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))(merge_one)
    merge_two = concatenate([merge_one_norm, convnet_output], axis=1)
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)
    
    final_model = tf.keras.models.Model(inputs=[s1_inp, s2_inp, vgg_model.input], outputs=l2_norm_final)

    return final_model


if __name__ == '__main__':
    model = model_architecture()
    model.load_weights('epoch11model.h5')
    x = tf.random.normal(shape=(1,224,224,3))
    y = model((x,x,x))
    print(y.shape)