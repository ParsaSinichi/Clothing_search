import json
import sys
import numpy as np
import tensorflow as tf
import glob
import os
import torch
import cv2
from matplotlib import pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.layers import Dropout, Flatten, Dense, Input, MaxPool2D, GlobalAveragePooling2D, Lambda, Conv2D, concatenate, ZeroPadding2D, Layer, MaxPooling2D , Flatten
from keras import backend as K
from keras.models import Model, load_model
from model import  model_architecture

from detection import cropped_images

def dataset(path):
  #if database json 

    with open(path , 'r') as f:
        data = json.load(f)

    imagePath = []
    embedding = []
    for key , value in data.items():
        imagePath.append(key)
        embedding.append(np.array(value))
    

    return embedding , imagePath

def query_embedding(model , query_image_path):

  #pass the model for bot
  
  #query_image_path = os.path.join(imagepath , query_image_path)
  #img = tf.image.decode_jpeg(tf.io.read_file(query_image_path), channels=3)
  img = tf.image.resize(query_image_path, [224, 224])
  img = tf.reshape(img, [1,224,224,3])
  return model((img, img, img))



def cosine_distance(x, y):

  return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))


def get_neighbours(train, query_embedding, k):

  distance = [(index, cosine_distance(row, query_embedding)) for index, row in enumerate(train)]
  #print(f" this is index  and this is distance {distance}")
  distance = [(i, d) for i, d in distance if d > 0.90]
  distance.sort(key = lambda x : x[1], reverse=True)
  

  return distance[0:k]






if __name__ == '__main__':

    model = model_architecture()
    model.load_weights('epoch11model.h5')

    embedding , imagePath = dataset(path='data.json')
    
    image_query = glob.glob("*.jpg")
    #print(cropped_images(image_query[0])[0])
    img = tf.convert_to_tensor(cropped_images(image_query[0])[0])
    img = tf.cast(img , dtype=tf.float32)
    lst = get_neighbours(embedding, query_embedding(img).numpy()[0], 4)
    
    extracted_items = [imagePath[item[0]] for item in lst]
    
    print(extracted_items)
    fig=plt.figure(figsize=(15, 20))
    image = tf.image.decode_jpeg(tf.io.read_file(image_query[0]), channels=3)
    subplot_index = 1
    fig.add_subplot(1, len(extracted_items)+1, subplot_index)
    plt.axis('off')
    plt.imshow(image)
    subplot_index += 1
    for item in extracted_items:
        img = tf.image.decode_jpeg(tf.io.read_file(item), channels=3)
        fig.add_subplot(1, len(extracted_items)+1, subplot_index)
        plt.axis('off')
        plt.imshow(img)
        subplot_index += 1

    plt.axis('off')
    plt.show()
