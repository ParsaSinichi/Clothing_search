
import csv 
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tqdm import tqdm
import sys
import json
import sqlite3

from dataset import insert_name_image
from model import model_architecture




##################    assert len(image) % batch_size == 0
BATCH_SIZE = 2


def get_image_path(path):
  '''
  This function is used in our case to read the data that we have in our test set
  It will read the test csv file that we have, return 2 separate list, one with 
  query images and other with images that we already have in system.
  '''
  query_image = []
  
  
  
  
  img_path = "C:/Users/Farhang/Desktop/visual-search/orginal_imgs/"
  with open(path , 'r') as csv_file:
    data = csv.reader(csv_file, delimiter = ',')
    
    for row in data:
      
      query_image.append(img_path + row[0])

  return query_image, list(set(query_image))


def parser(input_tensor):
  '''
  This function can be used as a parser function for all the images we have that
  will be recommended to the user when the user will select a query image
  '''
  img = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(input_tensor), channels=3), [224, 224])
  print(img)
  return img



def get_saved_images_embedding(data_path, batch_size = BATCH_SIZE):
  '''
  This function is used to generate a pipeline that will be used to create a batch.
  Main purpose of having this is to overcome the RAM issue faced since we are using 
  everything on colab, in real world all our images will be stored in a bucket from where
  we can directly read them from their and pass on to model function to generate the 
  embedding which will be stored in a cache DB for low latency
  '''
  #####
  name_img_anch = data_path
  
  dataset = tf.data.Dataset.from_tensor_slices((np.array((data_path))))
  dataset = dataset.map(parser, num_parallel_calls = tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size)
  return  dataset


def final_embed(model , path_anchor:str):
    global BATCH_SIZE
    
    model = model 
    
    query_lst, stored_images_list = get_image_path(path_anchor)

    
    saved_images_data = get_saved_images_embedding(stored_images_list)
    test_iterator = iter(saved_images_data)
    
    # Generate embeddings for test data
    test_data_lst = []
    test_iterator = iter(saved_images_data)
    for i in tqdm(range(len(stored_images_list) // BATCH_SIZE)):
        test_data = test_iterator.get_next()
        test_data_lst.append(model((test_data, test_data, test_data)))
    
    # Concatenate embeddings and return
    final_embedding = tf.concat(test_data_lst, axis = 0)
    del test_data_lst
    
    return final_embedding , stored_images_list 


def insert_value(data):
  

  conn = sqlite3.connect('database.db')
  c = conn.cursor()

  insert_query = f"INSERT INTO product(embedding, filename) VALUES (?, ?);"

  
  c.executemany(insert_query, data)

  conn.commit()
  conn.close()


def get_embeddings_dataset(embedding, image_paths):
    #ready the embedding for execut database 
    embeddings_list = []
    for i in range(len(image_paths)):
        embeddings_list.append((sqlite3.Binary(embedding[i]) , image_paths[i]))
    return embeddings_list

import sys





if __name__ == '__main__':
    
    model = model_architecture()
    model.load_weights('epoch11model.h5')
    final , stored_images_list  = final_embed(model, path_anchor=r'C:\Users\Farhang\Desktop\visual-search\model\preprocessing\testAnchor.csv')
   
    
    #sys.exit()
    embeddings_list = get_embeddings_dataset(final.numpy(), stored_images_list)
    
    insert_value(embeddings_list)
    
    
    # with open("data.json", "w") as outfile:
    #     json.dump(embeddings_dict, outfile)
    assert len(stored_images_list) % BATCH_SIZE == 0
    # data = []
    # for i in range(len(stored_images_list)):
    #   path = stored_images_list[i].split("/")[-1]
    #   tup = (i , path)
    #   data.append(tup)
    
    # insert_name_image(data)

