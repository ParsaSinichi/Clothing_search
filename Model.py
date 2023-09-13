from keras.applications.vgg19 import VGG19
from keras.layers import Dropout, Flatten, Dense, Input, MaxPool2D, GlobalAveragePooling2D, Lambda, Conv2D, concatenate, ZeroPadding2D, Layer, MaxPooling2D, Flatten
from keras import backend as K
from keras.models import Model, load_model
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def model_architecture():
    '''
    Ref: https://github.com/gofynd/mildnet/blob/master/trainer/model.py
    This function contains model architecture which will be used to generate final embedding for each image.
    model(output): Final model object
    '''
    pre_trained_model = VGG19(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in pre_trained_model.layers[:-12]:
        layer.trainable = False

    convnet = GlobalAveragePooling2D()(pre_trained_model.output)
    convnet = Flatten()(convnet)

    convnet = Dense(128, activation='relu')(convnet)
    convnet = Dropout(0.5)(convnet)
    convnet = Dense(128, activation='relu')(convnet)

    convnet = Lambda(lambda x: K.l2_normalize(x, axis=1))(convnet)

    s1_inp = Input(shape=(224, 224, 3))
    s1 = MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(s1_inp)
    s1 = ZeroPadding2D(padding=(4, 4), data_format=None)(s1)
    s1 = Conv2D(96, kernel_size=(8, 8), strides=(4, 4), padding='valid')(s1)
    s1 = ZeroPadding2D(padding=(2, 2), data_format=None)(s1)
    s1 = MaxPool2D(pool_size=(7, 7), strides=(4, 4), padding='valid')(s1)
    s1 = Flatten()(s1)
    s1 = Dense(64)(s1)

    s2_inp = Input(shape=(224, 224, 3))
    s2 = MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding='valid')(s2_inp)
    s2 = ZeroPadding2D(padding=(4, 4), data_format=None)(s2)
    s2 = Conv2D(96, kernel_size=(8, 8), strides=(4, 4), padding='valid')(s2)
    s2 = ZeroPadding2D(padding=(1, 1), data_format=None)(s2)
    s2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(s2)
    s2 = Flatten()(s2)
    s2 = Dense(64)(s2)

    merge = concatenate([s1, s2])
    merge = Lambda(lambda x: K.l2_normalize(x, axis=1))(merge)
    merge = concatenate([merge, convnet], axis=1)
    embedding = Dense(256)(merge)

    model = tf.keras.models.Model(
        inputs=[s1_inp, s2_inp, pre_trained_model.input], outputs=embedding)
    return model


def contrastive_loss_function(q_emd, p_emd, n_emd, batch_size):
    '''
    Ref: https://github.com/gofynd/mildnet/blob/master/trainer/loss.py
    This function takes embedding generated by model for each of the image
    part of the triplet and return the loss value for the batch.
    q_emd(input): embedding generated by model for query image(tensor of size [batch, 4096])
    p_emd(input): embedding generated by model for query image(tensor of size [batch, 4096])
    n_emd(input): embedding generated by model for query image(tensor of size [batch, 4096])
    batch_size(input): batch size for each step
    loss(output): Final loss for a batch
    '''
    def _contrastive_loss(y_true, y_pred):
        return tfa.losses.contrastive_loss(y_true, y_pred)

    loss = tf.convert_to_tensor(0, dtype=tf.float32)
    g = tf.constant(1.0, shape=[1], dtype=tf.float32)
    h = tf.constant(0.0, shape=[1], dtype=tf.float32)

    for obs_num in range(batch_size):
        dist_query_pos = tf.sqrt(tf.reduce_sum(
            (q_emd[obs_num] - p_emd[obs_num])**2))
        dist_query_neg = tf.sqrt(tf.reduce_sum(
            (q_emd[obs_num] - n_emd[obs_num])**2))
        loss_query_pos = _contrastive_loss(g, dist_query_pos)
        loss_query_neg = _contrastive_loss(h, dist_query_neg)
        loss = loss + loss_query_pos + loss_query_neg

    loss = loss/(batch_size*2)
    zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
    return tf.maximum(loss, zero)


##
"""def contrastive_loss_function(q_emd, p_emd, n_emd, batch_size): CHATGPT OPTIMIZED 
    '''
    Ref: https://github.com/gofynd/mildnet/blob/master/trainer/loss.py
    This function takes embedding generated by model for each of the image
    part of the triplet and return the loss value for the batch.
    q_emd(input): embedding generated by model for query image(tensor of size [batch, 4096])
    p_emd(input): embedding generated by model for positive image(tensor of size [batch, 4096])
    n_emd(input): embedding generated by model for negative image(tensor of size [batch, 4096])
    batch_size(input): batch size for each step
    loss(output): Final loss for a batch
    '''
    def _contrastive_loss(y_true, y_pred):
        return tfa.losses.contrastive_loss(y_true, y_pred)

    dist_query_pos = tf.sqrt(tf.reduce_sum((q_emd - p_emd)**2, axis=1))
    dist_query_neg = tf.sqrt(tf.reduce_sum((q_emd - n_emd)**2, axis=1))
    loss_query_pos = _contrastive_loss(tf.ones(batch_size), dist_query_pos)
    loss_query_neg = _contrastive_loss(tf.zeros(batch_size), dist_query_neg)
    loss = tf.reduce_mean(loss_query_pos + loss_query_neg)
    zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
    return tf.maximum(loss, zero)   """


def accuracy(q_emd, p_emd, n_emd, batch_size):
    '''
    Ref: https://github.com/gofynd/mildnet/blob/master/trainer/accuracy.py
    This function takes in embedding and return the accuracy value for the batch
    q_emd(input): embedding generated by model for query image(tensor of size [batch, 4096])
    p_emd(input): embedding generated by model for query image(tensor of size [batch, 4096])
    n_emd(input): embedding generated by model for query image(tensor of size [batch, 4096])
    batch_size(input): batch size for each step
    accuracy(output): Final accuracy value for a batch
    '''
    accuracy = 0
    for obs_num in range(batch_size):
        dist_query_pos = tf.sqrt(tf.reduce_sum(
            (q_emd[obs_num] - p_emd[obs_num])**2))
        dist_query_neg = tf.sqrt(tf.reduce_sum(
            (q_emd[obs_num] - n_emd[obs_num])**2))
        accuracy += tf.cond(dist_query_neg > dist_query_pos,
                            lambda: 1, lambda: 0)

    return (accuracy * 100) / batch_size
def cosine_distance(x, y):
  '''
  This function inputs 2 vector and calculates cosine distance.
  '''
  return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))
def get_neighbours(train, query_embedding, k):
  '''
  This function finds the NN based on the cosine distance, it return top k NNs
  '''
  distance = [(index, cosine_distance(row, query_embedding)) for index, row in enumerate(train)]
  print(f" this is index  and this is distance {distance}")
  distance.sort(key = lambda x : x[1], reverse=True)
  print(f" this is index  and this is distance {distance}")

  return distance[0:k]