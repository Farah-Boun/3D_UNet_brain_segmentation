import random
import datetime
from glob import glob
import h5py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import keras
import nibabel as nib
from keras.utils import np_utils
from keras.callbacks import  ModelCheckpoint,Callback,EarlyStopping,CSVLogger
from PIL import Image

 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K 
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    UpSampling3D,
    Reshape,
    BatchNormalization, Dense,
    Dropout,
    Maximum
)
from tensorflow.keras.layers import concatenate, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import os
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from sklearn.utils import class_weight
import gc


from metrics import *


def decision(probability):
    return random.random() < probability

#Added a save feature so we can continue training if it freezes, also removed the shuffle so that can be possible
class VolumeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                sample_list,
                 base_dir,
                 batch_size=1,
                 shuffle=True,
                 dim=(128, 128, 128),
                 num_channels=4,
                 num_classes=4,
                 verbose=1):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.dim = dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.verbose = verbose
        self.sample_list = sample_list
        self.on_epoch_end()
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.sample_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.sample_list) / self.batch_size))
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        gc.collect()
        X = np.zeros((self.batch_size, *self.dim,self.num_channels),
                     dtype=np.float64)
        y = np.zeros((self.batch_size, *self.dim,self.num_classes),
                     dtype=np.float64)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.verbose == 1:
                print("Training on: %s" % self.base_dir + ID)
 
            
            with h5py.File(self.base_dir + ID, 'r') as f:
                X[i] = np.array(f.get("X"))
                label = np.array(f.get("y"))
                if (not decision(0.7)) :
                  X[i] = np.flipud(X[i])
                  label = np.flipud(label)
                label = to_categorical(label, num_classes = 4)
                y[i] = label
                f.close()
        
        return X, y
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size: (index + 1) * self.batch_size]
        # Find list of IDs
        sample_list_temp = [self.sample_list[k] for k in indexes]
        # Generate data
        
        X, y = self.__data_generation(sample_list_temp)
        vector_label = y.flatten()
        class_weights = class_weight.compute_class_weight('balanced',np.unique(vector_label),vector_label)
        sample_weights = generate_sample_weights(y, class_weights)
       
        return X, y, sample_weights


def conv_block(input_mat,num_filters,kernel_size,batch_norm):
  X = Conv3D(num_filters,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(input_mat)
  if batch_norm:
    X = BatchNormalization()(X)
  
  X = Activation('relu')(X)
 
  X = Conv3D(num_filters,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(X)
  if batch_norm:
    X = BatchNormalization()(X)
  
  X = Activation('relu')(X)
  
  return X
 
 
def Unet_3d(input_img, n_filters = 8, dropout = 0.2, batch_norm = True):
 
  c1 = conv_block(input_img,n_filters,3,batch_norm)
  p1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(c1)
  p1 = Dropout(dropout)(p1)
  
  c2 = conv_block(p1,n_filters*2,3,batch_norm)
  p2 = MaxPooling3D(pool_size=(2,2,2) ,strides=2)(c2)
  p2 = Dropout(dropout)(p2)
 
  c3 = conv_block(p2,n_filters*4,3,batch_norm)
  p3 = MaxPooling3D(pool_size=(2,2,2) ,strides=2)(c3)
  p3 = Dropout(dropout)(p3)
  
  c4 = conv_block(p3,n_filters*8,3,batch_norm)
  p4 = MaxPooling3D(pool_size=(2,2,2) ,strides=2)(c4)
  p4 = Dropout(dropout)(p4)
  
  c5 = conv_block(p4,n_filters*16,3,batch_norm);
 
  u6 = Conv3DTranspose(n_filters*8, (3,3,3), strides=(2, 2, 2), padding='same')(c5)
  u6 = concatenate([u6,c4])
  c6 = conv_block(u6,n_filters*8,3,batch_norm)
  c6 = Dropout(dropout)(c6)
 
  u7 = Conv3DTranspose(n_filters*4,(3,3,3),strides = (2,2,2) , padding= 'same')(c6)
  u7 = concatenate([u7,c3])
  c7 = conv_block(u7,n_filters*4,3,batch_norm)
  c7 = Dropout(dropout)(c7)
 
  u8 = Conv3DTranspose(n_filters*2,(3,3,3),strides = (2,2,2) , padding='same')(c7)
  u8 = concatenate([u8,c2])
  c8 = conv_block(u8,n_filters*2,3,batch_norm)
  c8 = Dropout(dropout)(c8)
 
  u9 = Conv3DTranspose(n_filters,(3,3,3),strides = (2,2,2) , padding='same')(c8)
  u9 = concatenate([u9,c1])
 
  c9 = conv_block(u9,n_filters,3,batch_norm)
  outputs = Conv3D(4, (1, 1,1), activation='softmax')(c9)
  
  print(outputs.shape)
  model = Model(inputs=input_img, outputs=outputs)
  print("Model created")
  return model
 

def generate_sample_weights(training_data, class_weights): 
#replaces values for up to 4 classes with the values from class_weights#
    if (len(class_weights) == 4):
        sample_weights = [np.where(y==0,class_weights[0],
                            np.where(y==1,class_weights[1],
                            np.where(y==2,class_weights[2],
                            np.where(y==3,class_weights[3],y)))) for y in training_data]
    if (len(class_weights) == 3):
        sample_weights = [np.where(y==0,class_weights[0],
                            np.where(y==1,class_weights[1],
                            np.where(y==2,class_weights[2],y))) for y in training_data]
    if (len(class_weights) == 2):
        sample_weights = [np.where(y==0,class_weights[0],
                            np.where(y==1,class_weights[1],y)) for y in training_data]                                                      
    return np.asarray(sample_weights)


def scheduler(epoch, lr):
  if epoch < 1:
    return lr
  else:
    print("Learning rate changed to: "+ str(lr * tf.math.exp(-0.1)))
    return lr * tf.math.exp(-0.1)