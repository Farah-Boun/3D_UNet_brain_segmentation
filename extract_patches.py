import json
import numpy as np
import os
import time
import sys
import nibabel as nib
import h5py
import random
from glob import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#from nipype.interfaces.ants import N4BiasFieldCorrection
import SimpleITK as sitk
import warnings
from shutil import copyfile


def get_image_array(i, patient_list):
  patient_file = patient_list[i]
  print(patient_file, end="")
  folder_path = path + '/' + patient_file
  #print("Loading image...")
  data = np.zeros((240,240,155,4))
  image_label=np.zeros((240,240,155))

  modalities = os.listdir(folder_path)
  modalities.sort()
  #print(modalities)
  w = 0
  for j in range(len(modalities)):
    
    image_path = folder_path + '/' + modalities[j]
    if (image_path[-10:] == 'seg.nii.gz'):
      img = nib.load(image_path)
      image_label = img.get_fdata()
      image_label = np.asarray(image_label)
    else:
      img = nib.load(image_path)
      image_data = img.get_fdata()
      image_data = np.asarray(image_data)
      image_data = standardize(image_data)
      data[:,:,:,w] = image_data
      w = w+1

  image_label[image_label==4] = 3
  #print("Image loaded")
  return data, image_label



def save_patch_on_drive(image_train, label_train, patient_list, i,patch_num, patch_type="right",train = True):
  if (train): 
    filename =  "drive/MyDrive/BraTS_dataset/BraTS2021/128_16_patches/train/"+patient_list[i]+"_"+patch_type+"_num_"+str(patch_num)+".h5"
  else : 
    filename =  "drive/MyDrive/BraTS_dataset/BraTS2021/128_16_patches/valid/"+patient_list[i]+"_"+patch_type+"_num_"+str(patch_num)+".h5"
  with h5py.File(filename, "w") as f:
      dstx = f.create_dataset("X",data = image_train)
      dsty = f.create_dataset("y",data = label_train)
      f.close()


def generate_image_patches(data, image_label, patient_list, i, train = True):
  #generate regular patch 
  reshaped_data=data[56:184,80:208,13:141,:]
  reshaped_image_label=image_label[56:184,80:208,13:141]
  reshaped_data=reshaped_data.reshape(128,128,128,4)
  reshaped_image_label=reshaped_image_label.reshape(128,128,128)
  print("generate patches")
  drawProgressBar(0)
  
  j=0
  while (j<5):
    if (j<4) : 
      image, label = get_sub_volume_xyz(reshaped_data, reshaped_image_label)
      save_patch_on_drive(image,label, patient_list, i,j, "tum_right",train)
    else :
      image, label = get_sub_volume_xyz_no_tumor(reshaped_data, reshaped_image_label)
      save_patch_on_drive(image,label, patient_list, i,0, "notum_right",train)
    j = j + 1
    drawProgressBar(0.1*j)
  

  #generate left patch 
  reshaped_data=data[56:184,50:178,13:141,:]
  reshaped_image_label=image_label[56:184,50:178,13:141]
  reshaped_data=reshaped_data.reshape(128,128,128,4)
  reshaped_image_label=reshaped_image_label.reshape(128,128,128)
  j=0
  while (j<5):
    if (j<4) : 
      image, label = get_sub_volume_xyz(reshaped_data, reshaped_image_label)
      save_patch_on_drive(image,label, patient_list, i,j, "tum_left",train)
    else :
      image, label = get_sub_volume_xyz_no_tumor(reshaped_data, reshaped_image_label)
      save_patch_on_drive(image,label, patient_list, i,1, "notum_left",train)
    j = j + 1
    drawProgressBar(0.1*j*2)
  

def get_sub_volume_xyz(image, label, 
                   orig_x = 128, orig_y = 128, orig_z = 128, 
                   output_x = 128, output_y = 128, output_z = 16,
                   num_classes = 4, max_tries = 1000, 
                   background_threshold=0.95):
    """
    Extract random sub-volume from original images.

    Args:
        image (np.array): original image, 
            of shape (orig_x, orig_y, orig_z, num_channels)
        label (np.array): original label. 
            labels coded using discrete values rather than
            a separate dimension, 
            so this is of shape (orig_x, orig_y, orig_z)
        orig_x (int): x_dim of input image
        orig_y (int): y_dim of input image
        orig_z (int): z_dim of input image
        output_x (int): desired x_dim of output
        output_y (int): desired y_dim of output
        output_z (int): desired z_dim of output
        num_classes (int): number of class labels
        max_tries (int): maximum trials to do when sampling
        background_threshold (float): limit on the fraction 
            of the sample which can be the background

    returns:
        X (np.array): sample of original image of dimension 
            (num_channels, output_x, output_y, output_z)
        y (np.array): labels which correspond to X, of dimension 
            (num_classes, output_x, output_y, output_z)
            
    """
    # Initialize features and labels with `None`
    X = None
    y = None

    tries = 0    
    while tries < max_tries:
        #trying to hack this to fix cases where the threshold is too much 
        if tries >900:
          background_threshold = 0.98
        if tries > 990:
          background_threshold = 0.999
        # randomly sample sub-volume by sampling the corner voxel
        # hint: make sure to leave enough room for the output dimensions!
        start_x = np.random.randint(0, orig_x - output_x+1)
        start_y = np.random.randint(0, orig_y - output_y+1)
        start_z = np.random.randint(0, orig_z - output_z+1)

        # extract relevant area of label
        y = label[start_x: start_x + output_x,
                  start_y: start_y + output_y,
                  start_z: start_z + output_z]
        
       
        # One-hot encode the categories.
        # This adds a 4th dimension, 'num_classes'
        # (output_x, output_y, output_z, num_classes)
        y_tmp = to_categorical(y, num_classes=num_classes)
       
        # compute the background ratio
        bgrd_ratio = np.sum(y_tmp[:, :, :, 0])/(output_x * output_y * output_z)

        # increment tries counter
        tries += 1

        # if background ratio is below the desired threshold,
        # use that sub-volume.
        # otherwise continue the loop and try another random sub-volume
        if bgrd_ratio < background_threshold:

            # make copy of the sub-volume
            X = np.copy(image[start_x: start_x + output_x,
                              start_y: start_y + output_y,
                              start_z: start_z + output_z, :])
           
            return X, y

    # if we've tried max_tries number of samples
    # Give up in order to avoid looping forever.
    print(f"Tried {tries} times to find a sub-volume. Giving up...")


def get_sub_volume_xyz_no_tumor(image, label, 
                   orig_x = 128, orig_y = 128, orig_z = 128, 
                   output_x = 128, output_y = 128, output_z = 16,
                   num_classes = 4, max_tries = 1000, 
                   background_threshold=0.95):
    """
    Extract random sub-volume from original images.

    Args:
        image (np.array): original image, 
            of shape (orig_x, orig_y, orig_z, num_channels)
        label (np.array): original label. 
            labels coded using discrete values rather than
            a separate dimension, 
            so this is of shape (orig_x, orig_y, orig_z)
        orig_x (int): x_dim of input image
        orig_y (int): y_dim of input image
        orig_z (int): z_dim of input image
        output_x (int): desired x_dim of output
        output_y (int): desired y_dim of output
        output_z (int): desired z_dim of output
        num_classes (int): number of class labels
        max_tries (int): maximum trials to do when sampling
        background_threshold (float): limit on the fraction 
            of the sample which can be the background

    returns:
        X (np.array): sample of original image of dimension 
            (num_channels, output_x, output_y, output_z)
        y (np.array): labels which correspond to X, of dimension 
            (num_classes, output_x, output_y, output_z)
            
    """
    # Initialize features and labels with `None`
    X = None
    y = None

    tries = 0    
    while tries < max_tries:
        #trying to hack this to fix cases where the threshold is too much 
        if tries >900:
          background_threshold = 0.98
        if tries > 990:
          background_threshold = 0.999
        # randomly sample sub-volume by sampling the corner voxel
        # hint: make sure to leave enough room for the output dimensions!
        start_x = np.random.randint(0, orig_x - output_x+1)
        start_y = np.random.randint(0, orig_y - output_y+1)
        start_z = np.random.randint(0, orig_z - output_z+1)

        # extract relevant area of label
        y = label[start_x: start_x + output_x,
                  start_y: start_y + output_y,
                  start_z: start_z + output_z]
        
       
        # One-hot encode the categories.
        # This adds a 4th dimension, 'num_classes'
        # (output_x, output_y, output_z, num_classes)
        y_tmp = to_categorical(y, num_classes=num_classes)
       
        # compute the background ratio
        bgrd_ratio = np.sum(y_tmp[:, :, :, 0])/(output_x * output_y * output_z)

        # increment tries counter
        tries += 1

        # if background ratio is below the desired threshold,
        # use that sub-volume.
        # otherwise continue the loop and try another random sub-volume
        if bgrd_ratio > background_threshold:

            # make copy of the sub-volume
            X = np.copy(image[start_x: start_x + output_x,
                              start_y: start_y + output_y,
                              start_z: start_z + output_z, :])
           
            return X, y

    # if we've tried max_tries number of samples
    # Give up in order to avoid looping forever.
    print(f"Tried {tries} times to find a sub-volume. Giving up...")


def drawProgressBar(percent, barLen = 20):
    # percent float from 0 to 1. 
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(barLen * percent), barLen, percent * 100))
    if (percent == 1):
      sys.stdout.write("  Patches generated")
    sys.stdout.flush()


def standardize(image):

  standardized_image = np.zeros(image.shape)
  
      # iterate over the z dimension
  for z in range(image.shape[2]):
      # get a slice of the image 
      # at channel c and z-th dimension z
      image_slice = image[:,:,z]

      # subtract the mean from image_slice
      centered = image_slice - np.mean(image_slice)
      
      # divide by the standard deviation (only if it is different from zero)
      if(np.std(centered)!=0):
          centered = centered/np.std(image) 

      # update  the slice of standardized image
      # with the scaled centered and scaled image
      standardized_image[:, :, z] = centered

  return standardized_image
'''

with open("drive/MyDrive/BraTS_dataset/BraTS2021/patient_list.json") as json_file:
        patient_list = json.load(json_file)

train_patient_list = patient_list['train_list']
valid_patient_list = patient_list['valid_list']

#Paths for Brats2020 dataset
path = 'drive/MyDrive/BraTS_dataset/BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData'

while (i<len(train_patient_list)):
  print(" ")
  print("patient n°"+str(i))
  image,label = get_image_array(i,train_patient_list)
  generate_image_patches(image,label,train_patient_list,i, train= True)
  i = i + 1
  '''
