import os
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
from glob import glob
import argparse
import predict
from model import *
from metrics import *
import SimpleITK as sitk




def load_latest_model():
  input_img = Input((128,128,16,4))
  model = Unet_3d(input_img, n_filters = 8, dropout = 0.1, batch_norm = True)
  model = load_model(".\model_save.hdf5",custom_objects = {'dice_coef_loss' : dice_coef_loss , 'dice_coef' : dice_coef, 'whole' : whole, 'core' : core, 'enhanc' : enhanc})
  lr = 0.00001
  decay_rate = 0.001
  model.compile(optimizer=Adam(learning_rate=lr, decay = decay_rate), loss=dice_coef_loss, metrics=[dice_coef,whole,core,enhanc], sample_weight_mode="temporal")

  return model



parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Path do folder containing the 4 brain MRI modalities in nii.gz format',action='store')
parser.add_argument('--output', type=str, help='Path to segmentation result',action='store')
args = parser.parse_args()



in_path=args.input#glob("C:\\Users\\seven\\Desktop\\Test\\Train18\\screenshots\\**")
out_path=args.output#"C:\\Users\\seven\\Desktop\\Test\\Train18\\screenshots_etiqueté\\"


#print(in_path)
#print(out_path)

model = load_latest_model()
pred = predict.predict_from_path(in_path,model)
pred = np.swapaxes(pred,0,2)
#print(pred.shape)
#plt.imshow(pred[:,:,60], cmap="gray")

#Save prediction
pred = np.argmax(pred,axis=-1)
pred[pred==3] = 4
predicted_image = np.array(pred.astype(np.uint8))
tmp=sitk.GetImageFromArray(predicted_image)
x =  out_path
sitk.WriteImage (tmp,x)

img = nib.load(x)
image_data = img.get_fdata()
image_data = np.asarray(image_data)
print(image_data.shape)

print(out_path)