import pandas as pd
import numpy as np 
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from tensorflow.keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense,AveragePooling2D
from tensorflow.keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math  
import datetime
import time
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix

print('Tensorflow_VER= V',tf.version.VERSION)
print(confusion_matrix)


# In[4]:


train_data_dir = '/root/Downloads/V/data/train/'  
test_data_dir = '/root/Downloads/V/data/test/'
validation_data_dir = '/root/Downloads/V/data/val/'  

#TUNING SEBAGIAN DISINI
batch_size = 2
lr=1e-4
opt='rmsprop'

img_width, img_height =224, 224  
   
top_model_weights_path = 'bottleneck_fc_model.h5' 
