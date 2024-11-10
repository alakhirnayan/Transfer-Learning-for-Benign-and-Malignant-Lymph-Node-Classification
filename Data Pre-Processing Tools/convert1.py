import pydicom 
import matplotlib.pyplot as plt 
import scipy.misc 
import pandas as pd
import numpy as np
import os 
import glob
import imageio
from skimage.transform import resize
from PIL import Image



file_path = glob.glob("/data/Lymph_Data/Data/Malignant/*/*/*/*.dcm")
count =1
for files in file_path:

  out_path = "/data/Lymph_Data/Output/Malignant/"+str(count)+".jpg" 
  ds = pydicom.dcmread(files)
  img =  ds.pixel_array.astype(float)
  rescale_image = (np.maximum(img,0)/img.max())*255
  final_image = np.uint8(rescale_image)
  final_image = Image.fromarray(final_image)
  imageio.imwrite(out_path,final_image) 
  count=count+1
  
  print('Image Saved: ', count)



     

