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



file_path = glob.glob("/data/Lymph_Data/Data/Benign/*/*/*/*.dcm")
count =1
for files in file_path:

  out_path = "/data/Lymph_Data/Output/Benign/"+str(count)+".jpg" 
  ds = pydicom.read_file(files)
  img =  ds.pixel_array # extracting image information
  imageio.imwrite(out_path,img) 
  count=count+1
  
  print('Image Saved: ', count)



     

