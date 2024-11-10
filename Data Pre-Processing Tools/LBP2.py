import cv2
import numpy as np
from skimage import io, color
from skimage.feature import local_binary_pattern
import glob
count =1
filepath = glob.glob("/data/Lymph_Data/Output/Malignant/*.jpg")
for files in filepath:
	image = cv2.imread(files)
	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img2 = local_binary_pattern(img, 24, 1, method='default')
	output = '/data/Lymph_Data/LBP/Malignant/'+str(count)+'1.jpg'
	cv2.imwrite(output, img2)
	count=count+1
	print(count)
