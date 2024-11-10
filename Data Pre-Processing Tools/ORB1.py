import cv2
import numpy as np
from skimage import io, color
from skimage.feature import local_binary_pattern
import glob
count =1

filepath = glob.glob("/data/Lymph_Data/Output/Benign/*.jpg")

for files in filepath:
	# Reading the image and converting into B/W
	image = cv2.imread(files)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	  
	  
	# Applying the function
	orb = cv2.ORB_create(nfeatures=2000)
	kp, des = orb.detectAndCompute(gray_image, None)
	  
	# Drawing the keypoints
	kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)

	output = '/data/Lymph_Data/ORB/Benign/'+str(count)+'.jpg'
	cv2.imwrite(output, kp_image)
	count=count+1
	print(count)

