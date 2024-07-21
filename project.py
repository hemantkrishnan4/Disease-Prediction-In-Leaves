import cv2
import numpy as np
import os
import unrar
from keras.models import load_model

import rarfile

# Open the zip file
#with rarfile.RarFile(f'D:/Hemant/Python/Projects/Cardamom/Cardamom_Plant_Dataset.rar', 'r') as rarfile_ref:
    # Extract all the files from the zip file
#    rarfile_ref.extractall('D:\Hemant\Python\Projects\Cardamom\Cardamom_Plant_Dataset')


from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.save('model.h5')



# Load the images and labels from the dataset
X = []
y = []
for label in ['healthy', 'leafspot', 'leafblight']:
	
	#List all the files in the directory
	for filename in os.listdir(f'D:/Hemant/Python/Projects/Cardamom/Cardamom_Plant_Dataset/{label}'):
		#Load the image and label
		image =cv2.imread(f'D:\Hemant\Python\Projects\Cardamom\Cardamom_Plant_Dataset/{label}/{filename}')
		label = 0 if label =='healthy' else 1 if label == 'leafspot' else 2
		
	


# Load the trained machine learning model
	model = load_model('model.h5')

# Use the webcam to capture an image of a cardamom plant leaf
	webcam = cv2.VideoCapture(0)
	_, image = webcam.read()
	webcam.release()

# Preprocess the image
	image = cv2.resize(image, (224, 224))
	image = np.expand_dims(image, axis=0)
	image = image / 255.0

# Use the machine learning model to classify the image
	prediction = model.predict(image)

# Output the classification result
	if prediction[0][0] > prediction[0][1] and prediction[0][0] > prediction[0][2]:
		print('Healthy')
	elif prediction[0][1] > prediction[0][0] and prediction[0][1] > prediction[0][2]:
		print('Leafspot')
	else:
		print('Leafblight')

		 


