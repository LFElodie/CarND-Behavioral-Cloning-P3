import pandas as pd
# import cv2
import matplotlib.image as mpimg
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# read samples split to train and validation
samples = pd.read_csv('data/2/driving_log.csv')
train_samples, valid_samples = train_test_split(samples,test_size=0.2)

def generator(samples, batch_size=32, valid = False):
	num_samples = samples.shape[0]
	while 1:
		samples = shuffle(samples)
		for offset in range(0, num_samples, batch_size//2):
			batch_samples = samples.iloc[offset:offset+batch_size]
			
			images = []
			angles = batch_samples['steering']
			for batch_sample in batch_samples['center']:
				path = 'data/2/IMG/' + batch_sample.split('/')[-1]
				image = mpimg.imread(path)
				images.append(image)
			if not valid:
				augmented_images, augmented_angles = [],[]
				for image, angle in zip(images,angles):
					augmented_images.append(image)
					augmented_angles.append(angle)
					augmented_images.append(np.fliplr(image))
					augmented_angles.append(-angle) 
				X_train = np.array(augmented_images)
				y_train = np.array(augmented_angles)
				yield shuffle(X_train,y_train)
			else:
				X_train = np.array(images)
				y_train = np.array(angles)
				yield shuffle(X_train,y_train)
BATCH_SIZE = 64
train_generator = generator(train_samples,batch_size=BATCH_SIZE)
valid_generator = generator(valid_samples,batch_size=BATCH_SIZE,valid=True)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import MaxPool2D, Conv2D, Cropping2D
import matplotlib.pyplot as plt
model = Sequential()
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=[160,320,3]))
model.add(Cropping2D(cropping=((70,20),(0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
history_object = model.fit_generator(train_generator,
	steps_per_epoch=train_samples.shape[0]//BATCH_SIZE,
	validation_data=valid_generator,
	validation_steps=valid_samples.shape[0]//BATCH_SIZE,
	epochs=5,
	verbose=2)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
