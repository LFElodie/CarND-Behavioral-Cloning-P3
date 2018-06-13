
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, Cropping2D, Dropout
from keras.callbacks import EarlyStopping

from utils import load_data
from utils import training_generator, validation_generator

# set parameters
BATCH_SIZE = 64
EPOCHS = 15
PATIENCE = 5

# Three data folders contain driving forward/backward on track one
# and driving forward on track two.
data_folders = ['data/forward', 'data/back', 'data/track2']

# Read driving_log.csv from data folders
samples = load_data(data_folders)

# Split to training set and validation set
train_samples, valid_samples = train_test_split(samples, test_size=0.1)

print('training samples:', train_samples.shape[0])
print('validation samples:', valid_samples.shape[0])


BATCH_SIZE = 64
# generator for training and validation
train_generator = training_generator(train_samples, batch_size=BATCH_SIZE)
valid_generator = validation_generator(valid_samples, batch_size=BATCH_SIZE)


def nvidia_model():
    ''' 
    Define the model
    '''
    model = Sequential()
    # Crop the lower and higher parts of the image to ignore
    # the hood (bottom 20 pixels), sky/hills/trees (top 70 pixels)
    model.add(Cropping2D(cropping=((70, 20), (0, 0)),
                         input_shape=[160, 320, 3]))
    # Normalize the data
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    # Nvidia model
    # Conv layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # Flatten
    model.add(Flatten())
    # FC layers
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model

model = nvidia_model()
# Use Adam optimizer to minimize the mean squared error
model.compile(optimizer='adam', loss='mse')

# use early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)
# trian the model
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=train_samples.shape[0] // BATCH_SIZE,
                                     validation_data=valid_generator,
                                     validation_steps=valid_samples.shape[0] // BATCH_SIZE,
                                     epochs=EPOCHS,
                                     verbose=1,
                                     callbacks=[early_stopping])

model.save('model.h5')

# # print the keys contained in the history object
# print(history_object.history.keys())

# # plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
