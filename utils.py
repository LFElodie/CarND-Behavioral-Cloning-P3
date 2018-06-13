import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.image as mpimg

CORRECTION = 0.2


def load_data(data_folders):
    '''
    Read driving_log.csv from data folders 
    '''
    samples = []
    for data_folder in data_folders:
        # Raw sample contains [center,left,right,steering,throttle,brake,speed] each row
        raw_sample = pd.read_csv(os.path.join(data_folder, 'driving_log.csv'))
        # Extract filenames and convert to right image path
        image_path = raw_sample.iloc[:, :3].applymap(
            lambda x: os.path.join(data_folder, 'IMG', os.path.basename(x)))
        sample = pd.concat([image_path, raw_sample.iloc[:, 3]], axis=1)
        samples.append(sample)
    samples = pd.concat(samples, axis=0)
    return samples


def training_generator(samples, batch_size=32):
    '''
    Generator for training data
    '''
    num_samples = samples.shape[0]
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset + batch_size]

            images = []
            angles = []
            for index, batch_sample in batch_samples.iterrows():
                # read in images from center, left and right cameras
                for i in range(3):
                    image = mpimg.imread(batch_sample[i])
                    images.append(image)
                # steering center
                angles.append(batch_sample[3])
                # steering left
                angles.append(batch_sample[3] + CORRECTION)
                # steering right
                angles.append(batch_sample[3] - CORRECTION)

            # data augment flip the image
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                # flip the image only if angle is larger than 0.1
                if angle > 0.1:
                    augmented_images.append(np.fliplr(image))
                    augmented_angles.append(-angle)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield shuffle(X_train, y_train)


def validation_generator(samples, batch_size=32):
    '''
    Generator for validation data
    '''
    num_samples = samples.shape[0]
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset + batch_size]
            images = []
            angles = []
            for index, batch_sample in batch_samples.iterrows():
                image = mpimg.imread(batch_sample[0])
                images.append(image)
                angles.append(batch_sample[3])
            X_train = np.array(images)
            y_train = np.array(images)

            yield X_train, y_train
