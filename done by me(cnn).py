# -*- coding: utf-8 -*-
"""Copy of convolutional_neural_network.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tOeE_cK9-WEGkyfaKoqToEN1fvsGI3P9

# Convolutional Neural Network

### Importing the libraries
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

!wget "http://dl.dropboxusercontent.com/s/w9aqbqxmj4i2my8/dataset.zip"
!unzip dataset.zip
!ls

"""## Part 1 - Data Preprocessing

### Preprocessing the Training set
"""

train_data_gen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)#to prevent overfitting of data
train_set=train_data_gen.flow_from_directory('dataset/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

"""### Preprocessing the Test set"""

test_data_gen=ImageDataGenerator(rescale=1./255)
test_set=test_data_gen.flow_from_directory('dataset/test_set',
                                           target_size=(64,64),
                                           batch_size=32,
                                           class_mode='binary')

"""## Part 2 - Building the CNN

### Initialising the CNN
"""

cnn=tf.keras.models.Sequential()

"""### Step 1 - Convolution"""

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))

"""### Step 2 - Pooling"""

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

"""### Adding a second convolutional layer"""

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))#input shape only required when first time

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))#strides are steps where filter moves in convolutional layer

"""### Step 3 - Flattening"""

cnn.add(tf.keras.layers.Flatten())

"""### Step 4 - Full Connection"""

cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

"""### Step 5 - Output Layer"""

cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

"""## Part 3 - Training the CNN

### Compiling the CNN
"""

cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

"""### Training the CNN on the Training set and evaluating it on the Test set"""

import numpy as np

# Convert the test_set data into a NumPy array
test_data = np.array(test_set)

# Reshape the test_data
test_data = test_data.reshape((test_data.shape[0],) + train_set.shape[1:])

# Train the CNN model
cnn.fit(x=train_set, validation_data=test_data, epochs=25)

"""## Part 4 - Making a single prediction"""

import numpy as np
from keras.preprocessing import image
test_image=image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',
                          target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=cnn.predict(test_image)
train_set.class_indices
if result[0][0][0][0]==1:
  prediction='dog'
else:
  prediction='cat'
print(prediction)