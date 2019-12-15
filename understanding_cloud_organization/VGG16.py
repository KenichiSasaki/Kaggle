# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:25:28 2019

@author: Kenichi
"""

import keras
from keras.layers import Conv2D, MaxPooling2D, Lambda, Input, Dense, Flatten, BatchNormalization
from keras.models import Model
from keras.layers.core import Dropout
from keras import optimizers
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau,TensorBoard, EarlyStopping
from keras.utils import multi_gpu_model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import cv2
# import gc
import numpy as np

print(tf.__version__)
print(keras.__version__)

data = np.load("data512_1000.npy")
label_array = np.load("label512_1000.npy")

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(data, label_array, test_size=0.2)

# inputs = Input(shape=(224, 224, 3))
inputs = Input(shape=(512, 512, 1))
x = Lambda(lambda image: tf.image.resize_images(image, (224, 224)))(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = BatchNormalization()(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x)
flattened = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(flattened)
x = Dropout(0.5, name='dropout1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.5, name='dropout2')(x)
predictions = Dense(5, activation='softmax', name='predictions')(x)

BATCH_SIZE = 4
#sgd = optimizers.SGD(lr=0.01,
#                     momentum=0.9,
#                     decay=5e-4)#, nesterov=False)
adam = keras.optimizers.Adam(lr=0.001, 
                             beta_1=0.9, beta_2=0.999, epsilon=None, 
                             decay=0.0, amsgrad=False)
model = Model(inputs=inputs, outputs=predictions)

# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=1)
parallel_model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train
rlop = ReduceLROnPlateau(monitor='val_acc',
                         factor=0.1,
                         patience=5,
                         verbose=1,
                         mode='auto',
                         epsilon=0.0001,
                         cooldown=0,
                         min_lr=0.00001)
es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
 
parallel_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=200, verbose=1,
              callbacks=[rlop, es_cb], validation_data=(X_test, y_test))

parallel_model.save('VGG256_1000.h5')
y_pred = parallel_model.predict(X_test, verbose=1)
score = parallel_model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

predict_class = parallel_model.predict(X_test, verbose=1)
predict_class = np.argmax(predict_class, axis=1)
true_class = np.argmax(y_test, axis=1)
cmx = confusion_matrix(true_class, predict_class)
report = classification_report(true_class, predict_class)
np.save('cmx.npy', cmx)
np.save('report.npy', report)
print('Confusion Matrix: ', cmx)
print('Classification Sumarry: ', report)