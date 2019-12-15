# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:38:22 2019

@author: Kenichi
"""

import os
import keras
import keras.backend.tensorflow_backend as KTF
from keras.layers import Conv2D, MaxPooling2D, Lambda, Input, Dense, Flatten, BatchNormalization
from keras.models import Model
from keras.layers.core import Dropout
from keras import optimizers
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau,TensorBoard, EarlyStopping
from keras.utils import multi_gpu_model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from keras.datasets import cifar10
import cv2
# import gc
import numpy as np

print(tf.__version__)
print(keras.__version__)

data = np.load("data5_100.npy")
label_array = np.load("label5_100.npy")

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(data, label_array, test_size=0.2)
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')

X_train /=256
y_train /=256

old_session = KTF.get_session()

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)
    
    inputs = Input(shape=(128, 128, 1))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)
    flattened = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(flattened)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    predictions = Dense(5, activation='softmax', name='predictions')(x)

    BATCH_SIZE = 256
    #sgd = optimizers.SGD(lr=0.01,
    #                     momentum=0.9,
    #                     decay=5e-4)#, nesterov=False)
    adam = keras.optimizers.Adam(lr=0.001, 
                                 beta_1=0.9, beta_2=0.999, epsilon=None, 
                                 decay=0.0, amsgrad=False)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    # This assumes that your machine has 8 available GPUs.
    parallel_model = multi_gpu_model(model, gpus=2)
    
    parallel_model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
#    # Train
    rlop = ReduceLROnPlateau(monitor='val_acc',
                             factor=0.1,
                             patience=5,
                             verbose=1,
                             mode='auto',
                             epsilon=0.0001,
                             cooldown=0,
                             min_lr=0.00001)
    
    #es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    #tb_cb = TensorBoard(log_dir="./tensorlog", histogram_freq=1)
     
    parallel_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=200, verbose=1,
                  callbacks=[rlop], validation_data=(X_test, y_test))
    parallel_model.save('VGG6_100.h5')
    y_pred = parallel_model.predict(X_test, verbose=1)
    score = parallel_model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
KTF.set_session(old_session)