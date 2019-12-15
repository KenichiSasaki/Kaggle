import keras
from keras.layers import Conv2D, MaxPooling2D, Lambda, Input, Dense, Flatten, BatchNormalization
from keras.models import Model
from keras.layers.core import Dropout
from keras import optimizers
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau,TensorBoard

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

data = np.load("data64_1000.npy")
label_array = np.load("label64_1000.npy")

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(data, label_array, test_size=0.2)

inputs = Input(shape=(64, 64, 1))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)
x = Conv2D(192, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x)
flattened = Flatten(name='flatten')(x)
x = Dense(1024, activation='relu', name='fc1')(flattened)
x = Dropout(0.5, name='dropout1')(x)
x = Dense(1024, activation='relu', name='fc2')(x)
x = Dropout(0.5, name='dropout2')(x)
predictions = Dense(4, activation='softmax', name='predictions')(x)

BATCH_SIZE = 512
#sgd = optimizers.SGD(lr=0.01,
#                     momentum=0.9,
#                     decay=5e-4)#, nesterov=False)
adam = keras.optimizers.Adam(lr=0.001, 
                             beta_1=0.9, beta_2=0.999, epsilon=None, 
                             decay=0.0, amsgrad=False)

model = Model(inputs=inputs, outputs=predictions)


model.compile(optimizer=adam,
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

 
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=50, verbose=1,
              callbacks=[rlop], validation_data=(X_test, y_test))

y_pred = model.predict(X_test, verbose=1)