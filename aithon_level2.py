#!/usr/bin/env python
import numpy as np 
import pandas as pd 

import math
import numpy as np
import pandas as pd


import seaborn as sns
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
##
import sys, os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/source/')


#import source.classification  as cls


'''
The following function will be called to train and test your model.
The function name, signature and output type is fixed.
The first argument is file name that contain data for training.
The second argument is file name that contain data for test.
The function must return predicted values or emotion for each data in test dataset
sequentially in a list.
['sad', 'happy', 'fear', 'fear', ... , 'happy']
'''
# Preprocessing data
def preprocess_data(trainingcsv):
    df = pd.read_csv(trainingcsv)
    imgList = []
    for k in range(len(df)):
        x = df.iloc[k,1:] 
        imgs = np.array(x).reshape(48, 48, 1).astype('float32')
        imgList.append(imgs)
     # print(img_array.shape)
    img_array = np.stack(arrays=imgList,axis =0)
    return img_array
   
# Building a deep CNN model
def build_net(optim,img_width,img_height,img_depth,num_classes):
    """
    This is a Deep Convolutional Neural Network (DCNN). For generalization purpose I used dropouts in regular intervals.
    I used `ELU` as the activation because it avoids dying relu problem but also performed well as compared to LeakyRelu
    atleast in this case. `he_normal` kernel initializer is used as it suits ELU. BatchNormalization is also used for better
    results.
    """
    net = Sequential(name='DCNN')

    net.add(
        Conv2D(
            filters=64,
            kernel_size=(5,5),
            input_shape=(img_width, img_height, img_depth),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_1'
        )
    )
    net.add(BatchNormalization(name='batchnorm_1'))
    net.add(
        Conv2D(
            filters=64,
            kernel_size=(5,5),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_2'
        )
    )
    net.add(BatchNormalization(name='batchnorm_2'))
    
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_1'))
    net.add(Dropout(0.4, name='dropout_1'))

    net.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_3'
        )
    )
    net.add(BatchNormalization(name='batchnorm_3'))
    net.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_4'
        )
    )
    net.add(BatchNormalization(name='batchnorm_4'))
    
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_2'))
    net.add(Dropout(0.4, name='dropout_2'))

    net.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_5'
        )
    )
    net.add(BatchNormalization(name='batchnorm_5'))
    net.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
            name='conv2d_6'
        )
    )
    net.add(BatchNormalization(name='batchnorm_6'))
    
    net.add(MaxPooling2D(pool_size=(2,2), name='maxpool2d_3'))
    net.add(Dropout(0.5, name='dropout_3'))

    net.add(Flatten(name='flatten'))
        
    net.add(
        Dense(
            128,
            activation='elu',
            kernel_initializer='he_normal',
            name='dense_1'
        )
    )
    net.add(BatchNormalization(name='batchnorm_7'))
    
    net.add(Dropout(0.6, name='dropout_4'))
    
    net.add(
        Dense(
            num_classes,
            activation='softmax',
            name='out_layer'
        )
    )
    
    net.compile(
        loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy']
    )
    
    net.summary()
    
    return net
#
# python -c 'from emotion.py import *; aithon_level2_api("aithon2020_level2_traning.csv","aithon2020_level2_traning.csv")'
def  aithon_level2_api(traingcsv, testcsv):

    # The following dummy code for demonstration.
    df1 = pd.read_csv(traingcsv)
    img_array=preprocess_data(traingcsv)
    
    #cls.train_a_model(traingcsv)
    
    # Train the model with preprocessed data
    
    le = LabelEncoder()
    img_labels = le.fit_transform(df1.emotion)
    img_labels = np_utils.to_categorical(img_labels)
    img_labels.shape
    
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
    X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                    shuffle=True, stratify=img_labels,
                                                    test_size=0.1, random_state=42)
    X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
    del df1
    del img_array
    del img_labels
    img_width = X_train.shape[1]
    img_height = X_train.shape[2]
    img_depth = X_train.shape[3]
    num_classes = y_train.shape[1]
    
    X_train = X_train / 255.
    X_valid = X_valid / 255.
    
    """
    We're using two callbacks one is `early stopping` to avoid overfitting training data
    and `ReduceLROnPlateau` for learning rate.
    """

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00005,
        patience=2,
        verbose=1,
        restore_best_weights=True,
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1,
    )

    callbacks = [
        early_stopping,
        lr_scheduler,
    ]
    
    # As the data in hand is less as compared to the task, using keras' ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        brightness_range=[0.2,1.0],
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    train_datagen.fit(X_train)
    
    batch_size = 64 # we're going with a batch of 64.
    epochs = 33    
    train_datagen = ImageDataGenerator(
            rotation_range=30,
            brightness_range=[0.2,1.0],
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
        )
    train_datagen.fit(X_train)
    optims = [
        optimizers.Nadam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam'),
        optimizers.Adam(0.001),
    ]

    model = build_net(optims[1],img_width,img_height,img_depth,num_classes) 
    history = model.fit_generator(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_valid, y_valid),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=epochs,
        callbacks=callbacks,
        use_multiprocessing=True
    )
    
#     model_yaml = model.to_yaml()
#     with open("model.yaml", "w") as yaml_file:
#         yaml_file.write(model_yaml)

#     model.save("model.h5")

    # Test that model with test data
    # And return predicted emotions in a list
    
    test_X = preprocess_data(testcsv)    
    predictions = model.predict_classes(test_X)
    mapper = {
        0:'Fear', 1:'Happy', 2:'Sad'
    }
    res= []
    #print(np.unique(predictions))
    #print(mapper[yhat_valid])
    for i in predictions:
        #print(mapper[yhat_valid[i]])
        res.append(mapper[predictions[i]])
    #print(res)
    
    return res

# if __name__ == "__main__":
#     hello(sys.argv[2])
