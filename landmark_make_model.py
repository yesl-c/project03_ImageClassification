# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:55:47 2020

@author: Seoul IT
"""


#%% 00 라이브러리 호출

import sys, os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.layers import LeakyReLU
import numpy as np

#%% 01 경로 지정 및 기타

root_dir = './landmark/'

categories = [
"Eiffel Tower"
,"Cristo Redentor"
,"Triumphal Arch"
,"Wànlĭ Chángchéng"
,"Taj Mahal"
,"Sungnyemun Gate"
,"Moai"
,"Seokguram buddha"
,"Golden Gate"
,"Statue of Liberty"
,"Torre di Pisa"
,"Colosseum"
,"Santiago Bernabéu"
,"Sphinx"
,"Burj Khalifa"
,"London Eye"
,"London Tower Bridge" 
] 

nb_classes = len(categories)

image_size = 256

#%% 02 데이터 로딩 함수 생성

def load_dataset():
    x_train, x_test, y_train, y_test = np.load('./landmark/landmark.npy',
                                               allow_pickle=True)
    print(x_train.shape, y_train.shape)
    
    x_train = x_train.astype('float') / 256
    x_test = x_test.astype('float') / 256
    
    y_train = np_utils.to_categorical(y_train, nb_classes)   # one-hot-encoding
    y_test = np_utils.to_categorical(y_test, nb_classes)
    
    return x_train, x_test, y_train, y_test

load_dataset()

#%% 03 모델 구성 함수 생성

def build_model(in_shape):
    model = Sequential()
    
    model.add(Convolution2D(32,3,3, border_mode='Same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64,3,3, border_mode='Same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128,3,3, border_mode='Same'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

#%% 04 모델 학습 함수 생성

def model_train(x, y):
    print(x.shape[1:])
    model = build_model(x.shape[1:])
    model.fit(x, y, batch_size=32, epochs=30)
    
    return model

#%% 05 모델 평가 함수 생성

def model_eval(model, x, y):
    score = model.evaluate(x, y)
    print('loss = ', score[0])
    print('accuracy = ', score[1])
    
#%% 모델 생성, 학습, 평가

# 데이터 불러오기
x_train, x_test, y_train, y_test = load_dataset()

# 모델 학습
model = model_train(x_train, y_train)

#%% 학습이 완료된 모델을 저장

model.save('./landmark/landmark_model12.h5')


#%%

model_eval(model, x_test, y_test)




#%%


import tensorflow as tf
from tensorflow import keras 

import numpy as np
import matplotlib.pyplot as plt 

# 4개의 데이터 셋 반환(numpy 배열)


plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(True) # grid 선
plt.show()


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(categories[y_train[i]])
plt.show()

predictions[0]
np.argmax(predictions[0])
y_test[0]


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
    100*np.max(predictions_array),
    class_names[true_label]),color=color)

def plot_value_array(i, predictions_array, true_label):
predictions_array, true_label = predictions_array[i], true_label[i]
plt.grid(False)
plt.xticks([])
plt.yticks([])
thisplot = plt.bar(range(10), predictions_array, color="#777777")
plt.ylim([0, 1])
predicted_label = np.argmax(predictions_array)
thisplot[predicted_label].set_color('red')
thisplot[true_label].set_color('blue')



i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, pre, x_test, y_test)
plt.subplot(1,2,2)
plot_value_array(i, pre, y_test)
plt.show()


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, pre, x_test, y_test)
plt.subplot(1,2,2)
plot_value_array(i, pre, y_test)
plt.show()


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, pre, y_test, x_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, pre, y_test)
plt.show()