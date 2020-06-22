# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:04:19 2020

@author: Seoul IT
"""


#%%
#!pip install sklearn
#!pip install PIL
#!pip install numpy

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from PIL import Image

import os, glob
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
import time

# 현재 경로 확인
print(os.getcwd())

#%% 
# 이미지 경로 지정
#root_dir = './landmark/'

# 카테고리 정보를 담는 리스트를 선언
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

image_width = 128
image_height = 128

#%%

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=15,  # 정수 무작위 회전의 각도 범위
    rescale=1./255,     # 크기 재조절. 기본은 None이고 None인 경우 크기 재조절이 안됨.
    shear_range=0.1,    # 부동 소수점. 층밀리기 강도(도 단위의 반시계 방향 층밀리기 각도)
    zoom_range=0.2,     # 부동 소수점.[하한, 상한] 무작위 줌의 범위 [1-zoom_range, 1+zoom_range]
    horizontal_flip=True,  # 인풋을 무작위로 가로로 뒤집기
    width_shift_range=0.1, # 1D 형태의 유사배열 혹은 정수
    height_shift_range=0.1 # 1D 형태의 유사배열 혹은 정수
)
#%%

for idx, category in enumerate(categories):   # enumerate : 인덱스, 값 반환
    #image_dir = root_dir + category
    files = glob.glob(category + '/' + '*.jpg')   # glob : 지정한 경로에 있는 파일 리스트를 가져옴
    print(category + '/' + '*.jpg')
    print('해당 폴더 파일 갯수 : ',len(files))   # 이미지를 제대로 불러왔는지 확인
    
    for i, f in enumerate(files):   # 이미지 로딩
        img = Image.open(f)         # 01 이미지 파일 불러오기
        img = img.convert('RGB')    # 02 RGB로 변환
        img = img.resize((image_width, image_height))  # 03 이미지 크기를 resize
        x = img_to_array(img)
        x = x.reshape((1,)+x.shape)
        i = 0
        for batch in train_datagen.flow(x, batch_size=1, 
                                  save_to_dir=category, 
                                  save_prefix=category, 
                                  save_format='jpg'):
            i += 1
            if i>10:
                break

#%%
# 데이터 변수
X = []   # 이미지 데이터
Y = []   # 레이블 데이터

for idx, category in enumerate(categories):   # enumerate : 인덱스, 값 반환
    #image_dir = root_dir + category
    files = glob.glob(category + '/' + '*.jpg')   # glob : 지정한 경로에 있는 파일 리스트를 가져옴
    print(category + '/' + '*.jpg')
    print('해당 폴더 파일 갯수 : ',len(files))   # 이미지를 제대로 불러왔는지 확인
    
    for i, f in enumerate(files):   # 이미지 로딩
        img = Image.open(f)         # 01 이미지 파일 불러오기
        img = img.convert('RGB')    # 02 RGB로 변환
        #img = img.resize((image_width, image_height))  # 03 이미지 크기를 resize
        data = np.asarray(img)      # 04 해당 이미지를 숫자 배열 데이터로 변경
        X.append(data)              # 05 변경한 데이터를 X의 리스트에 추가
        Y.append(idx)               # 06 해당 idx(이미지가 속한 범주)에 추가(Y값)

X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

#%%
print(X.shape, Y.shape)
xy = (X,Y)
np.save('landmark_test_n_split.npy',xy)

#%%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)

# 데이터 파일로 저장
np.save('landmark_aug.npy', xy)





#%%
# 데이터 변수
X = []   # 이미지 데이터
Y = []   # 레이블 데이터

root_dir = './landmark_test/'

for idx, category in enumerate(categories):   # enumerate : 인덱스, 값 반환
    image_dir = root_dir + category
    files = glob.glob(image_dir + '/' + '*.jpg')   # glob : 지정한 경로에 있는 파일 리스트를 가져옴
    print(category + '/' + '*.jpg')
    print('해당 폴더 파일 갯수 : ',len(files))   # 이미지를 제대로 불러왔는지 확인
    
    for i, f in enumerate(files):   # 이미지 로딩
        img = Image.open(f)         # 01 이미지 파일 불러오기
        img = img.convert('RGB')    # 02 RGB로 변환
        img = img.resize((image_width, image_height))  # 03 이미지 크기를 resize
        data = np.asarray(img)      # 04 해당 이미지를 숫자 배열 데이터로 변경
        X.append(data)              # 05 변경한 데이터를 X의 리스트에 추가
        Y.append(idx)               # 06 해당 idx(이미지가 속한 범주)에 추가(Y값)

X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

#%%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, train_size=0.90)
print(X_train.shape, Y_train.shape)

xy = (X_train, X_test, Y_train, Y_test)

np.save('landmark_test.npy',xy)              
