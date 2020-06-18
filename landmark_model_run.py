# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:46:32 2020

@author: Seoul IT
"""


#%% 라이브러리 호출
import sys, os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.models import load_model
from PIL import Image
import numpy as np

#%% 테스트 이미지 목록
image_files = ["./landmark/test/998C7A375B4A190501.jpg",
               "./landmark/test/10003379-0.jpg",
               "./landmark/test/58908041.1.jpg",
               "./landmark/test/images (1).jpg",
               "./landmark/test/images (2).jpg",
               "./landmark/test/images (3).jpg",
               "./landmark/test/images.jpg",
               "./landmark/test/istockphoto-505051855-1024x1024.jpg"]

image_size = 64

nb_classes = len(image_files)

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

print(nb_classes)



#%% 예측할 새로운 이미지 불러오기

X = []
files = []

for fname in image_files:
    img = Image.open(fname)   # 파일 불러오기
    img = img.convert('RGB')
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    in_data = in_data.astype('float')/256
    X.append(in_data)
    files.append(fname)
    
X = np.array(X)
print(X.shape)

#%% 모델을 불러와서 예측

model = load_model('./landmark/landmark_model.h5')

pre = model.predict(X)

# 결과 출력
for i, p in enumerate(pre):
    y = p.argmax()
    print('입력 : ', files[i])
    print('예측 : ', y)
    

#%%
model.evaluate(x_test,y_test)
    
    
    
    
    
    
    