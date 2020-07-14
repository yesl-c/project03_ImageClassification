# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:04:19 2020

@author: Seoul IT
"""


#%% 라이브러리 설정

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from PIL import Image

import os, glob
import numpy as np

# 현재 경로 확인
print(os.getcwd())


#%% 

# 이미지 경로 지정
root_dir = './landmark/'

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

# 이미지 사이즈 설정
image_width = 128
image_height = 128


#%% 이미지를 숫자 배열 데이터로 변경

# 데이터 변수
X = []   # 이미지 데이터
Y = []   # 레이블 데이터

for idx, category in enumerate(categories):   # enumerate : 인덱스, 값 반환
    image_dir = root_dir + category
    files = glob.glob(category + '/' + '*.jpg')   # glob : 지정한 경로에 있는 파일 리스트를 가져옴
    print(image_dir + '/' + '*.jpg')
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


#%% 훈련, 테스트 데이터 설정

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)

# 데이터 파일로 저장
np.save('landmark_agumentation.npy', xy)

        
        
        
        
        