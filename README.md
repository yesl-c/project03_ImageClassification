# Image Classification
Creating a landmark classification model for ezch country using the **CNN**

---
## 국가별 랜드마크 이미지 분류 모델 생성
### (1) 데이터 수집 및 정제
   * 웹 크롤링을 통해 구글 이미지 검색 결과 수집
   * 17개의 카테고리 별 각 400개 정도의 이미지 수집 (약 6,800장)
        |  국가명  |       한글명칭      |       영문명칭      |
        |:--------:|:-------------------:|:-------------------:|
        |  두바이  |     부르즈할리파    |     Burj Khalifa    |
        |   미국   |        금문교       |     Golden Gate     |
        |   미국   |    자유의 여신상    |  Statue of Liberty  |
        |  브라질  |    구원의 예수상    |   Cristo Redentor   |
        |  스페인  | 산티아고 베르나베우 |  Santiago Bernabéu  |
        |   영국   |      타워브릿지     | London Tower Bridge |
        |   영국   |      런던 아이      |      London Eye     |
        |  이집트  |       스핑크스      |        Sphinx       |
        | 이탈리아 |     피사의 사탑     |    Torre di Pisa    |
        | 이탈리아 |       콜로세움      |      Colosseum      |
        |   인도   |       타지마할      |      Taj Mahal      |
        |   중국   |       만리장성      |   Wànlĭ Chángchéng  |
        |   칠레   |      모아이석상     |         Moai        |
        |  프랑스  |        개선문       |    Triumphal Arch   |
        |  프랑스  |        에펠탑       |     Eiffel Tower    |
        |   한국   |        숭례문       |   Sungnyemun Gate   |
        |   한국   |    석굴암 본존불    |   Seokguram buddha  |
        
   * 잘못 수집된 데이터는 삭제
   * 최종 이미지 데이터 수 : 4,426장

   * Augmentation
     + 약 96,000장으로 이미지 증식(약 5GB) : RAM 용량(35GB) 부족으로 training 실패
     + 약 65,000장으로 이미지 증식 후 사용(약 3GB)

### (2) 모델 설정
- Sequential Model
   * Convolution2D
   * Activation Function: LeakyReLU, softmax
   * MaxPooling
   * Dropout
- Model compile
   * loss : categorical_crossentropy
   * optimizer : adam
   * metrics : accuracy

### (3) 모델 학습 및 평가


|              | Loss   | Accuracy | # of layers | image_size | Epoch | Batchsize | 1st_Conv2d | 1st_Activation        | Max_Pooling2D | Dropout | 2nd or 3rd Activation | Model.compile_optimizer |
|--------------|--------|----------|-------------|------------|-------|-----------|------------|-----------------------|---------------|---------|-----------------------|-------------------------|
| try_1        | 1.6263 | 0.6468   | 5           | 32         | 10    | 64        | (32,3,3)   | Relu                  | (2,2)         | -       | -                     | rmsprop                 |
| try_2        | 5.8495 | 0.6134   | 5           | 64         | 50    | 32        | (32,3,3)   | Relu                  | (2,2)         | 0.25    | -                     | rmsprop                 |
| try_3        | 2.2813 | 0.6486   | 5           | 128        | 30    | 32        | (32,3,3)   | Relu                  | (2,2)         | 0.25    | -                     | rmsprop                 |
| try_4        | 2.4084 | 0.6748   | 5           | 256        | 30    | 32        | (32,3,3)   | Relu                  | (2,2)         | -       | -                     | rmsprop                 |
| try_5        | 2.3220 | 0.6757   | 7           | 256        | 30    | 32        | (32,3,3)   | Relu                  | (2,2)         | 0.25    | Relu                  | rmsprop                 |
| try_6        | 1.3419 | 0.7200   | 7           | 256        | 30    | 32        | (32,3,3)   | Relu                  | (2,2)         | 0.25    | Relu                  | adam                    |
| try_7        | 1.3986 | 0.7254   | 7           | 256        | 30    | 32        | (32,3,3)   | LeakyRelu(alpha=0.01) | (2,2)         | 0.25    | LeakyRelu(alpha=0.01) | rmsprop                 |
| try_8        | 2.5792 | 0.6480   | 7           | 256        | 30    | 32        | (32,3,3)   | Elu                   | (2,2)         | 0.25    | Elu                   | adam                    |
| try_9        | 1.747  | 0.6820   | 7           | 128        | 30    | 32        | (32,3,3)   | LeakyRelu(alpha=0.01) | (2,2)         | 0.25    | LeakyRelu(alpha=0.01) | adam                    |
| try_10(top3) | 1.5888 | 0.9437   | 7           | 128        | 30    | 32        | (32,3,3)   | LeakyRelu(alpha=0.01) | (2,2)         | 0.25    | LeakyRelu(alpha=0.01) | adam                    |
| try_11(top3) | 4.5057 | 0.9941   | 7           | 128        | 300   | 32        | (32,3,3)   | LeakyRelu(alpha=0.01) | (2,2)         | 0.25    | LeakyRelu(alpha=0.01) | adam                    |
| try_12       | 2.2183 | 0.6775   | 7           | 256        | 30    | 32        | (32,3,3)   | LeakyRelu(alpha=0.01) | (2,2)         | 0.25    | LeakyRelu(alpha=0.01) | adam                    |
| try_13       | 4.8315 | 0.7299   | 7           | 128        | 300   | 32        | (32,3,3)   | LeakyRelu(alpha=0.01) | (2,2)         | 0.25    | LeakyRelu(alpha=0.01) | adam                    |
| try_14       | 2.173  | 0.6893   | 7           | 128        | 50    | 32        | (32,3,3)   | LeakyRelu(alpha=0.01) | (2,2)         | 0.25    | LeakyRelu(alpha=0.01) | adam                    |
| try_15(=14)       | 1.799  | 0.72     | 7           | 128        | 50    | 32        | (32,3,3)   | LeakyRelu(alpha=0.01) | (2,2)         | 0.25    | LeakyRelu(alpha=0.01) | adam                    |
| try_16(=14)       | 1.8641 | **0.7103**   | 7           | 128        | 50    | 32        | (32,3,3)   | LeakyRelu(alpha=0.01) | (2,2)         | 0.25    | LeakyRelu(alpha=0.01) | adam                    |
| try_16(top3) | 2.3362 | 0.9703   | 7           | 128        | 50    | 32        | (32,3,3)   | LeakyRelu(alpha=0.01) | (2,2)         | 0.25    | LeakyRelu(alpha=0.01) | adam                    |
| try_17(VGG)  | 10.6   | 0.4336   | 13          | 128        | 50    | 32        | (64,3,3)   | Relu                  | (2,2)         | -       | Relu                  | adam                    |
| try_18(VGG)  | 5.823  | 0.4435   | 13          | 128        | 50    | 32        | (64,3,3)   | Relu                  | (2,2)         | 0.25    | Relu                  | adam                    |
| try_23       | 2.1555 | **0.8553**   | 7           | 128        | 140   | 128       | (32,3,3)   | LeakyRelu(alpha=0.01) | (2,2)         | 0.25    | LeakyRelu(alpha=0.01) | adam                    |
   

