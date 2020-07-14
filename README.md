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
- Train
   * batch_size=128
   * epochs=140
- Evaluate
   * loss
   * accuracy
   
- 사용 라이브러리

