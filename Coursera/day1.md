# Week1
Software Maestro 7th ML Study.

## Supervised Learning
감독학습이란, 과거의 평가된 데이터(Training Data)로부터 하나의 함수를 유추해내기 위한 방법이다.

 - Classification : 분류
   - 단절된 요소를 나누는 것 (discrete value)
   - 이메일이 도착했을 때 스펨 메일인지 아닌지 구분
     환자의 종양 크기에 따라 악성 종양인지, 아닌지 판단
 - Regression : 회귀 (추상, 트렌드, 경향)
   - Regression은 continuous value 를 예측하는 것.
   - 집의 평수에 따라 가격을 예측
     수 많은 제품들을 3달 안에 판매할 수 있는지 예측

## Unsupervised Leaning
자율학습이란, 대상에 대한 어떤 정보도 주어지지 않은 상태(labeling 되지 않은 데이터)에서 예측하는 방법이다.

 - Clustering : 군집화
   - Google News 에서 비슷한 주제의 기사를 모아서 보여주는 것
   - 유전자 나열을 통해 사람이 몇 명 있는지 구분하는 것
 - Cocktail party problem (Audio processing)
   - 잡음이 많은 칵테일 파티에서 녹음된 다양한 소리 중에서 한 사람의 목소리만 추출해 주는 문제
   - 파장의 높낮이를 구분해서 서로 다른 주체라는 것을 기계가 파악함

## Model and Cost Function
주어진 Training Set을 통해 Cost Function을 구하는 문제이다.
Hypothesis란, 함수와 비슷한 개념 (매핑, 예측)이다.
Hypothesis h는 다음과 같이 계산할 수 있다.

$h(\theta) = \theta_0 + \theta_1x$

여기에서 $h(\theta)$는 결과 값, $\theta$는 가중치를 의미하며,
예측 값 - 실제 값 = 0 이면 완벽한 모델링이다.
그래서 기울기 계산(미분 값)을 통해 계속 0으로 가까워지도록 해야 한다.
아이디어는 $\theta_0$과 $\theta_1$을 조절하여 $h(\theta)$가 함수 (x,y)에서 실제 결과 값인 y에 가까워져야 하는 것이다.
Cost Function 또는 Squred Error Function J의 계산은 다음과 같다.

$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^i) - y^i)^2$

```
위의 수식에서 왜 제곱을 할까?
절대값 씌우면 되는거 아닌가...
```
위의 수식에서 파라메터(input)는 $\theta_0, \theta_1$ 이다.
이제 J를 최소화하는 방법을 알아보기 위해서 파라메터가 1개인 경우부터 살펴본다.
$h_\theta(x)$와 $J(\theta_1)$의 그래프를 그려보면 1차 함수와 2차 함수 그래프가 나타난다.
```
어차피 y는 상수이고 세타에 대한 2차 식으로 표현된다면,
모든 머신러닝 케이스가 전부 2차 그래프로 표현되는 것 아닌가?
```
2개의 파라메터로 그래프를 그린다면, 3차원 그래프로 표현된다.
그리고 이를 평면에 나타낸다면 등고선의 형태(contour plot)로 표현된다.
