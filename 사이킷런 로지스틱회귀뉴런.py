from sklearn.linear_model import SGDClassifier
#분류용 데이터 세트 준비
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
x=cancer.data
y=cancer.target

import matplotlib.pyplot as plt
import numpy as np
plt.boxplot(cancer.data)
plt.xlabel('feature')
plt.ylabel('value')
plt.show()

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.2, random_state=42)

#SGDClassifier 클래스 사용
sgd = SGDClassifier(loss='log', max_iter=100, tol=1e-3, random_state=42) #매개변수 loss, 손실함수 log, 반복횟수 max-iter, 난수 초깃값을 42로 설정해서 책이랑 같은 결과 나오게 함
#tol 지정한 만큼 손실함수의 값이 감소되지 않으면 반복을 중단함. tol값을 설정하지 않으면 max_iter값을 늘리라고 할 수도 있는데 이는 손실 함수의 값이 최적값으로 수렴할정도로 충분한 반복횟수를 입력했는지 알려주는 것이라 유용하기도 함.
#fit으로 훈련, score로 평가
sgd.fit(x_train, y_train)
sgd.score(x_test, y_test)

#predict로 양성, 음성 예측
sgd.predict(x_test[0:10])
