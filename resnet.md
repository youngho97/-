# Deep Residual Learning for Image Recognition 논문 리뷰
해당 논문을 읽고, 논문을 요약, 이해하기 어려운 부분을 분석하여 적었고, 논문에 대한 나름대로의 평가를 정리하였습니다.

## 논문의 배경
딥러닝에서의 기본적인 개념
*  신경망을 깊게 쌓을 수록 표현력이 증가된다.
*  parameter 수가 증가되면 generalization 성능이 떨어진다.

즉, 좋은 신경망을 만들려면 layer를 깊게 쌓아야 한다. 하지만 실제로 어느정도 이상으로 깊게 쌓으면 성능이 오히려 안좋아진다(학습시키기 어렵기 때문에). 이에 대한 해결책은 무엇이 있을까..?


## 논문 요약
deep neural net은 학습시키기 어렵다. 하지만 이 논문에서는 residual learning framework를 통해 training을 쉽게 처리하였다. 그래서 다른 논문과 다르게 훨씬 deep한 network를 학습시켰다.

cnn을 깊게 쌓는 것은 이미지 분류 분야에서 돌파구 역할을 하였다. layer가 깊다는 것은 network의 표현력이 풍부해진다는 것이다.
그러면 layer를 깊게 쌓은 network를 학습시킬 수 있을까?
이에대한 문제로는 vanishing gradient, degradation 문제가 있다.
vanishing gradient(layer가 깊어지면서 역전파의 gradient가 사라지는 것) 문제는 적당한 activation function을 사용하고 가중치 값을 적절하게 초기화 시켜주고, batchNorm을 적용함으로써 해결 가능하다.
degradation(정확도가 어느 순간 정체되어 layer가 깊어질수록 성능이 나빠지는 것) 문제는 어떻게 해결 할 수 있을까..?

이에 대하여 본 논문은 degradation 문제를 해결하기 위해 deep residual learning framework를 제한하였다.
구체적으로, 의도했던 mapping H(x) (layer를 표현한 함수라고 생각하면 된다.)를 학습하기 보단 학습하기 쉬운 residual mapping을 따로 정의하여 그것을 대신 학습시킨다.
즉 F(x) = H(x)-x 를 정의하여 F(x)를 학습시킨다는 것이다. 그리고 이때 layer의 output은 F(x)+x이며 이 값은 기존 H(x)와 동일하다.
이렇게 F(x)에다가 x를 더하는 것은 short connection 이라고 한다.( = skip connection).
논문에서는 이런 short connection을 identity mapping 으로 사용할 수 있다고 한다. (identity mapping을 그냥 x를 더해주는 것으로 생각하면 편하다.)
이때 F(x)는 해당 layer에서 출력값에서 입력값을 뺀 값, 즉 잔여한 정보이므로 residual mapping이라고 하는 것 같다.

그러면 왜 F(x)를 학습시키는 것이 더욱 편리하다는 것일까?
논문에서는 극단적인 상황을 예로 들어 설명하고 있다. 극단적인 경우, identity mapping이 최적 해인 경우 F(x)를 0이 되게, 즉 잔여정보가 0이 되게끔 학습시키는 것이므로 학습 난이도가 쉽다고 한다.

학습난이도가 쉬워지는 것은 잔여정보만 학습하는 것이 아니라면 함수는 매 layer마다 새로운 weight을 학습해야 하며, 잔여학습을 적용하면 그럴 필요가 없이 잔여정보만 학습하면 되기 때문이다.

위의 설명에서 잘 이해가 안되는 부분은 다음과 같다.
최적 해가 identity mapping 이 되는 경우, 즉 잔여정보가 0이 되는 상황을 왜 예로 든 것인가?

본 논문에는 해당 내용이 설명되어 있지 않다. 따라서 필자가 자의적으로 해석해보았다.
잔여정보가 0이 되는 상황을 예로 든 것은 residual network는 깊은 layer를 가지고 있기 때문이라고 추측된다.
layer 수가 적은 경우 각 layer를 통과할 때마다 입력과 출력의 변화가 클 것이다.
하지만 layer 수가 많다면, 그리고 학습이 어느정도 된 상태라면 layer의 입력값과 통과하고난 출력값의 차이가 크지 않을 것이다. 이러한 이유로 깊은 신경망에서는 degradation 문제가 발생할 것이며,  따라서 최적 해가 identity mapping에 가까워질 것이라고 생각한다. (input이 x일때 out 또한 x인 것)
따라서 이러한 깊은 신경망에서의 입력값과 출력값이 차이가 없어지므로 잔여학습의 방법을 사용하면 차이가 없는 정보들을 잘 학습할 수 있어 degradation 문제를 해결할 수 있고, 쉽게 학습이 될 것이다. 


주의할 내용은 각 layer마다 이런 identity mapping을 적용해주는 것이 아니라, 두 단의 convolution layer마다 적용한다는 것이다. 쉽게 예를들어 설명하면 하나의 layer에 x를 더해주게 된다면 이는 하나의 linear layer가 사용된 것과 동일하기 때문에 아무런 이득이 없다.

본 논문에서는 residual newtork의 장점을 다음과 같이 말하고 있다
* 깊은 layer를 사용하더라도 학습 난이도가 쉽다.
* layer의 깊이가 깊어질수록 높은 정확도를 보여준다.
*  단순히 출력값에 x를 더해주는 것으로 끝나기 때문에 파라미터 추가도 없고, 복잡도 증가도 없다.

이후의 내용은 다음과 같다.
다른 network와 달리 residual learning을 적용한 resnet은 layer를 깊게 쌓아도 학습이 잘 된다는 것.
layer를 깊게 쌓은 resnet은 성능이 우수하여 여러 image classification 대회에서 1등을 하였다.
object decetion, image segmentation 등에서도 resnet이 우수한 성능을 보인다.
(간단히 잔여정보만 학습 할 수 있도록 더해주는 과정만 추가 되었으므로 다른 분야에서도 무리없이 사용 가능하다.)

## 총평
image classification 등의 분야에 획기적인 성능 향상을 일으킨 resnet...
알고보면 정말 단순한 아이디어다.
혁신적인 것은 복잡한 생각이아니라 정말 단순한 생각일 수도 있다.


## Reference
* He, Kaiming, et al. "**Deep residual learning for image recognition**." arXiv preprint arXiv:1512.03385 (2015) 
https://arxiv.org/pdf/1512.03385.pdf
*  라온피플 blog - Resnet 
https://blog.naver.com/PostView.naver?blogId=laonple&logNo=220761052425&parentCategoryNo=&categoryNo=22&viewDate=&isShowPopularPosts=false&from=postList
