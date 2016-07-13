#### 모듈
 - 이미 만들어진 코드를 가져와 쓰는 방법
 - import 모듈이름
 - 사용방법 : 모듈 이름, 모듈안의 구성요소
 ```Python
   math.pi
   random.choice()
 ```

#### 모듈의 종류
- import math
  - 수학과 관련된 기능
- import random
  - 무작위와 관련된 기능
- import urllib request
  - 인터넷의 내용을 가져오는 기능


#### enumerate 함수

- 리스트가 있는 경우 순서와 리스트의 값을 전달하는 기능

```Python
  names = ['A', 'B', 'C']
  for i, name in enumerate(names):
    print('{}번: {}'.format(i + 1, name))
```

#### 모듈 만들기
 1. 사용할 함수, 메서드 코드를 작성한 모듈 파일을 생성
 2. 모듈이 쓰일 파일에 import를 사용하여 모듈을 호출
 3. 사용방법은 기존의 모듈과 동일
 4. 주의할 점은 사용자가 만든 모듈과 모듈을 쓸 파일이 같은 폴더에 있어야 한다.
