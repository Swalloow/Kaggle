## 함수의 정의
 1. 함수는 코드의 덩어리에 이름을 붙인 것이다.
 2. 새 함수를 정의할 수 있다.
 3. print는 미리 만들어진 함수이다.
 4. 함수를 한번 만들고 나면, 그 안은 잊어버려도 좋다.


```Python
  def function():
    print('Hello, function!')

print('first line')
function()
print('last line')

 ```

## 함수의 리턴
 - return을 이용해 값을 돌려줄 수 있다.

 ```Python
 def add_10(value):
      result = value + 10
      return result

  n = add_10(5)
  print(n)

  ```

- 여러 값 반환
  - return 뒤에 여러 값을 쉼표로 구분해서 값을 보내고, 받을 때도 쉼표로 구분하여 받는다.


## 문자열.format()
  - 문자열의 대괄호 자리에 format 뒤의 괄호안에 들어 있는 값을 하나씩 넣는다.
  - 문자열에 포함된 대괄호 갯수 보다 format 안에 들어 있는 값의 수가 많으면 정상 동작
    - `print('{} 번 손님'.format(number, greeting))`
  - 문자열에 포함된 대괄호 갯수 보다 format 안에 들어 있는 값의 수가 적으면 에러
    - `print('{} 번 손님 {}'.format(number))`


```Python
number = 20
welcome = '환영합니다'
base = '{} 번 손님 {}'

# 아래 3개의 print는 값은 값을 출력
print(number, '번 손님', welcome)
print(base.format(number, welcome))
print('{} 번 손님 {}'.format(number, welcome))
# 20번 손님 환영합니다

```
