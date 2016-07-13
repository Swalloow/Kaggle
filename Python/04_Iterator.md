#### for in 반복문
 - 코드를 필요한 만큼 반복해서 실행
 ```Python
   for pattern in patterns:
     print(pattern)
 ```
 1. 리스트 patterns의 값을 하나씩 꺼내 pattern으로 전달
 2. 리스트의 길이만큼 print(pattern) 실행

#### range 함수

- 필요한 만큼의 숫자를 만들어내는 유용한 기능

```Python
  for i in range(5):
    print(i)
```

#### enumerate 함수

- 리스트가 있는 경우 순서와 리스트의 값을 전달하는 기능

```Python
  names = ['A', 'B', 'C']
  for i, name in enumerate(names):
    print('{}번: {}'.format(i + 1, name))
```
