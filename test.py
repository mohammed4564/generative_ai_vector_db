import requests
url=requests.get('https://api.github.com')
data=url.json()
print(data)

def sum_numbers(a, b):
    return a + b
print(sum_numbers(3, 5))