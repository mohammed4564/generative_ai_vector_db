import requests
url=requests.get('https://api.github.com')
data=url.json()
print(data)