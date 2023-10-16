import requests 

url = 'http://0.0.0.0:8000/predict/'

client = {
    'job': 'retired',
    'duration': 445,
    'poutcome': 'success'
}

response = requests.post(url, json=client).json()
print(response)