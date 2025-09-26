import time

import requests
import mmengine

url = "http://172.30.60.76:8888/get_reward"
headers = {
    "Content-Type": "application/json",
}

t = time.time()
response = requests.post(url, json=mmengine.load('tmp.pkl'), headers=headers, timeout=180)
print(f'time: {time.time() - t:.3f}')
print(response.json())
