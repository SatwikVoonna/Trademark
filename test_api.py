
import requests
import json

# First clear cache
r = requests.delete("http://127.0.0.1:8080/cache")
print(f"Cache cleared: {r.json()}")

# Now make a fresh search with filters (like the frontend sends)
url = "http://127.0.0.1:8080/query"
payload = {"query": "Photosynthesis", "filters": {"subject": [], "class": [], "chapter": []}}

response = requests.post(url, json=payload)
data = response.json()

print(f"\nStatus: {response.status_code}")
print(f"cache_hit: {data.get('cache_hit')}")
result = data.get('result')
print(f"result type: {type(result)}")
print(f"result length: {len(result) if result else 0}")
if result:
    print("\nFirst result:")
    print(json.dumps(result[0], indent=2))
