import requests

base = "http://127.0.0.1:8000"

# Simulate frontend pagination
all_biases = []
page = 1
total_pages = 1
while page <= total_pages:
    r = requests.get(f"{base}/search?page={page}&size=100&enrich=false", timeout=10)
    if r.status_code != 200:
        print(f"  Page {page} FAILED: {r.status_code}")
        break
    data = r.json()
    all_biases.extend(data.get("items", []))
    total_pages = data.get("total_pages", 1)
    print(f"  Page {page}/{total_pages}: got {len(data.get('items',[]))} items")
    page += 1

print(f"\nBIASES: {len(all_biases)} total")

# Fallacies
r = requests.get(f"{base}/fallacies", timeout=10)
fallacies = r.json()
print(f"FALLACIES: {r.status_code} -> {len(fallacies)} items")

# Models
r = requests.get(f"{base}/mental_models", timeout=10)
models = r.json()
print(f"MODELS: {r.status_code} -> {len(models)} items")

# Analyze
r = requests.post(f"{base}/analyze", json={"text": "They are liars attacking our values"}, timeout=10)
print(f"ANALYZE: {r.status_code} -> {r.text[:200]}")

total = len(all_biases) + len(fallacies) + len(models)
print(f"\n=== TOTAL VECTORS: {total} ===")
print(f"  Biases: {len(all_biases)}")
print(f"  Fallacies: {len(fallacies)}")
print(f"  Models: {len(models)}")
print(f"\nSample bias name: {all_biases[0].get('name', 'EMPTY') if all_biases else 'NONE'}")
print(f"Sample fallacy name: {fallacies[0].get('name', 'EMPTY') if fallacies else 'NONE'}")
print(f"Sample model name: {models[0].get('name', 'EMPTY') if models else 'NONE'}")
