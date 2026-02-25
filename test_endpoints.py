import requests

base = 'http://127.0.0.1:8000'

# 1. Health
r = requests.get(base + '/health', timeout=8)
print(f"HEALTH: {r.status_code} -> {r.text[:200]}")

# 2. Search
r = requests.get(base + '/search?page=1&size=5&enrich=false', timeout=8)
print(f"SEARCH: {r.status_code} -> items count: {len(r.json().get('items', []))} total: {r.json().get('total')}")

# 3. Fallacies
r = requests.get(base + '/fallacies', timeout=8)
d = r.json()
if isinstance(d, list):
    print(f"FALLACIES: {r.status_code} -> {len(d)} items. First: {d[0].get('name', 'N/A') if d else 'empty'}")
else:
    print(f"FALLACIES: {r.status_code} -> {str(d)[:200]}")

# 4. Mental Models
r = requests.get(base + '/mental_models', timeout=8)
d = r.json()
if isinstance(d, list):
    print(f"MENTAL_MODELS: {r.status_code} -> {len(d)} items. First: {d[0].get('name', 'N/A') if d else 'empty'}")
else:
    print(f"MENTAL_MODELS: {r.status_code} -> {str(d)[:200]}")

# 5. Analyze
r = requests.post(base + '/analyze', json={"text": "They are liars attacking our values"}, timeout=8)
print(f"ANALYZE: {r.status_code} -> {r.text[:300]}")

# 6. Check what the bias.csv actually has
import csv
with open('bias.csv', 'r', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
print(f"CSV ROWS: {len(rows)}")
print(f"CSV HEADERS: {list(rows[0].keys()) if rows else 'empty'}")
print(f"SAMPLE ROW: {rows[0] if rows else 'none'}")
