
import csv
import json
import os

DATA_FILE = "bias.csv"
OUTPUT_FILE = "biases.json"

def csv_to_json():
    biases = []
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    with open(DATA_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            b_id = row.get('id', '')
            if not b_id or b_id == 'bias':
                continue
            
            parts = b_id.split('.')
            category = parts[1] if len(parts) > 1 else ""
            subcategory = parts[2] if len(parts) > 2 else ""
            name = parts[-1]
            url = row.get('value', '').strip()
            
            # Only include leaf nodes (actual biases) for the viewer
            if url: 
                biases.append({
                    "name": name,
                    "category": category,
                    "subcategory": subcategory,
                    "url": url,
                    "description": "Click to view external source." # Placeholder
                })
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(biases, f, indent=2)
    print(f"Successfully converted {DATA_FILE} to {OUTPUT_FILE} with {len(biases)} biases.")

if __name__ == "__main__":
    csv_to_json()
