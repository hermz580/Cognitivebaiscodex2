import csv
import os
import httpx
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("CognitiveBiasCodex")

DATA_FILE = os.path.join(os.path.dirname(__file__), "bias.csv")

def load_data():
    biases = []
    if not os.path.exists(DATA_FILE):
        return []
        
    with open(DATA_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            b_id = row.get('id', '')
            if not b_id or b_id == 'bias':
                continue
            
            parts = b_id.split('.')
            # parts[0] is always 'bias'
            category = parts[1] if len(parts) > 1 else ""
            subcategory = parts[2] if len(parts) > 2 else ""
            
            # The name is the last part
            name = parts[-1]
            url = row.get('value', '').strip()
            
            biases.append({
                "id": b_id,
                "category": category,
                "subcategory": subcategory,
                "name": name,
                "url": url,
                "is_leaf": bool(url)
            })
    return biases

@mcp.tool()
def list_biases(category: Optional[str] = None) -> List[str]:
    """
    List all cognitive biases. 
    Optionally filter by category.
    """
    data = load_data()
    results = []
    for b in data:
        if b['is_leaf']:
            if category:
                if category.lower() in b['category'].lower():
                    results.append(b['name'])
            else:
                results.append(b['name'])
    return sorted(list(set(results)))

@mcp.tool()
def get_bias_details(name: str) -> str:
    """
    Get the category, subcategory, and reference URL for a specific cognitive bias.
    """
    data = load_data()
    for b in data:
        if b['name'].lower() == name.lower() and b['is_leaf']:
            res = f"Bias: {b['name']}\n"
            res += f"Category: {b['category']}\n"
            res += f"Subcategory: {b['subcategory']}\n"
            res += f"Reference: {b['url']}"
            return res
    return f"Bias '{name}' not found."

@mcp.tool()
async def get_wiki_summary(name: str) -> str:
    """
    Fetch a brief summary of a cognitive bias from Wikipedia.
    """
    bias = None
    data = load_data()
    for b in data:
        if b['name'].lower() == name.lower() and b['is_leaf']:
            bias = b
            break
    
    if not bias:
        return f"Bias '{name}' not found."
    
    url = bias['url']
    if "wikipedia.org" not in url:
        return f"No Wikipedia URL available for '{name}'. Try: {url}"
    
    # Extract page title from URL
    title = url.split("/wiki/")[-1]
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(api_url)
            resp.raise_for_status()
            wiki_data = resp.json()
            return wiki_data.get('extract', 'No summary available.')
        except Exception as e:
            return f"Error fetching summary: {str(e)}"

@mcp.tool()
def search_biases(query: str) -> List[Dict[str, str]]:
    """
    Search for biases matching a query string in their name or category.
    """
    data = load_data()
    results = []
    for b in data:
        if b['is_leaf'] and (query.lower() in b['name'].lower() or query.lower() in b['category'].lower()):
            results.append({
                "name": b['name'],
                "category": b['category'],
                "url": b['url']
            })
    return results

@mcp.tool()
def get_categories() -> Dict[str, List[str]]:
    """
    Get all categories and their associated sub-themes.
    """
    data = load_data()
    cats = {}
    for b in data:
        cat = b['category']
        if not cat: continue
        if cat not in cats:
            cats[cat] = set()
        if b['subcategory']:
            cats[cat].add(b['subcategory'])
    
    return {k: sorted(list(v)) for k, v in cats.items()}

@mcp.resource("cognitive-bias://full-codex")
def get_full_codex() -> str:
    """Returns the full hierarchical data of the cognitive bias codex in Markdown format."""
    data = load_data()
    output = "# Cognitive Bias Codex\n\n"
    
    tree = {}
    for b in data:
        cat = b['category']
        sub = b['subcategory']
        if not cat: continue
        if cat not in tree: tree[cat] = {}
        if sub not in tree[cat]: tree[cat][sub] = []
        if b['is_leaf']:
            tree[cat][sub].append(b)
            
    for cat, subs in tree.items():
        output += f"## {cat}\n"
        for sub, biases in subs.items():
            if sub:
                output += f"### {sub}\n"
            for b in biases:
                output += f"- [{b['name']}]({b['url']})\n"
            output += "\n"
    return output

if __name__ == "__main__":
    mcp.run()

