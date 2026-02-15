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
async def get_bias_context(name: str) -> str:
    """
    Fetch a summary or content for a cognitive bias from its reference URL.
    This works for Wikipedia links and general web pages.
    Now includes caching to speed up repeated requests.
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
    if not url:
        return f"No URL available for '{name}'."

    # Check Cache
    cached = get_cached_content(url)
    if cached:
        return f"[CACHED] {cached}"
    
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            content_to_save = ""
            # Special handling for Wikipedia
            if "wikipedia.org" in url:
                title = url.split("/wiki/")[-1]
                api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
                # Wikipedia API requires a User-Agent
                headers = {
                    "User-Agent": "CognitiveBiasCodexMCP/1.0 (https://github.com/mootpoots94-jpg/Cognitivebaiscodex)"
                }
                resp = await client.get(api_url, headers=headers)
                resp.raise_for_status()
                wiki_data = resp.json()
                content_to_save = f"Source: Wikipedia\n\n{wiki_data.get('extract', 'No summary available.')}"
            
            # General scraping for other sites
            else:
                resp = await client.get(url, headers={"User-Agent": "CognitiveBiasCodexMCP/1.0"})
                resp.raise_for_status()
                
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    
                    # Try to find the main content
                    content = ""
                    article = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
                    
                    if article:
                        paragraphs = article.find_all('p')
                    else:
                        paragraphs = soup.find_all('p')
                    
                    count = 0
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if len(text) > 50:
                            content += text + "\n\n"
                            count += 1
                        if count >= 3:
                            break
                            
                    if not content:
                        content_to_save = f"Source: {url}\n\nCould not extract main content. Please visit the link directly."
                    else:
                        content_to_save = f"Source: {url}\n\n{content}"
                    
                except ImportError:
                    return f"Source: {url}\n\nBeautifulSoup not installed. Cannot scrape content."
                except Exception as scrape_err:
                    return f"Source: {url}\n\nError parsing content: {str(scrape_err)}"

            # Save to Cache
            save_cached_content(url, content_to_save)
            return content_to_save

        except Exception as e:
            return f"Error fetching content from {url}: {str(e)}"

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



FALLACIES_FILE = os.path.join(os.path.dirname(__file__), "fallacies.json")

def load_fallacies():
    if not os.path.exists(FALLACIES_FILE):
        return []
    try:
        import json
        with open(FALLACIES_FILE, 'r', encoding='utf-8') as f:
            # Handle potential JSONL format or standard JSON
            content = f.read().strip()
            if content.startswith('['):
                return json.loads(content)
            else:
                # Assume JSONL
                return [json.loads(line) for line in content.split('\n') if line.strip()]
    except Exception as e:
        print(f"Error loading fallacies: {e}")
        return []

@mcp.tool()
def list_fallacies() -> List[str]:
    """
    List all logical fallacies available in the database.
    """
    data = load_fallacies()
    return sorted([f.get('name') for f in data if f.get('name')])

@mcp.tool()
def get_fallacy_details(name: str) -> str:
    """
    Get the description, logical form, and examples for a specific logical fallacy.
    """
    data = load_fallacies()
    for f in data:
        if f.get('name').lower() == name.lower():
            res = f"Fallacy: {f.get('name')}\n"
            res += f"Description: {f.get('description')}\n"
            if f.get('logical_form'):
                res += f"Logical Form: {f.get('logical_form')}\n"
            if f.get('explanation_with_examples'):
                res += f"\nExamples:\n{f.get('explanation_with_examples')}"
            return res
    return f"Fallacy '{name}' not found."

@mcp.tool()
def search_fallacies(query: str) -> List[Dict[str, str]]:
    """
    Search for logical fallacies matching a query in their name or description.
    """
    data = load_fallacies()
    results = []
    for f in data:
        name = f.get('name', '')
        desc = f.get('description', '')
        if query.lower() in name.lower() or query.lower() in desc.lower():
            results.append({
                "name": name,
                "description": desc[:100] + "..." if len(desc) > 100 else desc
            })
    return results

MENTAL_MODELS_FILE = os.path.join(os.path.dirname(__file__), "mental_models.json")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def load_mental_models():
    if not os.path.exists(MENTAL_MODELS_FILE):
        return []
    try:
        import json
        with open(MENTAL_MODELS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading mental models: {e}")
        return []

def get_cached_content(url: str) -> Optional[str]:
    import hashlib
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    cache_path = os.path.join(CACHE_DIR, url_hash + ".txt")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return None
    return None

def save_cached_content(url: str, content: str):
    import hashlib
    try:
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        cache_path = os.path.join(CACHE_DIR, url_hash + ".txt")
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"Error saving cache: {e}")

@mcp.tool()
def list_mental_models() -> List[str]:
    """
    List all Mental Models available in the database.
    """
    data = load_mental_models()
    return sorted([m.get('name') for m in data if m.get('name')])

@mcp.tool()
def get_mental_model_details(name: str) -> str:
    """
    Get the description and example for a specific Mental Model.
    """
    data = load_mental_models()
    for m in data:
        if m.get('name').lower() == name.lower():
            res = f"Mental Model: {m.get('name')}\n"
            res += f"Category: {m.get('category')}\n"
            res += f"Description: {m.get('description')}\n"
            res += f"Example: {m.get('example')}"
            return res
    return f"Mental Model '{name}' not found."

@mcp.tool()
def get_concept_details(concept_name: str) -> str:
    """
    Unified search tool that checks Biases, Fallacies, and Mental Models for a given concept name.
    """
    # Check Bias
    bias_details = get_bias_details(concept_name)
    if "not found" not in bias_details:
        return f"[Type: Cognitive Bias]\n{bias_details}"

    # Check Fallacy
    fallacy_details = get_fallacy_details(concept_name)
    if "not found" not in fallacy_details:
        return f"[Type: Logical Fallacy]\n{fallacy_details}"

    # Check Mental Model
    model_details = get_mental_model_details(concept_name)
    if "not found" not in model_details:
        return f"[Type: Mental Model]\n{model_details}"

    return f"Concept '{concept_name}' not found in any database (Biases, Fallacies, Mental Models)."

if __name__ == "__main__":
    mcp.run()


