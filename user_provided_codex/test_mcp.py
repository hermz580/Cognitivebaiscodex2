from main import load_data, list_biases, get_bias_details, get_bias_context, list_fallacies, get_fallacy_details, list_mental_models, get_mental_model_details, get_concept_details
import asyncio
import time

async def test_all():
    print("--- TESTING BIASES ---")
    data = load_data()
    print(f"Loaded {len(data)} biases.")
    print(get_bias_details('Confirmation bias'))

    print("\n--- TESTING FALLACIES ---")
    fallacies = list_fallacies()
    print(f"Loaded {len(fallacies)} fallacies.")
    print(get_fallacy_details('Ad Hominem Abusive'))

    print("\n--- TESTING MENTAL MODELS ---")
    models = list_mental_models()
    print(f"Loaded {len(models)} mental models.")
    print(f"Models: {models[:5]}")
    print(get_mental_model_details('First Principles'))

    print("\n--- TESTING CACHING (Wikipedia) ---")
    start_time = time.time()
    await get_bias_context('Confirmation bias')
    print(f"First Call Duration: {time.time() - start_time:.4f}s")
    
    start_time = time.time()
    res = await get_bias_context('Confirmation bias')
    print(f"Second Call Duration: {time.time() - start_time:.4f}s")
    if "[CACHED]" in res:
        print("Cache HIT verified.")
    else:
        print("Cache MISS (Error?)")

    print("\n--- TESTING UNIFIED CONCEPT SEARCH ---")
    print("Searching for 'Confirmation bias' (expecting Bias):")
    print(get_concept_details('Confirmation bias')[:100] + "...")
    
    print("\nSearching for 'Ad Hominem Abusive' (expecting Fallacy):")
    print(get_concept_details('Ad Hominem Abusive')[:100] + "...")
    
    print("\nSearching for 'First Principles' (expecting Mental Model):")
    print(get_concept_details('First Principles')[:100] + "...")

if __name__ == "__main__":
    asyncio.run(test_all())
