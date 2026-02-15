from main import load_data, list_biases, get_bias_details

data = load_data()
print(f"Loaded {len(data)} rows.")

biases = list_biases()
print(f"Total biases: {len(biases)}")
print(f"First 5 biases: {biases[:5]}")

print("\nDetails for 'Confirmation bias':")
print(get_bias_details('Confirmation bias'))
