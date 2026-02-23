import json

# To Save:
my_data = {
    "text": "Intel Iris Xe is great for AI",
    "vector": [0.12, -0.04, 0.88] # Your 768-dim list
}
with open('database.json', 'w') as f:
    json.dump(my_data, f)