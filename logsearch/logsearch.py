import ollama
import json
import math
import argparse
import os

# --- Helper Functions ---
def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def magnitude(v):
    return math.sqrt(sum(a * a for a in v))

def cosine_similarity(v1, v2):
    return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2))

def load_log_file(filepath, max_lines=500):
    chunks = []
    with open(filepath, 'r', errors='ignore') as f:
        lines = f.readlines()
    chunk_size = 5
    for i in range(0, len(lines), chunk_size):
        chunk = "".join(lines[i:i+chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks[:max_lines]

def embed_logs(log_files):
    data_to_store = []
    for log_path in log_files:
        if not os.path.exists(log_path):
            print(f"Warning: {log_path} not found, skipping...")
            continue
        print(f"Processing {log_path}...")
        chunks = load_log_file(log_path)
        for chunk in chunks:
            response = ollama.embed(model='nomic-embed-text', input=chunk)
            vector = response['embeddings'][0]
            data_to_store.append({
                "text": chunk,
                "source": log_path,
                "vector": vector
            })
        print(f"  → {len(chunks)} chunks embedded from {log_path}")

    with open("log_vectors.json", "w") as f:
        json.dump(data_to_store, f)
    print(f"\nDone! Saved {len(data_to_store)} chunks to log_vectors.json")

def search(query, top_k=3):
    if not os.path.exists("log_vectors.json"):
        print("No log_vectors.json found. Run with --index first.")
        return

    with open("log_vectors.json", "r") as f:
        database = json.load(f)

    response = ollama.embed(model='nomic-embed-text', input=query)
    query_vector = response['embeddings'][0]

    results = []
    for entry in database:
        score = cosine_similarity(query_vector, entry["vector"])
        results.append((score, entry["source"], entry["text"]))

    results.sort(reverse=True)
    for i, (score, source, text) in enumerate(results[:top_k], 1):
        print(f"\nResult #{i} (score: {score:.4f}) from {source}")
        print(f"{text}")
        print("---")

# --- CLI ---
parser = argparse.ArgumentParser(description="Semantic search over your log files")

parser.add_argument('--index', nargs='+', metavar='LOG_FILE',
                    help='Log files to embed e.g. --index /var/log/syslog /var/log/auth.log')

parser.add_argument('--query', type=str,
                    help='Search query e.g. --query "failed login attempts"')

parser.add_argument('--top', type=int, default=5,
                    help='Number of results to return (default: 3)')

args = parser.parse_args()

if args.index:
    embed_logs(args.index)

if args.query:
    search(args.query, top_k=args.top)

if not args.index and not args.query:
    parser.print_help()