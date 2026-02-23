# Embeddings Project

Perform semantic search over system logs and spreadsheets using local AI embeddings powered by Ollama.

## Overview

This project provides tools for semantic search using embeddings:
- **logsearch**: Search system logs (syslog, auth.log, etc.) semantically
- **excelsearch**: Search and filter Excel spreadsheets by semantic meaning
- **hello-world**: Simple example demonstrating embeddings with JSON

All embeddings are computed locally using [Ollama](https://ollama.ai/) with the `nomic-embed-text` model.

## Installation

1. Clone or navigate to this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure [Ollama](https://ollama.ai/) is running with the `nomic-embed-text` model available.





## Usage

### Log Search

Index and search system logs with semantic queries.

#### First time, index your logs:
```bash
python logsearch/logsearch.py --index /var/log/syslog /var/log/syslog
```

#### Then search:
```bash
python logsearch/logsearch.py --query "failed login attempts"
```

#### Control how many results come back:
```bash
python logsearch/logsearch.py --query "disk space warning" --top 5
```

#### You can also index and search in one command:
```bash
python logsearch/logsearch.py --index /var/log/syslog --query "network interface down"
```

#### Useful example queries:
- "failed login attempts" — finds brute force or SSH failures
- "service crashed or restarted" — finds systemd failures
- "disk space warning" — finds storage issues
- "network interface down" — finds connectivity problems
- "permission denied" — finds access errors

### Excel Search

Search Excel spreadsheets by semantic meaning and exact filters.

#### Index an Excel file:
```bash
python excelsearch/excelsearch.py --index test_data/data.xlsx
```

#### Perform semantic search:
```bash
python excelsearch/excelsearch.py --query "expensive trip"
python excelsearch/excelsearch.py --query "quick cheap visit"
```

#### Filter by exact values:
```bash
python excelsearch/excelsearch.py --query "Credit Card" --filter "Payment Method:Credit Card"
```

#### Combine semantic search with filters:
```bash
python excelsearch/excelsearch.py --query "long duration stay" --filter "Payment Method:Credit Card"
```

#### Multiple filters:
```bash
python excelsearch/excelsearch.py --query "expensive trip" --filter "Payment Method:Credit Card,Vehicle Type:Sedan"
```

## How It Works

1. **Embedding**: Text from logs or Excel rows is converted to vector embeddings using Ollama
2. **Storage**: Vectors are saved to JSON for fast retrieval
3. **Search**: Query is embedded and compared to stored vectors using cosine similarity
4. **Results**: Top matches are returned ranked by relevance score