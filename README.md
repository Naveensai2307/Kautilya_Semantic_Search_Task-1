# Kautilya_Semantic_Search_Task-1
This module implements a semantic search engine over the Postman Twitter API collection using:
Sentence Transformers for dense embeddings.
FAISS for fast similarity search.
Chunked document parsing from the Postman Twitter API GitHub repo.
Command-line querying with ranked JSON output.

The goal is to allow natural-language search queries such as:
"How do I fetch tweets with expansions?"
"Show me endpoints related to tweet lookup policies."

The system returns the most semantically relevant chunks from the API collection along with similarity scores.

Install dependencies:
pip install -r requirements.txt

requirements.txt:
faiss-cpu==1.7.4
sentence-transformers==2.2.2
scikit-learn==1.3.2
numpy==1.26.4
tqdm==4.66.1
python-dotenv==1.0.0
networkx==3.1
hdbscan==0.8.32
ujson==5.8.0
requests==2.34.0

Run the script:
python semantic_search.py --query "How to fetch tweets?"
