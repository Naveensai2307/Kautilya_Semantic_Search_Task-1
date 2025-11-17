#!/usr/bin/env python3
"""
semantic_search.py

Usage:
  python semantic_search.py --query "How do I fetch tweets with expansions?" --k 5 --repo_dir ./postman-twitter-api

Outputs JSON to stdout:
[
  {"score": 0.87, "chunk_id": 0, "source": "README.md", "text": "..."},
  ...
]
"""

import argparse
import os
import json
import tempfile
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors
import glob
import re
import requests
from tqdm import tqdm
import ujson as uj

REPO_GITHUB = "https://github.com/xdevplatform/postman-twitter-api"

def clone_or_download(repo_dir):
    repo_path = Path(repo_dir)
    if repo_path.exists():
        return repo_path
    # Try git clone
    try:
        import subprocess
        subprocess.run(["git","clone",REPO_GITHUB, str(repo_path)], check=True)
        return repo_path
    except Exception:
        # fallback: attempt to grab README and collection.json from GitHub raw (best-effort)
        repo_path.mkdir(parents=True, exist_ok=True)
        # try README
        raw_readme = "https://raw.githubusercontent.com/xdevplatform/postman-twitter-api/main/README.md"
        try:
            r = requests.get(raw_readme, timeout=10)
            if r.ok:
                (repo_path / "README.md").write_text(r.text, encoding="utf8")
        except Exception:
            pass
        return repo_path

def collect_text_chunks(repo_dir, chunk_size=300, overlap=50):
    repo_path = Path(repo_dir)
    chunks = []
    # If repo_dir points to a single JSON/JSONL file, try to extract items from it
    if repo_path.is_file() and repo_path.suffix.lower() in ('.json', '.jsonl', '.ndjson'):
        try:
            text = repo_path.read_text(encoding='utf8', errors='ignore')
            # try fast JSON parser first
            parsed = None
            try:
                parsed = uj.loads(text)
            except Exception:
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
            entries = []
            if isinstance(parsed, dict) and 'items' in parsed and isinstance(parsed['items'], list):
                entries = parsed['items']
            elif isinstance(parsed, list):
                entries = parsed
            else:
                # fall back to scanning for items array and extracting objects tolerant of newlines
                entries = []
                m = re.search(r'"items"\s*:\s*\[', text)
                if m:
                    i = m.end()
                    depth = 0
                    obj_buf = []
                    in_object = False
                    # scan until matching closing bracket of items array using brace counting
                    while i < len(text):
                        ch = text[i]
                        if ch == '{':
                            in_object = True
                            depth += 1
                            obj_buf.append(ch)
                        elif ch == '}' and in_object:
                            depth -= 1
                            obj_buf.append(ch)
                            if depth == 0:
                                # complete object
                                obj_text = ''.join(obj_buf).strip().lstrip(',')
                                try:
                                    entries.append(uj.loads(obj_text))
                                except Exception:
                                    try:
                                        entries.append(json.loads(obj_text))
                                    except Exception:
                                        pass
                                obj_buf = []
                                in_object = False
                        else:
                            if in_object:
                                obj_buf.append(ch)
                            else:
                                # check for end of items array
                                if ch == ']':
                                    break
                        i += 1
        except Exception:
            # try line-delimited JSON
            entries = []
            try:
                with open(repo_path, 'r', encoding='utf8', errors='ignore') as fh:
                    for line in fh:
                        line=line.strip()
                        if not line:
                            continue
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            # skip malformed lines
                            continue
            except Exception:
                entries = []

        # convert entries into chunks
        for e in entries:
            # common fields: story, content, text, title, headline
            text_fields = []
            for key in ('story', 'content', 'text', 'body', 'article'):
                v = e.get(key) if isinstance(e, dict) else None
                if isinstance(v, str) and v.strip():
                    text_fields.append(v.strip())
            # also include title/headline
            title = None
            if isinstance(e, dict):
                title = e.get('title') or e.get('headline') or ''
            text_combined = (title or '') + '\n' + '\n'.join(text_fields)
            if text_combined.strip():
                # split into overlapping character chunks
                i = 0
                while i < len(text_combined):
                    chunk = text_combined[i:i+chunk_size]
                    if len(chunk.strip())>20:
                        chunks.append({'source': repo_path.name, 'text': chunk})
                    i += chunk_size - overlap
        # deduplicate and return
        unique = []
        seen = set()
        for c in chunks:
            s = c['source'] + ':' + (c['text'][:100])
            if s not in seen:
                seen.add(s)
                unique.append(c)
        return unique

    files = list(Path(repo_dir).rglob("*.json")) + list(Path(repo_dir).rglob("*.md")) + list(Path(repo_dir).rglob("*.txt"))
    for f in files:
        try:
            text = f.read_text(encoding="utf8", errors="ignore")
        except Exception:
            continue
        # if json and Postman collection, attempt to extract request descriptions
        if f.suffix.lower() == ".json":
            try:
                j = json.loads(text)
                # Postman collections often have "item" arrays with "request/description/description"
                def extract_from_obj(obj):
                    out = []
                    if isinstance(obj, dict):
                        for k,v in obj.items():
                            if k in ("description","name","request","response","event","item"):
                                out.append(v)
                            elif isinstance(v,(dict,list)):
                                out += extract_from_obj(v)
                    elif isinstance(obj, list):
                        for it in obj:
                            out += extract_from_obj(it)
                    return out
                extracted = extract_from_obj(j)
                flat = []
                def flatten(x):
                    if isinstance(x, str):
                        flat.append(x)
                    elif isinstance(x,dict) or isinstance(x,list):
                        for y in (x if isinstance(x,list) else x.values()):
                            flatten(y)
                flatten(extracted)
                text = "\n".join([t for t in flat if isinstance(t,str)])
            except Exception:
                pass
        # normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # split into overlapping chunks by characters
        i = 0
        while i < len(text):
            chunk = text[i:i+chunk_size]
            if len(chunk.strip())>20:
                chunks.append({"source": str(f.relative_to(repo_dir)), "text": chunk})
            i += chunk_size - overlap
    # also add README if exists
    readme = Path(repo_dir)/"README.md"
    if readme.exists():
        t = readme.read_text(encoding="utf8", errors="ignore")
        chunks.append({"source":"README.md","text":t[:1000]})
    # deduplicate small
    unique = []
    seen = set()
    for c in chunks:
        s = c["source"] + ":" + (c["text"][:100])
        if s not in seen:
            seen.add(s)
            unique.append(c)
    return unique

def build_index(embeddings, dim):
    xb = np.vstack(embeddings).astype('float32')
    if _HAS_FAISS:
        index = faiss.IndexFlatIP(dim)  # use inner product with normalized vectors
        faiss.normalize_L2(xb)
        index.add(xb)
        return index
    else:
        # fallback: use sklearn NearestNeighbors with cosine distance
        nbrs = NearestNeighbors(n_neighbors=10, metric='cosine')
        nbrs.fit(xb)
        return nbrs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Query string")
    ap.add_argument("--k", type=int, default=5, help="Top-k results")
    ap.add_argument("--repo_dir", default="./postman-twitter-api", help="Local path (or will clone)")
    ap.add_argument("--max_chunks", type=int, default=None, help="Limit number of chunks to process (for testing)")
    ap.add_argument("--overwrite_index", action="store_true")
    args = ap.parse_args()

    # If repo_dir is a file, use it directly; otherwise clone/download
    repo_path = Path(args.repo_dir)
    if repo_path.is_file():
        repo = repo_path
    else:
        repo = clone_or_download(args.repo_dir)
    chunks = collect_text_chunks(repo)
    
    # Limit chunks if requested
    if args.max_chunks is not None and len(chunks) > args.max_chunks:
        chunks = chunks[:args.max_chunks]
    
    if len(chunks) == 0:
        print("[]")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    # compute embeddings in batches
    batch_size = 64
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        em = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(em)
    embeddings = np.vstack(embeddings)
    dim = embeddings.shape[1]
    # build index (FAISS if available, otherwise sklearn NearestNeighbors)
    if _HAS_FAISS:
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
    else:
        index = NearestNeighbors(n_neighbors=min(len(embeddings), args.k), metric='cosine')
        index.fit(embeddings.astype('float32'))

    # query embedding
    q_emb = model.encode([args.query], convert_to_numpy=True)
    q_emb = q_emb.astype('float32')
    if _HAS_FAISS:
        # FAISS expects normalized vectors when using inner product for cosine similarity
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, args.k)
    else:
        # sklearn returns cosine distances (0..2), convert to similarity
        distances, indices = index.kneighbors(q_emb, n_neighbors=min(args.k, len(embeddings)))
        D = np.asarray([1.0 - distances[0]])
        I = np.asarray([indices[0]])
    results = []
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": int(idx),
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"][:1000]
        })
    # print JSON to stdout
    print(uj.dumps(results, indent=2))

if __name__ == "__main__":
    main()
