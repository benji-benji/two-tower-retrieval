# 04_server.py

import torch
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from load_glove import load_glove
import models
import dataset

# ---- Config ----
embedding_dim = 100
glove_path = './glove_embeddings/glove_embeddings_6B_100d_w2v.txt'
tokenizer_path = './corpus/tokeniser.pkl'
checkpoint_path = './checkpoints/2025_06_20__15_15_32.19.350.two.pth'

# ---- Device ----
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Tokenizer ----
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
words_to_ids = tokenizer['words_to_idx']
ids_to_words = tokenizer['ids_to_words']

# ---- Embedding Layer ----
embeddings = load_glove(glove_path, words_to_ids, embedding_dim)
embedding_layer = torch.nn.Embedding.from_pretrained(embeddings, freeze=True)

# ---- Model ----
model = models.Towers(embedding_dim).to(dev)
model.load_state_dict(torch.load(checkpoint_path, map_location=dev))
model.eval()

# ---- Dataset & Document Embeddings ----
ds = dataset.Triplets(embedding_layer, words_to_ids)
doc_embeddings = torch.stack([
    model.doc(ds.to_emb(ds.docs[k]).to(dev))
    for k in ds.d_keys
])

# ---- FastAPI Setup ----
app = FastAPI()

# (Optional) Enable CORS for testing from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search")
async def search(q: str):
    if not q.strip():
        return []
    
    query_emb = ds.to_emb(q)
    if query_emb is None:
        return []
    
    query_vector = model.qry(query_emb.to(dev))
    sims = torch.nn.functional.cosine_similarity(query_vector, doc_embeddings)
    top_scores, top_indices = torch.topk(sims, k=4)

    results = [
        {
            "score": round(score.item(), 4),
            "doc": ds.docs[ds.d_keys[i.item()]]
        }
        for score, i in zip(top_scores, top_indices)
    ]
    return results
