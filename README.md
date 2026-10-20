# Two-tower document retrieval

A two-tower model that ranks MS MARCO passages by relevance to a search query. Query and passage are embedded by separate towers and compared by cosine similarity, trained with a triplet loss.

Machine Learning Institute residency, week 2 (June 2025). Team Backprop Bunch; this repository is my implementation.

## How it works

1. **Embed the text.** Each query and passage is tokenised, every token is mapped to a frozen 100-dimensional GloVe vector (6B, 100d), and the vectors are averaged into one 100-d vector.
2. **Two towers.** A query tower and a document tower, each `Linear(100→100) → ReLU → Linear(100→100)`. 40,400 trainable parameters in total.
3. **Triplet loss.** For a query *q*, its relevant passage *p⁺* and a randomly sampled passage *p⁻*:

   `loss = max(0, d(q, p⁺) − d(q, p⁻) + margin)`, with `d(a, b) = 1 − cos(a, b)` and margin 0.7.

   The loss is zero once the relevant passage is closer to the query than the random one by at least the margin.
4. **Search.** All passages are encoded once by the document tower. A query is encoded by the query tower and passages are ranked by cosine similarity (FastAPI endpoint, Streamlit front end).

Training: Adam, learning rate 0.002, batch size 256, 20 epochs, logged to Weights & Biases.

## Results

50 queries held out from training. For each, the relevant passage is ranked against 9 randomly drawn passages (10 candidates).

| Model | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| Random ranking | 0.10 | 0.30 | 0.29 |
| Averaged GloVe, no towers | 0.86 | 0.94 | 0.91 |
| **Two towers (trained)** | **0.88** | **0.98** | **0.93** |

**What this shows:** most of the performance comes from the averaged GloVe vectors, not from training. The towers add 2 points of Recall@1. Two reasons the gap is small:

- **The negatives are easy.** A random MS MARCO passage is usually off-topic, so word overlap alone separates it from the relevant one. Hard negatives (passages retrieved for a similar query) would test the towers properly.
- **Averaging discards word order.** Both towers see a bag of words, so they can only reweight the GloVe dimensions.

Next steps: evaluate against hard negatives and the full passage pool, and replace averaging with a sequence encoder.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python 00_download_merge.py    # MS MARCO v1.1, text8, GloVe vectors → corpus/, glove_embeddings/
python 01_tokenise_corpus.py   # vocabulary: GloVe words that appear in the corpus
python 02_doc_qry_data.py      # query → passage dictionaries, 50 held-out eval queries
python 03_train_tt.py          # train; checkpoints → checkpoints/, logs → W&B
python eval.py                 # Recall@1/3/5, MRR on the held-out queries

uvicorn 04_server:app          # search API on :8000 (set checkpoint_path in 04_server.py)
streamlit run streamlit_app.py # search front end
```

Data, embeddings and checkpoints are not committed; the scripts download or create them.

## Files

| File | Purpose |
|---|---|
| `models.py` | Query tower, document tower, triplet loss |
| `dataset.py` | Triplet dataset: query, relevant passage, random passage |
| `load_glove.py` | Builds the frozen embedding matrix from GloVe |
| `03_train_tt.py` | Training loop |
| `eval.py` | Ranking evaluation |
| `04_server.py`, `streamlit_app.py` | Search API and front end |
