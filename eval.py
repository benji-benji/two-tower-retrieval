def similarity_eval(model_path, tokenizer_path, evals_path, docs_path, num_queries=5, max_len=50):
    '''
    Fixed evaluation function with comprehensive metrics:
    - Recall@K (for K=1,3,5)
    - Mean Reciprocal Rank (MRR)
    - Average rank of correct document
    '''
    
    import random
    import torch
    import numpy as np
    from torch.nn.functional import cosine_similarity
    import pickle
    import models
    from dataset import Triplets
    preprocess = lambda t: Triplets.preprocess(None, t)
    random.seed(0)
    from load_glove import load_glove

    # Load model
    embedding_dim = 100
    model = models.Towers(embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    words_to_idx = tokenizer['words_to_idx']

    # Load GloVe embeddings
    glove_path = './glove_embeddings/glove_embeddings_6B_100d_w2v.txt'
    embeddings = load_glove(glove_path, words_to_idx, embedding_dim=embedding_dim)
    embedding_layer = torch.nn.Embedding.from_pretrained(embeddings, freeze=True)

    # Load eval queries and docs
    with open(evals_path, 'rb') as f:
        eval_data = pickle.load(f)

    # Load all docs
    with open(docs_path, 'rb') as f:
        all_docs = pickle.load(f)

    def embed_text(text, words_to_idx, embedding_layer, max_len=50):
        """Convert text to averaged embeddings tensor"""
        tokens = preprocess(text)
        indices = [words_to_idx.get(token, words_to_idx.get('<UNK>', 0)) for token in tokens[:max_len]]
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        embeddings = embedding_layer(indices_tensor)
        avg_embedding = embeddings.mean(dim=0, keepdim=True)
        return avg_embedding

    # Select random eval queries and their correct docs
    selected = random.sample(list(eval_data.items()), num_queries)
    
    # Extract query texts and correct doc IDs
    queries_text = [item[1]['text'] for item in selected]
    correct_doc_ids = [item[1]['docs'][0] for item in selected]
    
    # Get correct doc texts
    correct_docs_text = [all_docs[doc_id] for doc_id in correct_doc_ids]
    
    # Select random docs (excluding correct ones)
    candidate_doc_ids = list(set(all_docs.keys()) - set(correct_doc_ids))
    
    # Metrics storage
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    reciprocal_ranks = []
    avg_ranks = []

    print("Starting evaluation...\n")
    print(f"Evaluating on {num_queries} queries with 1 correct and 9 random documents each")

    for i, (query_text, correct_doc_id) in enumerate(zip(queries_text, correct_doc_ids)):
        # Convert text to embeddings
        query_embedding = embed_text(query_text, words_to_idx, embedding_layer, max_len)
        correct_doc_text = all_docs[correct_doc_id]
        correct_doc_embedding = embed_text(correct_doc_text, words_to_idx, embedding_layer, max_len)
        
        with torch.no_grad():
            # Get embeddings from model towers
            query_emb = model.query_tower(query_embedding)
            correct_doc_emb = model.doc_tower(correct_doc_embedding)
            
            # Calculate similarity with correct doc
            sim_correct = cosine_similarity(query_emb, correct_doc_emb, dim=1).item()
            
            print(f"\nQuery {i+1}: {query_text}")
            print(f"  Correct Doc: {correct_doc_id[:16]}... | Cosine Sim: {sim_correct:.3f}")
            
            # Select 9 random docs for this query (for better metrics)
            rand_doc_ids = random.sample(candidate_doc_ids, 9)
            candidate_docs = [correct_doc_id] + rand_doc_ids
            random.shuffle(candidate_docs)  # Important for unbiased ranking
            
            # Calculate similarities for all candidates
            candidate_sims = []
            for doc_id in candidate_docs:
                doc_text = all_docs[doc_id]
                doc_embedding = embed_text(doc_text, words_to_idx, embedding_layer, max_len)
                doc_emb = model.doc_tower(doc_embedding)
                sim = cosine_similarity(query_emb, doc_emb, dim=1).item()
                candidate_sims.append((doc_id, sim))
                
                # Print the first 3 random docs for consistency
                if doc_id in rand_doc_ids[:3]:
                    print(f"  Random Doc {rand_doc_ids.index(doc_id)+1}: {doc_id[:16]}... | Cosine Sim: {sim:.3f}")
            
            # Sort candidates by similarity (descending)
            candidate_sims.sort(key=lambda x: x[1], reverse=True)
            
            # Get ranked list of document IDs
            ranked_docs = [doc_id for doc_id, _ in candidate_sims]
            
            # Find rank of correct document (1-indexed)
            rank = ranked_docs.index(correct_doc_id) + 1
            avg_ranks.append(rank)
            
            # Calculate reciprocal rank for MRR
            reciprocal_rank = 1.0 / rank
            reciprocal_ranks.append(reciprocal_rank)
            
            # Update recall metrics
            if rank <= 1:
                recall_at_1 += 1
            if rank <= 3:
                recall_at_3 += 1
            if rank <= 5:
                recall_at_5 += 1
            
            print(f"  -> Correct doc rank: {rank}/{len(candidate_docs)} | Reciprocal Rank: {reciprocal_rank:.4f}")
    
    # Calculate final metrics
    recall_at_1 /= num_queries
    recall_at_3 /= num_queries
    recall_at_5 /= num_queries
    mrr = np.mean(reciprocal_ranks)
    avg_rank = np.mean(avg_ranks)
    
    # Print comprehensive metrics
    print("\n" + "=" * 60)
    print("Evaluation Summary:")
    print(f"Queries evaluated: {num_queries}")
    print(f"Recall@1: {recall_at_1:.4f} - Proportion where correct doc is top result")
    print(f"Recall@3: {recall_at_3:.4f} - Proportion where correct doc is in top 3")
    print(f"Recall@5: {recall_at_5:.4f} - Proportion where correct doc is in top 5")
    print(f"MRR (Mean Reciprocal Rank): {mrr:.4f} - Average of 1/rank (higher is better)")
    print(f"Average Rank: {avg_rank:.2f} - Average position of correct document (lower is better)")
    print("=" * 60)
    
# Run your evaluation
model_path = "./checkpoints/2025_06_20__15_15_32.19.350.two.pth"
tokenizer_path = "./corpus/tokeniser.pkl"
evals_path = "./corpus/evals.pkl"
docs_path = "./corpus/docs.pkl"

# First run debug to check model behavior
sample_texts = [
    "how does eating an apple help digestion",
    "caffeine supplements disadvantages",
    "protect yourself from emp attack",
    "good foods for gout sufferers",
    "what is a muscle car"
]

print("Running debug check first...")
#debug_model_outputs(model_path, tokenizer_path, sample_texts)

print("\n" + "="*60)
print("Running fixed evaluation...")
print("="*60)
similarity_eval(model_path, tokenizer_path, evals_path, docs_path, num_queries=50)