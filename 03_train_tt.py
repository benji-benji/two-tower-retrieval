import torch
import models
import pickle
import dataset
import datetime
import wandb
import tqdm
from load_glove import load_glove
import torch.nn as nn 

# HYPERPRAMETERS

#total_data_size =
num_epochs = 20
batch_size = 256
learning_rate = 0.002
loss_margin = 0.7
embedding_dim = 100

torch.manual_seed(42)
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
words_to_idx, ids_to_words = tkns['words_to_idx'], tkns['ids_to_words']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

glove_path = './glove_embeddings/glove_embeddings_6B_100d_w2v.txt'
embeddings = load_glove(glove_path, words_to_idx, embedding_dim=embedding_dim)
embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)

ds = dataset.Triplets(embedding_layer, words_to_idx)
print("Number of triplets:", len(ds))
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

two = models.Towers(glove_dim=embedding_dim).to(dev)
torch.save(two.state_dict(), f'./checkpoints/{ts}.0.0.two.pth')
print('two:', sum(p.numel() for p in two.parameters())) # 40,400
opt = torch.optim.Adam(two.parameters(), lr=learning_rate)


wandb.init(project='mlx6-week-02-two',
           config={
             "num_epochs": num_epochs,
             "batch_size": batch_size,
             "learning_rate": learning_rate,
             "loss_margin": loss_margin,
             "embedding_dim": embedding_dim,
           })

for epoch in range(num_epochs):
  prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
  for idx, (qry, pos, neg) in enumerate(prgs):
    
    qry, pos, neg = qry.to(dev), pos.to(dev), neg.to(dev)
    
    loss = two(qry, pos, neg, mrg=loss_margin) # calculate loss of the triplet
    
    opt.zero_grad() # zeros the gradients after each pass
    loss.backward() # computes the gradients of the loss, in order to reduce loss
    opt.step() # updates the new parameters 
    
    wandb.log({'loss': loss.item()})
    if idx % 50 == 0: torch.save(two.state_dict(), f'./checkpoints/{ts}.{epoch}.{idx}.two.pth')

#
wandb.finish()