import datasets 
import requests 
import os
from datasets import get_dataset_config_names, get_dataset_split_names
import random

#print(get_dataset_config_names("microsoft/ms_marco"))
#print(get_dataset_split_names("microsoft/ms_marco", "triplet"))
random.seed (42)
full_dataset = datasets.load_dataset('microsoft/ms_marco','v1.1')

all_docs = [passage for s in full_dataset.keys() for e in full_dataset[s] for passage in e['passages']['passage_text']]

complete_qrs = [e['query'] for s in full_dataset.keys()for e in full_dataset[s]]
eval_qrys = random.sample(complete_qrs, 50)
all_qrs = [e['query'] for s in full_dataset.keys() for e in full_dataset[s] if e['query'] not in eval_qrys]
#all_qrs = complete_qrs - eval_qrys

print("all_qrs length:", len(all_qrs))
print("eval_qrys length:", len(eval_qrys))
print("Difference between all_qrs and eval_qrys:", len(all_qrs) - len(eval_qrys))

# SENSE CHECK PRINTOUTS 

print("## Documents loaded ##")
print("\n")
print("example doc:\n"+(all_docs[1]))
print("number of docs:"+str(len(all_docs)))
print("\n")    
print("example query:\n"+(all_qrs[1]))
print("number of queries:"+str(len(all_qrs)))                                  

# COMBINE & SAVE TXT FILES

with open('./corpus/msmarco.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(set(all_docs +all_qrs)))

r = requests.get('https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8')
with open('./corpus/text8.txt', 'wb') as f: f.write(r.content)

r = requests.get('https://huggingface.co/datasets/nodozi/glove.6B.100d.word2vec.txt/resolve/main/glove.6B.100d.word2vec.txt?download=true')
with open('./glove_embeddings/glove_embeddings_6B_100d_w2v.txt', 'wb') as f: f.write(r.content)

# CHECK FILES EXIST
print("\nChecking files saved...\n")  
msmarco_path = "./corpus/msmarco.txt"
text8_path = "./corpus/text8.txt"
print ("msmarco.txt file exists:", os.path.exists(msmarco_path))
print ("text8.txt file exists:", os.path.exists(text8_path))

