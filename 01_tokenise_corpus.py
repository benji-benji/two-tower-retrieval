import pickle
import collections 
from collections import Counter

with open('./corpus/text8.txt') as f: text8: str =f.read()
with open('./corpus/msmarco.txt') as f: msmarco: str =f.read()

print(type(msmarco))
print(type(text8))

def preprocess(text: str) -> list[str]:
  text = text.lower()
  text = text.replace('.',  ' <PERIOD> ')
  text = text.replace(',',  ' <COMMA> ')
  text = text.replace('"',  ' <QUOTATION_MARK> ')
  text = text.replace('“',  ' <QUOTATION_MARK> ')
  text = text.replace('”',  ' <QUOTATION_MARK> ')
  text = text.replace(';',  ' <SEMICOLON> ')
  text = text.replace('!',  ' <EXCLAMATION_MARK> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace('(',  ' <LEFT_PAREN> ')
  text = text.replace(')',  ' <RIGHT_PAREN> ')
  text = text.replace('--', ' <HYPHENS> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace(':',  ' <COLON> ')
  text = text.replace("'",  ' <APOSTROPHE> ')
  text = text.replace("’",  ' <APOSTROPHE> ')
  words = text.split() 
  stats = collections.Counter(words)
  words = [word for word in words if stats[word] > 5]
  return words  

def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  word_counts = collections.Counter(words)
  vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
  int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
  int_to_vocab[0] = '<PAD>'
  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab

####### SET UP FOR PRETRAINED W2V EMBEDDINGS #########

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
special_tokens = [PAD_TOKEN, UNK_TOKEN]

#corpus: list[str] = preprocess(msmarco)



# WORD2VEC IMPLEMENTATION

#words_to_ids, ids_to_words = create_lookup_tables(corpus)
#tokeniser = { "words_to_ids": words_to_ids, "ids_to_words": ids_to_words }
#tokens: list[int] = [words_to_ids[word] for word in corpus]

#with open('./corpus/tokens.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(map(str, tokens)))
#with open('./corpus/tokeniser.pkl', 'wb') as f: pickle.dump(tokeniser, f)

#print("VOCAB:", len(words_to_ids))

####### PRETRAINED W2K TOKENISATION ########

texts = msmarco.splitlines()

def build_vocab(texts, glove_path):
    # Load GloVe vocabulary
    glove_words = set()
    with open(glove_path, 'r') as f:
        for line in f:
            word = line.split()[0]
            glove_words.add(word)
    
    # Create vocab with special tokens + GloVe words found in corpus
    word_counts = Counter()
    
    for text in texts:
        tokens = preprocess(text)
        word_counts.update(tokens)
    
    vocab = special_tokens + [word for word in glove_words if word in word_counts]
    # words_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab # , word_to_idx

#vocab, word_to_idx = build_vocab(texts, glove_path='./glove_embeddings/glove.6B.100d.word2vec.embeddings.txt')

full_vocab = build_vocab(texts, glove_path='./glove_embeddings/glove_embeddings_6B_100d_w2v.txt')
print("Length of full vocab:", len(full_vocab))
words_to_idx, ids_to_words = create_lookup_tables(full_vocab)
tokenizer = { "words_to_idx": words_to_idx, "ids_to_words": ids_to_words }
tokens: list[int] = [words_to_idx[word] for word in full_vocab]
with open('./corpus/tokens.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(map(str, tokens)))
with open('./corpus/tokeniser.pkl', 'wb') as f: pickle.dump(tokenizer, f)

