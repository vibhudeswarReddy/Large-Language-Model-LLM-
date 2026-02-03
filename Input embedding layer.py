import torch
import torch.nn as nn
# sentence
sentence = ['I', 'love', 'AI']
# vocabulary
vocab = {'I': 0, 'love': 1, 'AI': 2}
# convert words to indices
word_indices = torch.tensor([vocab[word] for word in sentence])
# position indices (0, 1, 2)
position_indices = torch.arange(len(sentence))
# embedding size
embedding_dim = 5
# word embedding layer
word_embedding = nn.Embedding(num_embeddings=3, embedding_dim=embedding_dim)
# position embedding layer
position_embedding = nn.Embedding(num_embeddings=3, embedding_dim=embedding_dim)
# get embeddings
word_vectors = word_embedding(word_indices)
position_vectors = position_embedding(position_indices)
# combine word + position embeddings
final_embeddings = word_vectors + position_vectors
print(final_embeddings)
