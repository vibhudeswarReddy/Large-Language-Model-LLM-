from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text1 = input("Enter first sentence: ")
text2 = input("Enter second sentence: ")

t1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
t2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    e1 = model(**t1).last_hidden_state.mean(dim=1)
    e2 = model(**t2).last_hidden_state.mean(dim=1)

score = cosine_similarity(e1, e2)
print("Similarity:", score[0][0])
