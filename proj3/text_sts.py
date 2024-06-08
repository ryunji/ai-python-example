# STEP 1
from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
# STEP 2
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
# STEP 3
# sentences = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]
sentence1 = "he weather is lovely today."
sentence2 = "It's so sunny outside!"
sentence3 = "He drove to the stadium."

# 2. Calculate embeddings by calling model.encode()
# STEP 4
#embeddings = model.encode(sentences)
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
#print(embeddings.shape)
print(embedding1.shape)
print(embedding2.shape)
# [3, 384]

# 3. Calculate the embedding similarities
#similarities = model.similarity(embeddings, embeddings)
similarities = model.similarity(embedding1, embedding2)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])