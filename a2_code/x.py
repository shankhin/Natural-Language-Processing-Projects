import torch.nn.functional as F
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

c = list()
for i in range(10):
    c.append('#1f77b4')
for i in range(11,21):
    c.append('#d62728')
print(len(c))
print(c[9],c[20])
# def project(a: np.array, b: np.array) -> "tuple[float, np.array]":
#     # TODO: your work here
#     numer = np.dot(a,b)
#     denom = np.dot(b,b)
#     scalar = np.divide(numer,denom)
#     vector_projection = np.multiply(scalar,b)
#     return scalar, vector_projection

# a = np.array([1,2,3])
# b = np.array([2,3,4])
# c = np.array([6,7,8])
# np.random.seed(21)
# z = np.random.rand(3,3)
# word_to_embedding = {
#     "professor" : a,
#     "car": b,
#     "mechanic" : c
# }
# words = list()
# words.append("professor")
# words.append("car")
# words.append("mechanic")

# word_emb = list()
# for word in words:
#     word_emb.append(word_to_embedding[word])
# print(word_emb)

# scalar_list = list()
# for i in range(z.shape[0]):
#     s, v = project(word_emb[i],z[i])
#     scalar_list.append(s)
# print(scalar_list)

# dicts = {}
# for i,x in enumerate(words):
#     dicts[x] = scalar_list[i]
# values_list = list(dicts.values())
# values_list.sort()
# print(dicts)
# newdicts = {}
# for i,x in enumerate(words):
#     newdicts[x] = values_list[i]
# print(newdicts)
# l = list(newdicts.keys())
# top_extreme_words = l[:2]
# print(top_extreme_words)
# l.reverse()
# bottom_extreme_words = l[:2]
# print(bottom_extreme_words)
