from transformers import ElectraTokenizer, ElectraModel
import torch

tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = ElectraModel.from_pretrained('google/electra-small-discriminator')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cpu")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state