This repository contains all my personal Natural Language Processing Projects done both as part of my schooling and on my own. Further description of the projects can be found in the reports of the respective projects.
# Project 1 - Natural Language Inference
In this assignemnt, we were given a dataset of sentences called the “premise”, and we wanted to predict if another datatset of sentences, the “hypothesis”, either entails or does not entail the corresponding premise. Saying that the premise entails a hypothesis means that, if one read the premise, then one would infer that the hypothesis is true.
# Project 2 - Learning Word Embeddings & Mitigating Gender Bias in Word Emebeddings
In this assignment, we were asked to implement two models from the word2vec project: Continuous Bag-of-Words (CBOW) and Skip-Gram and then we investigated techniques for measuring and mitigating gender bias in word embeddings.
# Project 3 - RNNs
In this assignment, we implemented a character-based generative RNN-based language model. The model will accept a stream of characters and learn to generate a distribution of the next character based on the previous context, over the model’s vocabulary (individual characters instead of words as you may be used to). And then, a similar process was applied to form a Unidirectional as well as Bidirectional LSTM from scratch.
# Project 4 -
In this assignment, we used the NLI training data (same as Project 1) to finetune a DistilBERT model and predict whether a premise entails a hypothesis or not. And for the next part, following [Lester et al (2021)](https://arxiv.org/abs/2104.08691), we implemented their prompt tuning method (called “soft prompts”). Finally, we implemented a model similar to Dense Passage Retrieval for Open-Domain Question Answering by [Karpukhin et al (2020)](https://arxiv.org/abs/2004.04906).
