# COMP-599-Project: Political Debiasing on Fake News Classification using BiLSTM + ANN

Model Training

Completed notebooks since December 13th, 2022

Download the zipped folder and obtain the contents and run the notebooks on Google Colaboratory.

To run the training, use the notebooks with (Final) in it to test each model (no biasing vs with biasing).

To notebook contains code to load in the ISOT Dataset, which can be downloaded here: https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php

To download glove embeddings use:

```
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
```

You can run the model training using the notebook, to see the collected results in the paper, you can also consult the collected training and test results in the data processing results google document.

Bias Analysis

To see the anlaysis of direct-bias and dataset political component, run the bias_calculation notebook.

The notebook's cell outputs show the directbias of the glove 300D embeddings and also the debiased glove 300D embeddings. 

The second half of the bias_calculation notebook contains the steps to obtain the component calculation for the true and false portions of the ISOT dataset.

Authored by Zhe Fan, Shankhin Brahmavar
