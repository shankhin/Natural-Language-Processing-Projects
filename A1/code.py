from typing import Union, Iterable, Callable
import random

import torch
import torch.nn as nn



def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.

    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    """
    Tokenize the text into individual words (nested list of string),
    where the inner list represent a single example.

    Parameters
    ----------
    text: list of strings
        Your cleaned text data (either premise or hypothesis).
    max_length: int, optional
        The maximum length of the sequence. If None, it will be
        the maximum length of the dataset.
    normalize: bool, default True
        Whether to normalize the text before tokenizing (i.e. lower
        case, remove punctuations)
    Returns
    -------
    list of list of strings
        The same text data, but tokenized by space.

    Examples
    --------
    >>> tokenize(['Hello, world!', 'This is a test.'], normalize=True)
    [['hello', 'world'], ['this', 'is', 'a', 'test']]
    """
    import re

    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]

    return [t.split()[:max_length] for t in text]


def build_word_counts(token_list: "list[list[str]]") -> "dict[str, int]":
    """
    This builds a dictionary that keeps track of how often each word appears
    in the dataset.

    Parameters
    ----------
    token_list: list of list of strings
        The list of tokens obtained from tokenize().

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.

    Notes
    -----
    If you have  multiple lists, you should concatenate them before using
    this function, e.g. generate_mapping(list1 + list2 + list3)
    """
    word_counts = {}

    for words in token_list:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    """
    Builds an index map that converts a word into an integer that can be
    accepted by our model.

    Parameters
    ----------
    word_counts: dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.
    max_words: int, optional
        The maximum number of words to be included in the index map. By
        default, it is None, which means all words are taken into account.

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        index in the embedding.
    """

    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]

    return {word: ix for ix, word in enumerate(sorted_words)}


def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    """
    Converts a nested list of tokens to a nested list of indices using
    the index map.

    Parameters
    ----------
    tokens: list of list of strings
        The list of tokens obtained from tokenize().
    index_map: dict of {str: int}
        The index map from build_index_map().

    Returns
    -------
    list of list of int
        The same tokens, but converted into indices.

    Notes
    -----
    Words that have not been seen are ignored.
    """
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


### 1.1 Batching, shuffling, iteration
def build_loader(
    data_dict: dict, batch_size: int = 64, shuffle: bool = False
) -> Callable[[], Iterable[dict]]:
    # TODO: Your code here
    random.seed(2022)
    torch.manual_seed(2022)

    def loader():
        l = len(data_dict['premise'])
        for i in range(0, l, batch_size):
            batch_dict = {}
            for key, value in data_dict.items():
                batch_list = value[i:min(i + batch_size, l)]
                batch_dict[key] = batch_list
                if(shuffle == True):
                    random.shuffle(batch_list)
            yield batch_dict
    return loader


### 1.2 Converting a batch into inputs
def convert_to_tensors(text_indices: "list[list[int]]") -> torch.Tensor:
    # TODO: Your code here
    #Padding with 0 using max length of sentence in list
    list_len = [len(i) for i in text_indices]
    max_len = max(list_len)
    for i in text_indices:
        i += [0] * (max_len - len(i))     
    return torch.tensor(text_indices, dtype=torch.int32)


### 2.1 Design a logistic model with embedding and pooling
def max_pool(x: torch.Tensor) -> torch.Tensor:
    # # TODO: Your code here
    # u = torch.empty(x.size()[0],x.size()[2])
    # for i in range(0,x.size()[0]):
    #     for j in range(0,x.size()[2]):
    #         u[i,j]  = max(x[i,0:,j])
    return torch.amax(x, dim =1)


class PooledLogisticRegression(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        # TODO: Your code here
        self.embedding = embedding
        self.sigmoid = nn.Sigmoid()
        self.layer_pred = nn.Linear(2*embedding.weight.size()[1],1)

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()

        # TODO: Your code 
        premise = emb(premise)
        hypothesis = emb(hypothesis)
        premise_max = max_pool(premise)
        hypothesis_max = max_pool(hypothesis)
        x = torch.cat((premise_max,hypothesis_max),1)
        x = sigmoid(layer_pred(x))
        x = torch.flatten(x)
        # x = (x>0.5).float()
        return x

### 2.2 Choose an optimizer and a loss function
def assign_optimizer(model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    # TODO: Your code here
    return torch.optim.SGD(model.parameters(), **kwargs)


def bce_loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # TODO: Your code here
    return torch.mean(-1*torch.add(y*torch.log(y_pred),(1-y)*torch.log(1-y_pred)))
            
### 2.3 Forward and backward pass
def forward_pass(model: nn.Module, batch: dict, device="cpu"):
    # TODO: Your code here
    premise_tensor = convert_to_tensors(batch['premise'])
    hypothesis_tensor = convert_to_tensors(batch['hypothesis'])
    model.to(device)
    y_pred = model.forward(premise_tensor,hypothesis_tensor)
    return y_pred


def backward_pass(
    optimizer: torch.optim.Optimizer, y: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    # TODO: Your code here
    optimizer.zero_grad()
    loss = bce_loss(y,y_pred)
    optimizer.step()
    return loss


### 2.4 Evaluation
def f1_score(y: torch.Tensor, y_pred: torch.Tensor, threshold=0.5) -> torch.Tensor:
    # TODO: Your code here
    if threshold is not None:
        y_pred = (y_pred>threshold).int()
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(y.size(dim=0)):
        if ((y[i].item() == 1) and (y_pred[i].item() == 1)):
            tp+=1
        elif ((y[i].item() == 0) and (y_pred[i].item() == 0)):
            tn+=1
        elif ((y[i].item() == 1) and (y_pred[i].item() == 0)):
            fn+=1
        elif ((y[i].item() == 0) and (y_pred[i].item() == 1)):
            fp+=1
    if((tp+fn==0)or(tp+fp==0)):
        recall = 0
        precision = 0
        f1 = 0    
    else:
        recall = (tp)/(tp+fn)
        precision = (tp)/(tp+fp)
        f1 = (2*precision*recall)/(precision+recall)
    return torch.tensor(f1)

### 2.5 Train loop
def eval_run(
    model: nn.Module, loader: Callable[[], Iterable[dict]], device: str = "cpu"
):
    # TODO: Your code here
    model.eval()
    y_true = list()
    y_pred = list()
    for batch in loader():
        y = forward_pass(model,batch,device)
        y_pred.append(y)
        labels = batch['label']
        y_true.append(labels)
    y_true = [j for sub in y_true for j in sub]
    y_pred = [j for sub in y_pred for j in sub]
    y_true = torch.FloatTensor(y_true)
    y_pred = torch.FloatTensor(y_pred)
    return y_true, y_pred

def train_loop(
    model: nn.Module,
    train_loader,
    valid_loader,
    optimizer,
    n_epochs: int = 3,
    device: str = "cpu",
):
    # TODO: Your code here
    f1_list = list()
    for epoch in range(n_epochs):
        #train
        model.train()
        for batch in train_loader():
            optimizer.zero_grad()
            y = torch.FloatTensor(batch['label'])
            y_pred = forward_pass(model,batch,device)
            loss = bce_loss(y,y_pred)
            loss.backward()
            optimizer.step()
        #eval
        y, y_pred = eval_run(model,valid_loader,device)
        f1 = f1_score(y,y_pred).item()
        f1_list.append(f1)
    return f1_list


### 3.1
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int):
        super().__init__()
        # TODO: continue here
        self.embedding = embedding
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()
        self.ff_layer = nn.Linear(2*embedding.weight.size()[1],hidden_size)      
        self.layer_pred = nn.Linear(hidden_size,1)


    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layer(self):
        return self.ff_layer

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layer = self.get_ff_layer()
        act = self.get_activation()

        # TODO: continue here

        premise = emb(premise)
        hypothesis = emb(hypothesis)
        premise_max = max_pool(premise)
        hypothesis_max = max_pool(hypothesis)
        x = torch.cat((premise_max,hypothesis_max),1)
        x = act(ff_layer(x))
        x = sigmoid(layer_pred(x))
        x = torch.flatten(x)
        return x


### 3.2
class DeepNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.embedding = embedding
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()
        self.ff_layers = nn.ModuleList([nn.Linear(2*embedding.weight.size()[1], hidden_size)])
        self.ff_layers.extend([nn.Linear(hidden_size,hidden_size) for i in range(num_layers-1)])  
        self.layer_pred = nn.Linear(hidden_size,1)

        # TODO: continue here

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layers(self):
        return self.ff_layers

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layers = self.get_ff_layers()
        act = self.get_activation()

        # TODO: continue here

        premise = emb(premise)
        hypothesis = emb(hypothesis)
        premise_max = max_pool(premise)
        hypothesis_max = max_pool(hypothesis)
        x = torch.cat((premise_max,hypothesis_max),1)
        for i in range(len(ff_layers)):
            x = act(ff_layers[i](x))
        x = sigmoid(layer_pred(x))
        x = torch.flatten(x)
        return x


if __name__ == "__main__":
    # If you have any code to test or train your model, do it BELOW!
    # Seeds to ensure reproducibility
    random.seed(2022)
    torch.manual_seed(2022)

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets("data")

    train_tokens = {
        "premise": tokenize(train_raw["premise"], normalize=True, max_length=64),
        "hypothesis": tokenize(train_raw["hypothesis"], normalize=True, max_length=64),
    }

    valid_tokens = {
        "premise": tokenize(valid_raw["premise"], normalize=True, max_length=64),
        "hypothesis": tokenize(valid_raw["hypothesis"], normalize = True, max_length=64),
    }

    word_counts = build_word_counts(
        train_tokens["premise"]
        + train_tokens["hypothesis"]
        + valid_tokens["premise"]
        + valid_tokens["hypothesis"]
    )
    index_map = build_index_map(word_counts, max_words=10000)

    train_indices = {
        "label": train_raw["label"],
        "premise": tokens_to_ix(train_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(train_tokens["hypothesis"], index_map)
    }

    valid_indices = {
        "label": valid_raw["label"],
        "premise": tokens_to_ix(valid_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(valid_tokens["hypothesis"], index_map)
    }
    print(train_raw["hypothesis"][0])
    print(train_tokens["hypothesis"][0])
    print(train_indices["hypothesis"][0])

    # 1.1

    train_loader = build_loader(train_raw)
    valid_loader = build_loader(valid_raw)

    # 1.2
    batch = next(train_loader())

    y = "your code here"

    # 2.1
    xembedding = "your code here"
    model = "your code here"

    # 2.2
    optimizer = "your code here"

    # 2.3
    y_pred = "your code here"
    loss = "your code here"

    # 2.4
    score = "your code here"

    # 2.5
    n_epochs = 2

    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"

    # 3.1
    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"

    # 3.2
    embedding = "your code here"
    model = "your code here"
    optimizer = "your code here"

    scores = "your code here"
