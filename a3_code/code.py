import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from datasets import load_dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### PROVIDED CODE #####

def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    import re
    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]
    return [t.split()[:max_length] for t in text]

def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]
    return {word: ix for ix, word in enumerate(sorted_words)}

def build_word_counts(dataloader) -> "dict[str, int]":
    word_counts = {}
    for batch in dataloader:
        for words in tokenize(batch['premise'] + batch['hypothesis']):
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


##### END PROVIDED CODE #####

class CharSeqDataloader():
    def __init__(self, filepath, seq_len, examples_per_epoch):
        self.unique_chars = sorted(list(set(open(filepath, 'r').read()))) # fill in
        self.vocab_size = len(self.unique_chars)# fill in
        self.mappings = self.generate_char_mappings(self.unique_chars) # fill in
        self.seq_len = seq_len # fill in
        self.examples_per_epoch = examples_per_epoch

        # your code here
        self.data = list(open(filepath, 'r').read())

    def generate_char_mappings(self, uq):
        char_to_idx = {}
        idx_to_char = {}
        for i in range(len(uq)):
            char_to_idx[uq[i]] = i
            idx_to_char[i] = uq[i]
        full_dict = {"char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char}
        return full_dict
        

    def convert_seq_to_indices(self, seq):
        # your code here
        char_to_idx = self.mappings["char_to_idx"]
        out_seq = list()
        for i in seq:
            out_seq.append(char_to_idx[i])
        return out_seq

    def convert_indices_to_seq(self, seq):
        # your code here
        idx_to_char = self.mappings["idx_to_char"]
        out_seq = list()
        for i in seq:
            out_seq.append(idx_to_char[i])
        return out_seq

    def get_example(self):
        # your code here
        data_idx_list = self.convert_seq_to_indices(self.data)
        for i in range(self.examples_per_epoch):
            data_ptr = np.random.randint(self.seq_len)
            in_seq = data_idx_list[data_ptr : data_ptr + self.seq_len]
            target_seq = data_idx_list[data_ptr + 1 : data_ptr + self.seq_len + 1]
            data_ptr += self.seq_len
            if data_ptr + self.seq_len > len(data_idx_list):
                break
            yield torch.tensor(in_seq), torch.tensor(target_seq)


class CharRNN(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_chars = n_chars
        self.embedding_size = embedding_size
        # your code here
        self.embedding_layer = nn.Embedding(self.n_chars, self.embedding_size)
        self.waa = nn.Linear(self.hidden_size, self.hidden_size)
        self.wax = nn.Linear(self.embedding_size, self.hidden_size)
        self.wya = nn.Linear(self.hidden_size, self.n_chars)

    def rnn_cell(self, i, h):
        # your code here
        h_new = torch.tanh(self.waa(h) + self.wax(i))
        o = self.wya(h_new)
        return o, h_new

    def forward(self, input_seq, hidden = None):
        # your code here
        h = torch.zeros(self.hidden_size) if hidden == None else hidden
        o_list = list()
        i = self.embedding_layer(input_seq)
        for idx in range(len(input_seq)):
            o, h_new = self.rnn_cell(i[idx],h)
            o_list.append(o)
            h = h_new
        out = torch.stack(o_list)
        hidden_last = h_new
        return out, hidden_last

    def get_loss_function(self):
        # your code here
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        # your code here
        return torch.optim.Adam(self.parameters(),lr=lr)
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5):
        # your code here
        i_list = list()
        i_list.append(starting_char)

        h = torch.zeros(self.hidden_size)
        i = torch.tensor(starting_char)

        for idx in range(seq_len):
            i_emb = self.embedding_layer(i)
            z_new, h_new = self.rnn_cell(i_emb,h)
            z_new = F.softmax(z_new/temp, dim=0)
            x = Categorical(z_new)
            i_new = x.sample()
            i_list.append(i_new.item())

            h = h_new
            i = i_new

        return i_list

class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars
        #  your code here
        print(self.n_chars, self.embedding_size, self.hidden_size)
        self.embedding_layer = nn.Embedding(self.n_chars, self.embedding_size)
        self.forget_gate = nn.Linear(self.embedding_size + self.hidden_size, self.hidden_size)
        self.input_gate =  nn.Linear(self.embedding_size + self.hidden_size, self.hidden_size)
        self.output_gate =  nn.Linear(self.embedding_size + self.hidden_size, self.hidden_size)
        self.cell_state_layer = nn.Linear(self.embedding_size + self.hidden_size, self.hidden_size)
        self.fc_output =  nn.Linear(self.hidden_size, self.n_chars)


    def forward(self, input_seq, hidden = None, cell = None):
        # your code here
        h = torch.zeros(self.hidden_size) if hidden == None else hidden
        c = torch.zeros(self.hidden_size) if cell == None else cell

        y_list = list()
        i = self.embedding_layer(input_seq)
        for idx in range(len(input_seq)):
            y, h_new, c_new = self.lstm_cell(i[idx], h, c)
            y_list.append(y)
            h = h_new
            c = c_new

        out_seq = torch.stack(y_list)
        hidden_last = h_new
        cell_last = c_new

        return out_seq, hidden_last, cell_last

    def lstm_cell(self, i, h, c):
        # your code here
        hx = torch.concat((i,h))

        ft = torch.sigmoid(self.forget_gate(hx))
        it = torch.sigmoid(self.input_gate(hx))
        ct = torch.tanh(self.cell_state_layer(hx))
        c_new = ft*c + it*ct
        ot = torch.sigmoid(self.output_gate(hx))
        h_new = ot*torch.tanh(c_new)
        y_new = self.fc_output(h_new)
        return y_new, h_new, c_new

    def get_loss_function(self):
        # your code here
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        # your code here
        return torch.optim.Adam(self.parameters(),lr=lr)

    def sample_sequence(self, starting_char, seq_len, temp=0.5):
        # your code here
        i_list = list()
        i_list.append(starting_char)

        h = torch.zeros(self.hidden_size)
        c = torch.zeros(self.hidden_size)
        i = torch.tensor(starting_char)

        for idx in range(seq_len):
            i_emb = self.embedding_layer(i)
            z_new, h_new, c_new = self.lstm_cell(i_emb, h, c)
            z_new = F.softmax(z_new/temp, dim=0)
            x = Categorical(z_new)
            i_new = x.sample()
            i_list.append(i_new.item())

            h = h_new
            i = i_new
            c = c_new

        return i_list

def train(model, dataset, lr, out_seq_len, num_epochs):

    # code to initialize optimizer, loss function

    n = 0
    running_loss = 0
    for epoch in range(num_epochs):
        for in_seq, out_seq in dataset.get_example():
            # main loop code

            n += 1

        # print info every X examples
        print(f"Epoch {epoch}. Running loss so far: {(running_loss/n):.8f}")

        print("\n-------------SAMPLE FROM MODEL-------------")

        # code to sample a sequence from your model randomly

        with torch.no_grad():
            pass

        print("\n------------/SAMPLE FROM MODEL/------------")

        n = 0
        running_loss = 0

    
    return None # return model optionally


def run_char_rnn():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10 # one epoch is this # of examples
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    
    train(model, dataset, lr=lr, 
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs)

def run_char_lstm():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10
    out_seq_len = 200
    data_path = "./data/shakespeare.txt"

    # code to initialize dataloader, model
    
    train(model, dataset, lr=lr, 
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs)


def fix_padding(batch_premises, batch_hypotheses):
    batch_premises_reversed = list(batch_premises)
    batch_hypotheses_reversed = list(batch_hypotheses)

    for i in range(len(batch_premises_reversed)):
        batch_premises_reversed[i] = torch.tensor(list(reversed(batch_premises_reversed[i])))
    for i in range(len(batch_hypotheses_reversed)):
        batch_hypotheses_reversed[i] = torch.tensor(list(reversed(batch_hypotheses_reversed[i])))

    for i in range(len(batch_premises)):
        batch_premises[i] = torch.tensor(batch_premises[i])
    for i in range(len(batch_hypotheses)):
        batch_hypotheses[i] = torch.tensor(batch_hypotheses[i])
    
    batch_premises = nn.utils.rnn.pad_sequence(batch_premises, True)
    batch_hypotheses = nn.utils.rnn.pad_sequence(batch_hypotheses, True)
    batch_premises_reversed = nn.utils.rnn.pad_sequence(batch_premises_reversed, True)
    batch_hypotheses_reversed = nn.utils.rnn.pad_sequence(batch_hypotheses_reversed, True)

    return batch_premises, batch_hypotheses, batch_premises_reversed, batch_hypotheses_reversed


def create_embedding_matrix(word_index, emb_dict, emb_dim):
    emb_list = list()
    for str1 in word_index:
        for str2 in emb_dict:
            if str1 == str2:
                emb_list.append(emb_dict[str2])
    return torch.FloatTensor(emb_list)

def evaluate(model, dataloader, index_map):
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        premise_in = tokens_to_ix(list(batch['premise']), index_map)
        hypothesis_in = tokens_to_ix(list(batch['hypothesis']), index_map)
        y_pred = model.forward(premise_in, hypothesis_in)
        y_pred = torch.flatten(y_pred)
        y_pred_label = torch.argmax(y_pred)
        y_true = batch['label']
        if y_pred_label == y_true:
            correct+=1
        total += 1
    return correct/total

class UniLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # your code here (embedding_size = hidden_dim)
        self.embedding_layer = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first = True)
        self.int_layer = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.out_layer = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, a, b):
        a, b = fix_padding(a, b)[:2]
        a = self.embedding_layer(a)
        b = self.embedding_layer(b)
        o_a, (h_a, c_a) = self.lstm(a)
        o_b, (h_b, c_b) = self.lstm(b)
        c = torch.cat((c_a, c_b), dim=2)
        c = self.int_layer(c)
        c = F.relu(c)
        out = self.out_layer(c)
        return torch.squeeze(out)

class ShallowBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(ShallowBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # your code here
        self.embedding_layer = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.lstm_forward = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first = True)
        self.lstm_backward = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first = True)
        self.int_layer = nn.Linear(4*self.hidden_dim, self.hidden_dim)
        self.out_layer = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, a, b):
        a, b, a_rev, b_rev = fix_padding(a, b)

        a = self.embedding_layer(a)
        b = self.embedding_layer(b)
        a_rev = self.embedding_layer(a_rev)
        b_rev = self.embedding_layer(b_rev)

        o_a, (h_a, c_a) = self.lstm_forward(a)
        o_b, (h_b, c_b) = self.lstm_forward(b)
        o_a_rev, (h_a_rev, c_a_rev) = self.lstm_backward(a_rev)
        o_b_rev, (h_b_rev, c_b_rev) = self.lstm_backward(b_rev)

        c = torch.cat((c_a, c_a_rev, c_b, c_b_rev), dim=2)
        c = self.int_layer(c)
        c = F.relu(c)
        out = torch.squeeze(self.out_layer(c))
        return out

def run_snli(model):
    dataset = load_dataset("snli")
    glove = pd.read_csv('./data/glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col=0)

    glove_embeddings = "" # fill in your code

    train_filtered = dataset['train'].filter(lambda ex: ex['label'] != -1)
    valid_filtered = dataset['validation'].filter(lambda ex: ex['label'] != -1)
    test_filtered =  dataset['test'].filter(lambda ex: ex['label'] != -1)

    # code to make dataloaders

    word_counts = build_word_counts(dataloader_train)
    index_map = build_index_map(word_counts)

    # training code

def run_snli_lstm():
    model_class = "" # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)

def run_snli_bilstm():
    model_class = "" # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)

if __name__ == '__main__':
    run_char_rnn()
    # run_char_lstm()
    # run_snli_lstm()
    # run_snli_bilstm()
