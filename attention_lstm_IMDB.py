from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
embedding_dim = 200
hidden_dim = 200
epochs = 5

# define Field
TEXT = data.ReversibleField(lower=True, include_lengths=True)
LABEL = data.Field(sequential=False)
# make splits for data
train, test = datasets.IMDB.splits(TEXT, LABEL)
# build the vocabulary
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=embedding_dim))
LABEL.build_vocab(train)
train_iter, test_iter = data.BucketIterator.splits((train, test),
                                                   sort_key=lambda x:len(x.text),
                                                   sort_within_batch=True,
                                                   batch_size=batch_size,
                                                   device=device,
                                                   repeat=False)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class AttnClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def set_embedding(self, vectors):
        self.embedding.weight.data.copy_(vectors)

    def forward(self, inputs, lengths):
        batch_size = inputs.size(1)
        # (L, B)
        embedded = self.embedding(inputs)
        # (L, B, E)
        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        out, hidden = self.lstm(packed_emb)
        out = nn.utils.rnn.pad_packed_sequence(out)[0]
        out = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        # (L, B, H)
        embedding, attn_weights = self.attention(out.transpose(0, 1))
        # (B, HOP, H)
        outputs = self.fc(embedding.view(batch_size, -1))
        # (B, 1)
        return outputs, attn_weights


model = AttnClassifier(len(TEXT.vocab), embedding_dim, hidden_dim).to(device)
model.set_embedding(TEXT.vocab.vectors)
# optim
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss().to(device)
# train model
for epoch in range(epochs):
    train(train_iter, model, optimizer, criterion)

def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(F.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def accuracy(model, test_iter):
    model.eval()
    total_acc = 0
    for batch in test_iter:
        (x, x_l), y = batch.text, batch.label - 1
        outputs,_ = model(x, x_l)
        total_acc += binary_accuracy(outputs.view(-1), y.float()).item()
    return total_acc / len(test_iter)

print(accuracy(model, test_iter))


