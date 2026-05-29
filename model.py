import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, pad_idx, cell_type):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.cell_type = cell_type
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, hidden = self.rnn(packed_embedded)
        if self.cell_type == 'lstm':
            hidden = hidden[0]
        last_hidden = hidden[-1, :, :]
        out = self.fc(self.dropout(last_hidden))
        return out