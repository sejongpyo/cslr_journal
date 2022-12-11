from torch import nn

class LSTM(nn.Module):
    def __init__(self, cnn_out, lstm_dim, lstm_layer, dropout):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(cnn_out, lstm_dim, lstm_layer, bidirectional=True,
                            dropout=dropout)
        
    def forward(self, x, prob_sizes):
        if len(prob_sizes) != 1:
            packed_emb = nn.utils.rnn.pack_padded_sequence(x, prob_sizes)
            packed_outputs, h = self.lstm(packed_emb)

            rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        else:
            rnn_outputs, h = self.lstm(x)
        
        return rnn_outputs.transpose(0, 1)