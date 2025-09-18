import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers = 1, drouput=0):

        self.n_layers = n_layers
        self.embedding = embedding
        self.hidden_size = hidden_size

        #The input size of the GRU will be set to hidden_size which will also be the size of the num of features in the word embedding
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.n_layers, dropout=(0 if n_layers==1 else drouput), bidirectional=True)


    def forward(self, input_seq, input_lengths, hidden=None):
        """
        input_seq: batch of input sentances with shape: (seq_length, batch_size)
        input_lengths: list of sentance lengths corresponding to each sentance in the batch, so that we can avoid the padded values
        hidden_state: (n_layers x n_directions, batch_size, hidden_size)\n
        ------X------\n
        outputs: Output features from the last layer of the GRU, for each timestep (sum of biderectional output); shape = (seq_length, batch_size, hidden_size)
        hidden: Hidden state for the last time step of shape = (n_layers x no_of_directions, batch_size, hidden_size)
        """

        # Conver word to word_embedding
        x = self.embedding(input_seq)
        # Pack padded batch of sequence for RNN Moduke
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths)
        outputs, hidden = self.gru(x, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum Biderectional GRU outputs
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]

        return outputs, hidden


    