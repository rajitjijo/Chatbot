import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers = 1, drouput=0):
        super().__init__()
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

class Attention(nn.Module):
    """
    method: normal dot product, dense(encoder_hidden) * decoder hidden, dense(concated encoder and decoder)
    """
    def __init__(self, method, hidden_size):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)
    
    def forward(self, hidden, encoder_output):
        """
        hidden: (1, batch_size, hidden_size)
        encoder_output: (max_sent_length, batch_size, hidden_size) 
        """
        attn_energies = self.dot_score(hidden, encoder_output) #shape (max_length, batch_size)
        attn_energies = attn_energies.t() #shape (batch_size, max_length)

        return nn.functional.softmax(attn_energies, dim=1).unsqueeze(1) #shape(batch_size, 1, max_length)
    
class Decoder(nn.Module):

    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layer=1, dropout=0.1):
        super().__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layer
        self.dropout = dropout
        #Layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=self.n_layers, dropout=(0 if n_layer==1 else dropout))
        self.concat = nn.Linear(in_features=hidden_size*2,out_features=hidden_size)
        self.attention = Attention(method=self.attn_model, hidden_size=hidden_size)
        self.out = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        input_step: 1 timestep or word of inputseq batch, shape = (1, batch_size)
        last_hidden: hidden layer of GRU, shape: (n_layers*n_directions, batch_size, hidden_size)
        encoder_outputs: encoder models output, shape: (max_sentance_length, batch_size, hidden_size)
        """
        # Step1: Encode current batch of words
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Step2: Forward through unidirectional GRU
        gru_ouput, hidden = self.gru(embedded, last_hidden) #output: (1, batch_size, hidden_size*n_directions), hidden: (n_layers*n_directions, batch_size, hidden_size)
        # Step3: Calculate attentions outputs
        attn_weights = self.attention(gru_ouput, encoder_outputs)
        # Step4: Multiply attention weights with encoder output to get new weighted sum context vector
        # (batch_size, 1, max_seq_len) bmm (batch_size, max_seq_len, hidden_size) -> (batch_size,1,hidden)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))
        # Step5: Concatenate weighted context vector and gru output and get the more context rich representation for output
        gru_ouput = gru_ouput.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((context,gru_ouput),dim=1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Step6: Dense layer to map hiddenSize -> max_vocab_size
        output = self.out(concat_output)
        # Step7: Convert it into a probability distribution to argmax the next word, which will give us its index in the corpus
        output = nn.functional.softmax(output,dim=1)
        return output, hidden
    



if __name__ == "__main__":
    pass