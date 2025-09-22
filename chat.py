from model import *
from Vocabulary import vocabulary
import pickle
from preprocessing import normalizeString

class GreedyDecoder(nn.Module):

    SOS = 1 #start of sentance token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

    def forward(self, input_seq, input_length, max_length):
        #forward input through encoder model
        encoder_output, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        #initializing decoder input
        decoder_input = torch.ones(1,1, device=self.device, dtype=torch.long) * self.SOS
        #Inititalize tensors to append decoded words to
        all_tokens = torch.zeros(0, device=self.device, dtype=torch.long)
        all_scores = torch.zeros(0, device=self.device)

        #iteratively decode one word at a time
        for _ in range(max_length):
            #forward pass through deocder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
            #obtain most likely word and its softmax score
            decoder_score, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores,decoder_score), dim=0)
            #Preparing the current token to be the next decoder input by adding a dimension in the row axis
            decoder_input = torch.unsqueeze(decoder_input,0)
        
        return all_tokens, all_scores
    
def evaluate(encoder:Encoder, decoder: Decoder, searcher:GreedyDecoder, voc:vocabulary, sentance:str, device, max_length=11):

    #format input sentance as a batch
    indexes_batch = [voc.indexfromSentance(sentance)]
    #create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # make (batch, seq_len) -> (seq_len, batch)
    input_batch = torch.LongTensor(indexes_batch).transpose(0,1)
    lengths = lengths.to("cpu")
    input_batch = input_batch.to(device)
    #Decode sentances with searher
    tokens, scores = searcher(input_batch, lengths, max_length)
    #indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words, scores

def chat(encoder:Encoder, decoder:Decoder, searcher:GreedyDecoder, vocab:vocabulary, device:torch.device):

    input_sentance = ""
    while True:
        # try:
        # Get input Sentance
        input_sentance = input("> ")
        if input_sentance == "q" or input_sentance == "exit" or input_sentance == "quit":break
        #normalize sentance
        input_sentance = normalizeString(input_sentance)
        #evaluate sentance
        output_words, scores = evaluate(encoder, decoder, searcher, vocab, input_sentance, device)
            # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        print('Bot:', ' '.join(output_words))
    
        # except Exception as e:
        #     print(e)
        #     print("Error: Unknown word encountered")

    print("Quitting")


if __name__ == "__main__":

    #loading app vocabulary
    vocab_path = "training_runs/chatbot_2/vocab.pkl"
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #initializing models
    embedding = nn.Embedding(vocab.num_words, 512)
    encoder = Encoder(512,embedding,2,0.1)
    decoder = Decoder("dot", embedding, 512, vocab.num_words, 2, 0.1)
    #loading appropriate weights
    missing, unexpected = encoder.load_state_dict(torch.load("training_runs/chatbot_2/train_50/encoder.pth"))
    decoder.load_state_dict(torch.load("training_runs/chatbot_2/train_50/decoder.pth"))
    # embedding.load_state_dict(torch.load("training_runs/chatbot_2/train_50/embedding.pth"))
    
    encoder.eval()
    decoder.eval()

    searcher = GreedyDecoder(encoder, decoder)

    chat(encoder, decoder, searcher, vocab, device)

    # for name, param in decoder.named_parameters():
    #     print(name, param.shape)
    