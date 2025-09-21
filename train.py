from preprocessing import get_pairs, batch2traindata
from Vocabulary import vocabulary
import torch
from NLLLoss import maskNLLLoss
from tqdm import tqdm
import random, os, csv
import pickle
from dataset import ChatDataset
from model import *

def train(input_array, lengths, target_array, target_mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer:torch.optim.Optimizer, decoder_optimizer:torch.optim.Optimizer, batch_size,
          clip, max_length, teacher_forcing_ratio, device):
    
    SOS = 1 #start of sentance token
    
    #zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #set device options
    input_array = input_array.to(device)
    target_array = target_array.to(device)
    target_mask = target_mask.to(device)
    lengths = lengths.to("cpu")

    #initialize variables for loss calculation
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    iter_loss = []
    total_items = 0

    #Forward pass through encoder
    encoder_output, encoder_hidden = encoder(input_array, lengths)

    #For dynamic batch size_modification
    batch_size = encoder_hidden.shape[1]

    #Creating initial decoder input ([SOS token for all inputs in our current batch])
    decoder_input = torch.LongTensor([[SOS for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    #Setting decoders first hidden state to encoders last hiddens state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    #determine if we are using teacher forcing or not
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len): 
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            #Teacher Forcing next decoder input is current target
            decoder_input = target_array[t].view(1,-1)
            #Calculate and accumulate loss
            mask_loss, ntotal = maskNLLLoss(decoder_output, target_array[t], target_mask[t].bool(), device)
            loss = loss + mask_loss
            iter_loss.append(mask_loss.item()*ntotal)
            total_items += ntotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
            #No Teacher forcing, decoders next input is its current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            #Calculate and accumulate loss
            mask_loss, ntotal = maskNLLLoss(decoder_output, target_array[t], target_mask[t].bool(), device)
            loss = loss + mask_loss
            iter_loss.append(mask_loss.item()*ntotal)
            total_items += ntotal

    #Perform Backprop and compute gradients
    loss.backward()

    #Clip Gradients
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    #Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(iter_loss) / total_items


def trainIters(model_name:str, vocab:vocabulary, dataloader:torch.utils.data.DataLoader, encoder, decoder,
               encoder_optimizer, decoder_optimizer, embedding, save_dir, n_epochs, batch_size,
               device, clip, teacher_forcing_ratio, max_seq_length):
    
    print(" Starting Training....")
    
    num_batches = len(dataloader)
    directory = os.path.join(save_dir, model_name)
    os.makedirs(directory, exist_ok=True)

    with open(os.path.join(directory, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    csv_path = os.path.join(directory, f"{model_name}_train_loss.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Log_Loss"])
    

    for epoch in range(1, n_epochs+1):

        epoch_loss = 0
        if epoch == n_epochs // 2:
            teacher_forcing_ratio = 0.5

        for index, batch in enumerate(dataloader):

            input_array, lengths, target_array, target_mask, max_target_len = batch
            batch_loss = train(input_array, lengths, target_array, target_mask, max_target_len, encoder, decoder, embedding,
                               encoder_optimizer, decoder_optimizer, batch_size, clip, max_seq_length, teacher_forcing_ratio, device)
            
            epoch_loss = epoch_loss + batch_loss
            
            print(f"[{epoch}/{n_epochs}]:: Current Batch: ({index+1}/{num_batches}), Loss: {batch_loss}")

        epoch_loss = epoch_loss / num_batches

        train_dir = os.path.join(directory, f"train_{epoch}")
        os.makedirs(train_dir, exist_ok=True)
        
        torch.save(encoder.state_dict(), os.path.join(train_dir, "encoder.pth"))
        torch.save(decoder.state_dict(), os.path.join(train_dir, "decoder.pth"))
        torch.save(embedding.state_dict(), os.path.join(train_dir, "embedding.pth"))

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_loss])


    print("Finished Training")

if __name__ == "__main__":

    print("Loading Dataset")
    datafile = "data/formatted_movie_lines.txt"
    #set up vocab object
    vocab = vocabulary("Cornell_Movie_Diaglogues")
    #build dataset
    pairs, vocab = get_pairs(datafile, vocab)
    chatdata = ChatDataset(pairs)
    print("Dataset Loaded")

    #Configure Training Run
    model_name = "chatbot_2"
    batch_size = 64
    attn_model = "dot" #couldve been "concat" or general
    hidden_size = 512
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    n_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_length = 11
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0

    print(f"""
            Configured Training Run:
            Model Name:            {model_name}
            Batch Size:            {batch_size}
            Attention Model:       {attn_model}
            Hidden Size:           {hidden_size}
            Encoder Layers:        {encoder_n_layers}
            Decoder Layers:        {decoder_n_layers}
            Dropout:               {dropout}
            Epochs:                {n_epochs}
            Device:                {device}
            Max Seq Length:        {max_seq_length}
            Gradient Clip:         {clip}
            Teacher Forcing Ratio: {teacher_forcing_ratio}
            Learning Rate:         {learning_rate}
            Decoder LR Ratio:      {decoder_learning_ratio}
            """)

    #make the dataloader object
    dataloader = torch.utils.data.DataLoader(chatdata, batch_size=batch_size, shuffle=True, collate_fn=lambda b:batch2traindata(vocab,b))
    
    #Initialize Embedding layer
    embedding = nn.Embedding(vocab.num_words, hidden_size)
    #Initializing Encoder and Decoder
    encoder = Encoder(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = Decoder(attn_model, embedding, hidden_size, vocab.num_words, decoder_n_layers, dropout)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    #Ensuring theyre in training mode
    encoder.train()
    decoder.train()

    #Initialize Optimizers
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), learning_rate*decoder_learning_ratio)

    #Running Main Training Loop
    trainIters(model_name=model_name,vocab=vocab,dataloader=dataloader,encoder=encoder,
               decoder=decoder,encoder_optimizer=encoder_optimizer,decoder_optimizer=decoder_optimizer,
               embedding=embedding,save_dir="training_runs",n_epochs=n_epochs,batch_size=batch_size,device=device,clip=clip,
               teacher_forcing_ratio=teacher_forcing_ratio,max_seq_length=max_seq_length)
    






    





