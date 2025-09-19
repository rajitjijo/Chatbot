import torch

def maskNLLLoss(decoder_out, target, mask, device):
    """
    Custom Negetive Log Liklehood Loss
    We can also use the normal CrossEntropyLoss(ingore_index=PAD)
    """
    ntotal = mask.sum()
    target = target.view(-1,1)
    #Decoder output shape: (batch_size, vocab_size), targets_size = (batch_size,1)
    gathered_tensor = torch.gather(decoder_out,1,target)
    cross_entropy = -torch.log(gathered_tensor)
    loss = cross_entropy.masked_select(mask)
    loss = loss.mean()
    loss = loss.to(device)
    return loss, ntotal.item()