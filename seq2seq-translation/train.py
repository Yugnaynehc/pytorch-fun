from data import get_training_var, source_lang_dict, target_lang_dict
from model import Encoder, Decoder
import torch
from torch.autograd import Variable


n_steps = 100000
print_every = 500

max_len = 20
hidden_size = 256
encoder = Encoder(source_lang_dict.n_words, hidden_size)
decoder = Decoder(hidden_size, target_lang_dict.n_words)
e_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-2)
d_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2)
loss_fn = torch.nn.functional.nll_loss

sos_var = Variable(torch.LongTensor([target_lang_dict.word2idx['SOS']]))
for s in range(1, n_steps+1):
    loss = 0
    encoder.zero_grad()
    decoder.zero_grad()
    source_seq, target_seq = get_training_var()
    e_hidden = None
    for i, x in enumerate(source_seq):
        e_output, e_hidden = encoder(x, e_hidden)

    d_input = sos_var
    d_hidden = e_output
    len_target = len(target_seq)
    for i in range(len_target):
        d_output, d_hidden = decoder(d_input, d_hidden)

        loss += loss_fn(d_output[0], target_seq[i])
        d_input = target_seq[i]

    loss.backward()
    e_optimizer.step()
    d_optimizer.step()

    if s % print_every == 0:
        torch.save(encoder, 'encoder.pkl')
        torch.save(decoder, 'decoder.pkl')
        print(loss.data[0]/len_target)
