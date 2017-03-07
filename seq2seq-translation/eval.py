import torch
from torch.autograd import Variable
import sys
from data import pairs, source_lang_dict, target_lang_dict
from data import sentence_to_idx_var, max_len
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


hidden_size = 256
encoder = torch.load('encoder.pt', map_location=lambda storage, loc: storage)
decoder = torch.load('decoder.pt', map_location=lambda storage, loc: storage)
sos_idx = target_lang_dict.word2idx['SOS']
sos_var = Variable(torch.LongTensor([sos_idx]))
eos_idx = target_lang_dict.word2idx['EOS']
max_len = max_len + 1  # 在每句的句末加上了一个<EOS>


encoder.cpu()
decoder.cpu()


def inference(s):
    source_seq = sentence_to_idx_var(s, source_lang_dict)
    target_seq = []
    encoder_outputs = Variable(torch.zeros(max_len, hidden_size))
    e_hidden = None
    for i, x in enumerate(source_seq):
        e_output, e_hidden = encoder(x, e_hidden)
        encoder_outputs[i] = e_output[0][0]

    d_input = sos_var
    d_hidden = e_output
    decoder_attns = torch.zeros(max_len, max_len)
    for i in range(max_len):
        d_output, d_hidden, attn = decoder(d_input, d_hidden, encoder_outputs)
        topv, topi = d_output.data.topk(1)
        idx = topi[0][0]
        if idx == eos_idx:
            target_seq.append('<EOS>')
            return target_seq, decoder_attns
        d_input = Variable(torch.LongTensor([idx]))
        target_seq.append(target_lang_dict.idx2word[idx])
        decoder_attns[i] = attn.data
    return target_seq, decoder_attns


def show_attention(source, target, attn):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(attn.numpy(), cmap='bone')
    fig.colorbar(cax)

    ax.set_xticklabels(['']+source+['<EOS>'], rotation=90)
    ax.set_yticklabels(['']+target)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def eval_randomly():
    for _ in range(100):
        p = random.choice(pairs)
        print('>', ' '.join(p[0]))
        print('=', ' '.join(p[1]))
        target_seq, decoder_attns = inference(p[0])
        print('<', ' '.join(target_seq))
        print('')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        source_seq = sys.argv[1:]
        target_seq, decoder_attns = inference(source_seq)
        print(' '.join(target_seq))
        show_attention(source_seq, target_seq, decoder_attns)
    else:
        eval_randomly()
