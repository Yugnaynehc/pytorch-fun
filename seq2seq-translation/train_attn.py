from data import get_training_var, source_lang_dict, target_lang_dict, max_len
from model import Encoder, AttentionDecoder
import torch
from torch.autograd import Variable
import time
import random


def since(start):
    now = time.time()
    elapsed = now-start
    m = elapsed // 60
    s = elapsed - m*60
    return '%d:%d' % (m, s)


def show_loss(step, loss):
    s = '{} {:.0f}% {} {:.4f}'.format(step, 100*step/n_steps, since(start), loss)
    print(s)


n_steps = 75000
print_every = 1000
plot_every = 100

teacher_forcing_ratio = 0.5
max_len = max_len + 1  # 在每句的句末加上了一个<EOS>
hidden_size = 256


if __name__ == '__main__':
    encoder = Encoder(source_lang_dict.n_words, hidden_size)
    decoder = AttentionDecoder(hidden_size, target_lang_dict.n_words, max_len)

    sos_var = Variable(torch.cuda.LongTensor([target_lang_dict.word2idx['SOS']]))
    eos_idx = target_lang_dict.word2idx['EOS']

    e_optimizer = torch.optim.SGD(encoder.parameters(), lr=1e-2)
    d_optimizer = torch.optim.SGD(decoder.parameters(), lr=1e-2)
    loss_fn = torch.nn.functional.nll_loss

    encoder.cuda()
    decoder.cuda()

    print_loss = 0
    plot_loss = 0
    start = time.time()
    for s in range(1, n_steps+1):
        loss = 0
        encoder.zero_grad()
        decoder.zero_grad()
        source_seq, target_seq = get_training_var(cuda=True)
        encoder_outputs = Variable(torch.zeros(max_len, hidden_size)).cuda()
        e_hidden = None
        for i, x in enumerate(source_seq):
            e_output, e_hidden = encoder(x, e_hidden)
            encoder_outputs[i] = e_output[0][0]

        d_input = sos_var
        d_hidden = e_output
        len_target = len(target_seq)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for i in range(len_target):
                d_output, d_hidden, _ = decoder(d_input, d_hidden, encoder_outputs)

                loss += loss_fn(d_output[0], target_seq[i])
                d_input = target_seq[i]
        else:
            for i in range(len_target):
                d_output, d_hidden, _ = decoder(d_input, d_hidden, encoder_outputs)

                loss += loss_fn(d_output[0], target_seq[i])
                topv, topi = d_output.data.topk(1)
                idx = topi[0][0]
                if idx == eos_idx:
                    break
                else:
                    d_input = Variable(torch.cuda.LongTensor([idx]))

        loss.backward()
        e_optimizer.step()
        d_optimizer.step()

        real_loss = loss.data[0] / len_target
        print_loss += real_loss
        plot_loss += real_loss

        if s % print_every == 0:
            torch.save(encoder, 'encoder.pkl')
            torch.save(decoder, 'decoder.pkl')
            show_loss(s, print_loss/print_every)
            print_loss = 0
