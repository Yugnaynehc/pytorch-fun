import time
import math
from data import Corpus
from model import RNNModel
import torch
import torch.nn as nn
from torch.autograd import Variable


def batchify(data, bsz):
    '''
    把数据修整一下，变成刚好的批量
    '''
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch*bsz)
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data


def get_batch(dataset, i, evaluation=False):
    seq_len = min(bptt_len, len(dataset)-1-i)
    source = Variable(dataset[i:i+seq_len], volatile=evaluation)
    target = Variable(dataset[i+1:i+1+seq_len].view(-1))
    return source, target


def repackage_hidden(h):
    if type(h) is Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def clip_gradient(model, clip):
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))


def train():
    total_loss = 0
    start = time.time()
    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0)-1, bptt_len)):
        source, target = get_batch(train_data, i)
        model.zero_grad()
        output, hidden = model(source, hidden)
        hidden = repackage_hidden(hidden)
        loss = loss_fn(output, target)
        loss.backward()

        clip_lr = lr * clip_gradient(model, clip_coefficient)
        for p in model.parameters():
            p.data.sub_(clip_lr, p.grad.data)

        total_loss += loss.data[0]

        if batch % print_every == 0 and batch > 0:
            print_loss = total_loss / print_every
            total_loss = 0
            elapsed = time.time() - start
            start = time.time()
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                  'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data)//bptt_len, lr,
                      elapsed*1000/print_every, print_loss,
                      math.exp(print_loss)))


def eval(dataset):
    total_loss = 0
    hidden = model.init_hidden(eval_batch_size)
    for batch, i in enumerate(range(0, dataset.size(0)-1, bptt_len)):
        source, target = get_batch(dataset, i, evaluation=True)
        output, hidden = model(source, hidden)
        hidden = repackage_hidden(hidden)
        loss = loss_fn(output, target)
        total_loss += loss.data[0]
    return total_loss/(batch+1)


# 参数设定
nepoch = 6
batch_size = 20
eval_batch_size = 10
bptt_len = 20
emsize = 200
nhid = 256
nlayers = 2
rnn_type = 'LSTM'
lr = 20
clip_coefficient = 0.5
cuda = True
print_every = 200

# 准备数据
corpus = Corpus('./data')
ntokens = len(corpus.dictionary)
train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# 建立模型
model = RNNModel(rnn_type, ntokens, emsize, nhid, nlayers)
if cuda:
    model.cuda()
loss_fn = nn.functional.cross_entropy

prev_val_loss = None
for epoch in range(1, nepoch+1):
    epoch_start_time = time.time()
    train()
    val_loss = eval(val_data)
    print('-'*80)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time()-epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-'*80)

    if prev_val_loss and val_loss > prev_val_loss:
        lr /= 4
    prev_val_loss = val_loss


test_loss = eval(test_data)
print('='*80)
print('| end of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('='*80)

with open('model.pt', 'wb') as f:
    torch.save(model, f)
