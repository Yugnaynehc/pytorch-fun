import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import Model
from data import training_pair, n_letters, n_language, language_from_output


def since(start):
    now = time.time()
    elapse = now-start
    m = elapse//60
    s = elapse - 60*m
    return '%dm %ds' % (m, s)


steps = 1000000
print_every = 5000

n_inp = n_letters
n_hid = 128
n_out = n_language
model = Model(n_inp, n_hid, n_out)
# optimizer = optim.SGD(model.parameters(), lr=5e-3)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

loss_acc = 0
loss_hist = []
start = time.time()
for i in range(1, steps+1):
    language_var, name_var, language, name = training_pair()
    model.zero_grad()

    output = model(name_var)

    loss = F.nll_loss(output, language_var)
    loss_acc += loss.data[0]
    loss.backward()
    optimizer.step()

    if i % print_every == 0:
        pred = language_from_output(output)
        indicator = '✓' if pred == language else '✗ (%s)' % language
        print('{} {:.0f}% ({}) {:.4f} {}/{} {}'.format(
              i, 100*i/steps, since(start), loss.data[0], name, pred, indicator))
        loss_hist.append(loss_acc/print_every)
        loss_acc = 0

torch.save(model, 'char-rnn-classification.pt')
