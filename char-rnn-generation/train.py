import torch
import torch.nn.functional as F
from model import Model
from data import generate_training_data, n_language, n_letters
import time


def since(start):
    now = time.time()
    elapsed = now-start
    m = elapsed // 60
    s = elapsed - m*60
    return '%dm %ds' % (m, s)


steps = 100000
print_every = 5000
save_every = 20000

n_hid = 128
n_inp = n_out = n_letters
net = Model(n_language, n_inp, n_hid, n_out)
optimizer = torch.optim.SGD(net.parameters(), lr=5e-4)

start = time.time()
for s in range(1, steps+1):
    lang_var, input_var, target_var = generate_training_data()
    hidden = net.init_hidden()
    net.zero_grad()

    loss = 0
    l = input_var.size()[0]
    for i in range(l):
        output, hidden = net(lang_var, input_var[i], hidden)
        loss += F.nll_loss(output, target_var[i])

    loss.backward()
    optimizer.step()

    if s % print_every == 0:
        info = '{:d} {:.0f}% {} {:.4f}'.format(s, 100*s/steps, since(start),
                                               loss.data[0]/l)
        print(info)


    if s % save_every == 0:
        torch.save(net, 'generator.pt')
