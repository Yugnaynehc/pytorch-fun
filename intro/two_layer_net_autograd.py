import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

lr = 1e-6
for i in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = (y_pred-y).pow(2).sum()
    print(i, loss.data[0])

    w1.grad.data.zero_()
    w2.grad.data.zero_()

    loss.backward()

    w1.data -= lr*w1.grad.data
    w2.data -= lr*w2.grad.data

    
