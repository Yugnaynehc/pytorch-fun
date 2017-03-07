import torch
from torch.autograd import Variable

class TwoLayerNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        x = self.linear1(input)
        x = self.relu(x)
        out = self.linear2(x)
        return out


N, D_in, H, D_out = 64, 1000, 100, 10

use_half = False

if use_half:
    dtype = torch.cuda.HalfTensor
else:
    dtype = torch.FloatTensor
x = Variable(torch.randn(N, D_in).type(dtype))
y = Variable(torch.randn(N, D_out).type(dtype))

net = TwoLayerNet(D_in, H, D_out)
if use_half:
    net.cuda().half()

lr = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr)

loss_fn = torch.nn.MSELoss(size_average=False)

for t in range(500):
    y_pred = net(x)

    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
