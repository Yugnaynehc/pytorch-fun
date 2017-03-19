import random
import torch
from torch.autograd import Variable


class DynamicNet(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()

        self.in_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.out_linear = torch.nn.Linear(H, D_out)

    def forward(self, input):
        x = self.in_linear(input).clamp(min=0)
        for _ in range(random.randint(1, 4)):
            x = self.middle_linear(x).clamp(min=0)
        x = self.out_linear(x)
        return x


N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

net = DynamicNet(D_in, H, D_out)

lr = 1e-4
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

loss_fn = torch.nn.MSELoss(size_average=False)

for t in range(500):
    y_pred = net(x)
    loss = loss_fn(y_pred, y)

    print(t, loss.data[0])

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
