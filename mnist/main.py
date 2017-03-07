import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets


class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

torch.manual_seed(1)
torch.cuda.manual_seed(1)

data_root = '../data'
transform_fn = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])
kwargs = {'num_workers': 1, 'pin_memory': True}
bs = 64

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_root, train=True,
                   transform=transform_fn, download=True),
    batch_size=bs, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_root, train=False,
                   transform=transform_fn, download=True),
    batch_size=bs, shuffle=True, **kwargs)


net = LeNet()
net.cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.5)


def train(epoch):
    net.train()
    for i, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = net(data)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('Train Epoch {}: [{}/{} ({:.2%})]\t Loss:{:.6f} '.format(
                epoch, i*len(data), len(train_loader.dataset),
                i*len(data)/len(train_loader.dataset),
                loss.data[0]))


def test(epoch):
    net.eval()
    loss = 0
    correct = 0
    for data, target in test_loader:
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda())
        output = net(data)
        current_loss = F.nll_loss(output, target).data[0]
        loss += current_loss
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    print('Test Epoch:{} \tAccuracy:{}/{} ({:.2f}%) \tAverage Loss:{:.6f}'.format(
        epoch, correct, len(test_loader.dataset),
        100. * correct/len(test_loader.dataset),
        loss/len(test_loader.dataset)))


epoch = 10

for t in range(epoch):
    train(t)
    test(t)
