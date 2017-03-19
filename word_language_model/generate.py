import argparse
import torch
from torch.autograd import Variable
from data import Corpus


parser = argparse.ArgumentParser(description='PTB语言生成模型')

parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--checkpoint', type=str, default='./model.pt')
parser.add_argument('--outf', type=str, default='./generated.txt')
parser.add_argument('--seed', type=int, default=11111)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--words', type=int, default=1000)

args = parser.parse_args()

# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     if args.cuda:
#         torch.cuda.manual_seed(args.seed)

if args.cuda:
    model = torch.load('model.pt')
else:
    model = torch.load('model.pt', lambda storage, location: storage)

corpus = Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input = input.cuda()

with open(args.outf, 'w') as f:
    for i in range(args.words):
        output, hidden = model(input, hidden)
        word_weights = output.data[0].div(args.temperature).exp().cpu()
        idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(idx)
        word = corpus.dictionary.idx2word[idx]
        if word == '<EOS':
            f.write('.\n')
        else:
            f.write(word + ' ')
