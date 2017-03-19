import time
import torch
import torch.nn as nn
from torch.autograd import Variable


def repackage(v):
    if type(v) is Variable:
        return Variable(v.data)
    else:
        return tuple(repackage(var) for var in v)


def decode_sequence(idx2word, seq):
    N, D = seq.size()
    output = []
    for i in range(N):
        s = ''
        for j in range(D):
            idx = seq[i][j]
            if idx > 0:
                if j >= 1:
                    s += ' '  # 补一个空格
                s += idx2word[idx]
            else:
                break  # id是0表示空字符，因此语句结束
        output.append(s)
    return output


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = input.contiguous().view(-1, input.size(2))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def since(start):
    now = time.time()
    elapsed = now - start
    m = elapsed // 60
    s = elapsed - m * 60
    return '%d:%d' % (m, s)
