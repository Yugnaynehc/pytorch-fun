import os
import torch
from misc.show_tell import ShowTellModel


def setup(opt):
    if opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    else:
        raise Exception('Caption model not supported: %s' % opt.caption_model)

    if vars(opt).get('start_from', None):
        assert os.path.isdir(opt.start_from), '%s must be a path' % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')), \
            'infos.pkl file does not exist in path %s' % opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model
