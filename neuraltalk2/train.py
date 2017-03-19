import os
from six.moves import cPickle
import time
import misc.utils as utils
import opts
import models
from dataloader import Dataloader
import torch
from torch.autograd import Variable

opt = opts.parse_opt()
loader = Dataloader(opt)

opt.vocab_size = loader.vocab_size
opt.max_seq_length = loader.max_seq_length

infos = {}

if opt.start_from is not None:
    # 载入保存好的info数据，并且检查是否和模型兼容
    with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
        infos = cPickle.load(f)
        saved_model_opt = infos['opt']
        need_be_same = ['caption_model', 'rnn_type', 'num_layers']
        for checkme in need_be_same:
            assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], \
                "Command line argument and saved model disagree on '%s' " % checkme


iteration = infos.get('iter', 0)
epoch = infos.get('epoch', 0)
val_result_history = infos.get('val_result_history', {})
loss_history = infos.get('loss_history', {})
lr_history = infos.get('lr_history', {})
ss_prob_history = infos.get('ss_prob_history', {})


loader.iterators = infos.get('iterators', loader.iterators)
if opt.load_best_score == 1:
    best_val_score = infos.get('best_val_score', None)

model = models.setup(opt)
model.cuda()

update_lr_flag = True
model.train()

criterion = utils.LanguageModelCriterion()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

if vars(opt).get('start_from', None):
    optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

start = time.time()

current_lr = opt.learning_rate
while True:
    if update_lr_flag:
        if opt.learning_rate_decay_start >= 0 and epoch > opt.learning_rate_decay_start:
            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            decay_factor = opt.learning_rate_decay_rate ** frac
            current_lr = opt.learning_rate * decay_factor
            utils.set_lr(optimizer, current_lr)
        if opt.scheduled_sampling_start >= 0 and epoch > opt.scheduled_sampling_start:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            model.ss_prob = min(opt.scheduled_sampling_increase_prob ** frac,
                                opt.scheduled_sampling_max_prob)
        update_lr_flag = False

    # 加载训练数据
    data = loader.get_batch('train')

    torch.cuda.synchronize()

    buf = [data['fc_feats'], data['att_feats'], data['captions'], data['masks']]
    buf = [Variable(torch.from_numpy(d), requires_grad=False).cuda() for d in buf]
    fc_feats, att_feats, captions, masks = buf

    optimizer.zero_grad()
    output = model(fc_feats, att_feats, captions)
    loss = criterion(output, captions[:, 1:], masks[:, 1:])
    loss.backward()
    utils.clip_gradient(optimizer, opt.grad_clip)
    optimizer.step()

    train_loss = loss.data[0]
    torch.cuda.synchronize()

    if iteration % opt.print_every == 0:
        print('elapsed {} iter {} (epoch {}), loss={:.3f}'.format(
            utils.since(start), iteration, epoch, train_loss))

    # 更新iteration和epoch信息
    iteration += 1
    if data['bounds']['wrapped']:
        epoch += 1

    # 记录训练历史信息
    if (iteration % opt.losses_log_every == 0):
        loss_history[iteration] = train_loss
        lr_history[iteration] = current_lr
        ss_prob_history[iteration] = model.ss_prob

    # 在验证集上测试一下模型性能，并且保存模型
    if (iteration % opt.save_checkpoint_every == 0):
        # # 评估模型
        # eval_kwargs = {'split': 'val',
        #                'dataset': opt.input_json}
        # eval_kwargs.update(vars(opt))
        # val_loss, predictions, lang_stats = eval_utils.eval_split(model, criterion, loader, eval_kwargs)
        # val_result_history[iteration] = {'loss': val_loss, 'predictions': predictions, 'lang_stats': lang_stats}

        # # 如果在验证集上的性能有所提升，那么就保存一下模型
        # if opt.language_eval == 1:
        #     current_score = lang_stats['CIDEr']
        # else:
        #     current_score = - val_loss

        best_flag = False
        if True:
            # if best_val_score is None or current_score > best_val_score:
            #     best_val_score = current_score
            #     best_flag = False

            checkpoint_file = os.path.join(opt.checkpoint_path, 'model.pth')
            torch.save(model.state_dict(), checkpoint_file)
            print('model saved to %s' % checkpoint_file)
            optimizer_file = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_file)

            # 转储一些额外的信息
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['val_result_history'] = val_result_history
            infos['loss_history'] = loss_history
            infos['lr_history'] = lr_history
            infos['ss_prob_history'] = ss_prob_history
            infos['vocab'] = loader.get_vocab()
            infos_filename = 'infos_' + opt.id + '.pkl'
            with open(os.path.join(opt.checkpoint_path, infos_filename), 'wb') as f:
                cPickle.dump(infos, f)

            if best_flag:
                print('It is a BEST model!')
                best_checkpoint_file = os.path.join(opt.checkpoint_path, 'model-best.pth')
                torch.save(model.state_dict(), best_checkpoint_file)
                print('model saved to %s' % best_checkpoint_file)
                best_infos_filename = 'best-infos_' + opt.id + '.pkl'
                with open(os.path.join(opt.checkpoint_path, best_infos_filename), 'wb') as f:
                    cPickle.dump(infos, f)

    if opt.max_epochs != -1 and epoch >= opt.max_epochs:
        break
