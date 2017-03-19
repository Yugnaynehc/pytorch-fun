import random
import json
import h5py
import opts
import atexit
from multiprocessing.dummy import Pool
import numpy as np


class Dataloader():

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.seq_per_img = opt.seq_per_img

        # 加载包含元数据的json文件
        print('Dataloader loading json file: ', opt.input_json)
        self.info = json.load(open(opt.input_json))
        self.idx2word = self.info['idx2word']
        self.vocab_size = len(self.idx2word)
        print('vocab size is: ', self.vocab_size)

        # 加载h5文件
        print('Dataloader loading h5 file: ', opt.input_fc_h5, opt.input_att_h5, opt.input_label_h5)
        self.h5_fc_file = h5py.File(opt.input_fc_h5, 'r')
        self.h5_att_file = h5py.File(opt.input_att_h5, 'r')
        self.h5_label_file = h5py.File(opt.input_label_h5, 'r', driver='core')

        # 获取图片数据库大小信息
        fc_size = self.h5_fc_file['fc'].shape
        att_size = self.h5_att_file['att'].shape
        assert fc_size[0] == att_size[0], 'fc and att are not the same number'
        self.num_images = fc_size[0]
        print('read %d image features' % self.num_images)

        # 载入文本序列数据
        seq_size = self.h5_label_file['all_captions'].shape
        self.max_seq_length = seq_size[1]
        print('max sequence length in data is ', self.max_seq_length)

        # 加载每副图片对应的caption的索引信息
        self.sample_start_idx = self.h5_label_file['sample_start_idx'][:]
        # 之所以要加[:]是因为这才会把数据实例化，不然只是一个数据的引用
        self.sample_end_idx = self.h5_label_file['sample_end_idx'][:]

        # 把图片按照所属的集合进行划分（训练集、测试集、验证集）
        self.split_idx = {'train': [], 'val': [], 'test': []}
        for idx in range(len(self.info['images'])):
            img_split = self.info['images'][idx]['split']
            if img_split != 'restval':
                self.split_idx[img_split].append(idx)
            elif opt.train_only == 0:
                self.split_idx['train'].append(idx)

        print('assigned %d images to split train' % len(self.split_idx['train']))
        print('assigned %d images to split val' % len(self.split_idx['val']))
        print('assigned %d images to split test' % len(self.split_idx['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self)

        @atexit.register
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                self._prefetch_process[split].terminate()
                self._prefetch_process[split].join()
        # atexit.register(cleanup)

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        bsz = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        # 为了利用GPU的并行计算，我们把一张图像的CNN特征复制多遍，
        # 使得一张图的多个caption能够同时训练
        # （如果让多个caption用同一份CNN特征，就没办法做并行计算了吗？）
        fc_batch = np.ndarray((bsz * seq_per_img,) + self.h5_fc_file['fc'].shape[1:], dtype=np.float32)
        # 注意这里的加法是tuple的加法，会将两个tuple合并
        att_batch = np.ndarray((bsz * seq_per_img,) + self.h5_att_file['att'].shape[1:], dtype=np.float32)

        # 为什么要+2呢?难道是加上<SOS>和<EOS>?
        caption_batch = np.zeros([bsz * seq_per_img, self.max_seq_length + 2], dtype=np.int)
        mask_batch = np.zeros([bsz * seq_per_img, self.max_seq_length + 2], dtype=np.float32)

        wrapped = False

        infos = []
        gts = []

        for i in range(bsz):
            fc_batch[i * seq_per_img: (i + 1) * seq_per_img], \
                att_batch[i * seq_per_img: (i + 1) * seq_per_img], \
                idx, tmp_wrapped = self._prefetch_process[split].get()

            # 获取caption数据
            start_idx = self.sample_start_idx[idx]  # 原程序是从1计数，这里我从0计数
            end_idx = self.sample_end_idx[idx]
            ncap = end_idx - start_idx
            assert ncap > 0, 'an image does not have any caption.'

            if ncap < seq_per_img:
                # caption的数量不足，需要进行采样来扩充数据，这里进行有放回的采样
                seq = np.zeros([seq_per_img, self.max_seq_length], dtype=np.int)
                for q in range(seq_per_img):
                    j = random.randint(start_idx, end_idx)
                    seq[q, :] = self.h5_label_file['all_captions'][j, :self.max_seq_length]
            else:
                # 如果数量足够或者超过所需,那就只取一部分
                j = random.randint(start_idx, end_idx - seq_per_img + 1)
                seq = self.h5_label_file['all_captions'][j: j + seq_per_img, :self.max_seq_length]
            caption_batch[i * seq_per_img: (i + 1) * seq_per_img, 1: self.max_seq_length + 1] = seq
            if tmp_wrapped:
                wrapped = True

            # 不知道干啥用,貌似数是用来做reward evaluation
            gts.append(self.h5_label_file['all_captions'][self.sample_start_idx[idx]: self.sample_end_idx[idx]])

            # 同时记录一些辅助信息
            info_dict = {}
            info_dict['idx'] = idx
            info_dict['id'] = self.info['images'][idx]['id']
            # info_dict['file_path'] = self.info['images'][idx]['file_path']
            infos.append(info_dict)

        # 生成mask,就是判断哪些位上是有效字符,包括<SOS>和<EOS>
        # nonzeros = np.array(map(lambda x: (x != 0).sum() + 2, caption_batch))
        nonzeros = np.array([(c != 0).sum() + 2 for c in caption_batch])
        for i, row in enumerate(mask_batch):
            row[:nonzeros[i]] = 1

        data = {}
        data['fc_feats'] = fc_batch
        data['att_feats'] = att_batch
        data['captions'] = caption_batch
        data['gts'] = gts
        data['masks'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_idx[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    def get_vocab(self):
        return self.idx2word

    def get_vocab_size(self):
        return self.vocab_size

    def get_max_seq_length(self):
        return self.max_seq_length

    def reset_iterator(self, split):
        self._prefetch_process[split].terminate()
        self._prefetch_process[split].join()
        self._prefetch_process[split] = BlobFetcher(split, self)
        self.iterators[split] = 0


class BlobFetcher():
    def __init__(self, split, dataloader):
        self.split = split
        self.split_idx_list = dataloader.split_idx[split]
        self.dataloader = dataloader

        # 进程池大小设为2，是因为我只有8个核，并且要处理三个不同的数据集合
        # 所以每个数据集的读取最多只能用2个核
        self.pool = Pool(2)
        # 任务队列
        self.fifo = []
        # 任务队列容量
        self.capacity = 512
        self.min_capacity = int(self.capacity * 0.8)

    def fill(self):
        '''
        填充任务队列
        '''
        if len(self.fifo) == 0:
            self.cur_idx = self.dataloader.iterators[self.split]
        # 获取当前数据集合对应的图片id列表
        # 把任务队列填满
        for i in range(self.capacity - len(self.fifo)):
            # cur_idx用来指示在当前数据集合中已经读取到哪张图片
            # 根据cur_idx和当前数据集合对应的图片id列表来得到图像的真正id
            idx = self.split_idx_list[self.cur_idx]
            if self.cur_idx + 1 >= len(self.split_idx_list):
                # 如果把图片遍历完一遍了，那么重头开始
                self.cur_idx = 0
            else:
                self.cur_idx += 1
            # 让读训练数据的任务异步执行，并且放到任务队列中，以便之后读取数据
            self.fifo.append(self.pool.apply_async(self._get_minibatch, (idx, )))

    def _get_minibatch(self, idx):
        wrapped = False
        if idx == self.split_idx_list[-1]:
            # wrapped指示是否到达了图片列表的尾部，用来判定epoch的结束
            wrapped = True
        return (self.dataloader.h5_fc_file['fc'][idx, :].astype('float32'),
                self.dataloader.h5_att_file['att'][idx, :, :, :].astype('float32'),
                idx,
                wrapped)

    def get(self):
        if len(self.fifo) < self.min_capacity:
            self.fill()

        idx, wrapped = self._get_next_minibatch_inds()
        tmp = self.fifo.pop(0).get()
        assert tmp[2] == idx, 'idx not equal'
        # wrapped指示是否到达了图片列表的尾部，用来判定epoch的结束
        assert tmp[3] == wrapped, 'wrapped not equal'
        return tmp

    def _get_next_minibatch_inds(self):
        '''
        本函数返回的结果用来做数据校验
        '''
        max_index = len(self.split_idx_list)
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next
        idx = self.split_idx_list[ri]
        return idx, wrapped

    def terminate(self):
        self.pool.terminate()

    def join(self):
        self.pool.join()


if __name__ == '__main__':
    loader = Dataloader(opts.parse_opt())
    data = loader.get_batch('train')
