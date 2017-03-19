import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    # 数据读取的相关参数
    parser.add_argument('--input-json', type=str, default='./data/metadata.json')
    parser.add_argument('--input-fc-h5', type=str, default='./data/h5data_fc.h5')
    parser.add_argument('--input-att-h5', type=str, default='./data/h5data_att.h5')
    parser.add_argument('--input-label-h5', type=str, default='./data/h5data_label.h5')
    parser.add_argument('--start-from', type=str, default=None)

    # 模型参数
    parser.add_argument('--caption-model', type=str, default='show_tell')
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--rnn-type', type=str, default='LSTM')
    parser.add_argument('--embedding-size', type=int, default=512)
    parser.add_argument('--fc-feat-size', type=int, default=2048)

    # 训练参数: 全局
    parser.add_argument('--max-epochs', type=int, default=-1)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--grad-clip', type=float, default=0.1)
    parser.add_argument('--drop-prob-lm', type=float, default=0.5)
    parser.add_argument('--seq-per-img', type=int, default=5)
    parser.add_argument('--beam-size', type=int, default=1)

    # 训练参数: 语言模型
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--learning-rate', type=float, default=4e-4)
    parser.add_argument('--learning-rate-decay-start', type=int, default=-1,
                        help="at what iteration to start decaying learning rate? (-1 = don't) (in epoch)")
    parser.add_argument('--learning-rate-decay-every', type=int, default=3)
    parser.add_argument('--learning-rate-decay-rate', type=float, default=0.8)
    parser.add_argument('--optim-alpha', type=float, default=0.8)
    parser.add_argument('--optim-beta', type=float, default=0.999)
    parser.add_argument('--optim-epsilon', type=float, default=1e-8)

    parser.add_argument('--scheduled-sampling-start', type=int, default=-1)
    parser.add_argument('--scheduled-sampling-increase-every', type=int, default=5)
    parser.add_argument('--scheduled-sampling-increase-prob', type=float, default=0.05)
    parser.add_argument('--scheduled-sampling-max-prob', type=float, default=0.25)

    # 评估以及保存
    parser.add_argument('--val-images-use', type=int, default=3200)
    parser.add_argument('--save-checkpoint-every', type=int, default=2500)
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints')
    parser.add_argument('--losses-log-every', type=int, default=25)
    parser.add_argument('--load-best-score', type=int, default=1)

    # 杂项
    parser.add_argument('--id', type=str, default='')
    parser.add_argument('--train-only', type=int, default=0,
                        help='if 0 then use 110k, else use 80k')
    parser.add_argument('--print-every', type=int, default=10)

    args = parser.parse_args()

    # 参数检验
    assert args.hidden_size > 0, 'rnn-size should be greater than 0'
    assert args.num_layers > 0, 'num-layers should bt greater than 0'
    assert args.embedding_size > 0, 'embedding-size should be greater than 0'
    assert args.batch_size > 0, 'batch-size should be greater than 0'

    return args
