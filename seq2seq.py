from module.config import Config
from module.model import Seq2seq
import tensorflow.keras as keras
import tensorflow as tf
from module.utils.sentence_processor import SentenceProcessor
from module.utils.data_processor import DataProcessor
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trainset_path', dest='trainset_path', default='data/raw/trainset_cut300000.txt', type=str, help='训练集位置')
parser.add_argument('--validset_path', dest='validset_path', default='data/raw/validset.txt', type=str, help='验证集位置')
parser.add_argument('--testset_path', dest='testset_path', default='data/raw/testset.txt', type=str, help='测试集位置')
parser.add_argument('--embed_path', dest='embed_path', default='data/embed.txt', type=str, help='词向量位置')
parser.add_argument('--result_path', dest='result_path', default='result', type=str, help='测试结果位置')
parser.add_argument('--print_per_step', dest='print_per_step', default=100, type=int, help='每更新多少次参数summary学习情况')
parser.add_argument('--log_per_step', dest='log_per_step', default=30000, type=int, help='每更新多少次参数保存模型')
parser.add_argument('--log_path', dest='log_path', default='log', type=str, help='记录模型位置')
parser.add_argument('--inference', dest='inference', default=False, type=bool, help='是否测试')  #
parser.add_argument('--max_len', dest='max_len', default=60, type=int, help='测试时最大解码步数')
parser.add_argument('--model_path', dest='model_path', default='log//', type=str, help='载入模型位置')  #
parser.add_argument('--gpu', dest='gpu', default=True, type=bool, help='是否使用gpu')  #
parser.add_argument('--max_epoch', dest='max_epoch', default=20, type=int, help='最大训练epoch')

args = parser.parse_args()  # 程序运行参数

config = Config()  # 模型配置

def main():

    # 载入数据集
    trainset, validset, testset = [], [], []
    if args.inference:  # 测试时只载入测试集
        with open(args.testset_path, 'r', encoding='utf8') as fr:
            for line in fr:
                testset.append(json.loads(line))
        print('载入测试集%d条' % len(testset))
    else:  # 训练时载入训练集和验证集
        with open(args.trainset_path, 'r', encoding='utf8') as fr:
            for line in fr:
                trainset.append(json.loads(line))
        print('载入训练集%d条' % len(trainset))
        with open(args.validset_path, 'r', encoding='utf8') as fr:
            for line in fr:
                validset.append(json.loads(line))
        print('载入验证集%d条' % len(validset))

    # 载入词汇表，词向量
    vocab, embeds = [], []
    with open(args.embed_path, 'r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            word = line[: line.find(' ')]
            vec = line[line.find(' ') + 1:].split()
            embed = [float(v) for v in vec]
            assert len(embed) == config.embedding_size  # 检测词向量维度
            vocab.append(word)
            embeds.append(embed)
    print('载入词汇表: %d个' % len(vocab))
    print('词向量维度: %d' % config.embedding_size)

    # 通过词汇表构建一个word2index和index2word的工具
    sentence_processor = SentenceProcessor(vocab, config.pad_id, config.start_id, config.end_id, config.unk_id)

    # 创建模型
    model = Seq2seq(config)
    epoch = 0  # 训练集迭代次数
    global_step = 0  # 参数更新次数

    # 训练
    if not args.inference:

        dp_train = DataProcessor(trainset, config.batch_size, sentence_processor)

        for data in dp_train.get_batch_data():

            feed_input = {'posts': tf.convert_to_tensor(data['posts'], dtype=tf.int64),
                          'responses': tf.convert_to_tensor(data['responses'], dtype=tf.int64)}

            outputs_prob = model(feed_input)


    else:  # 测试

        pass



if __name__ == '__main__':
    main()
