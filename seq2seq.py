from module.config import Config
from module.model import Seq2seq
import tensorflow as tf
from module.utils.sentence_processor import SentenceProcessor
from module.utils.data_processor import DataProcessor
import json
import argparse
import os
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--trainset_path', dest='trainset_path', default='data/raw/trainset_cut300000.txt', type=str, help='训练集位置')
parser.add_argument('--validset_path', dest='validset_path', default='data/raw/validset.txt', type=str, help='验证集位置')
parser.add_argument('--testset_path', dest='testset_path', default='data/raw/testset.txt', type=str, help='测试集位置')
parser.add_argument('--embed_path', dest='embed_path', default='data/embed.txt', type=str, help='词向量位置')
parser.add_argument('--result_path', dest='result_path', default='result', type=str, help='测试结果位置')
parser.add_argument('--print_per_step', dest='print_per_step', default=100, type=int, help='每更新多少次参数summary学习情况')
parser.add_argument('--log_per_step', dest='log_per_step', default=20000, type=int, help='每更新多少次参数保存模型')
parser.add_argument('--log_path', dest='log_path', default='log', type=str, help='记录模型位置')
parser.add_argument('--inference', dest='inference', default=False, type=bool, help='是否测试')  #
parser.add_argument('--max_len', dest='max_len', default=60, type=int, help='测试时最大解码步数')
parser.add_argument('--model_path', dest='model_path', default='log//', type=str, help='载入模型位置')  #
parser.add_argument('--max_epoch', dest='max_epoch', default=60, type=int, help='最大训练epoch')

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

    model = Seq2seq(config, np.array(embeds))
    global_step = 0

    # 载入模型
    if os.path.isfile(args.model_path):
        model.load_weights(args.model_path)
        print('载入模型完成')
        log_dir, model_file = os.path.split(args.model_path)
        global_step = int(model_file[5: model_file.find('.')])
    elif args.inference:
        print('请载入一个模型进行测试')
        return
    else:
        print('初始化模型完成')
        log_dir = os.path.join(args.log_path, 'run%d' % int(time.time()))
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)

    # 创建优化器
    optimizer = tf.optimizers.Adam(config.lr)

    # 训练
    if not args.inference:

        dp_train = DataProcessor(trainset, config.batch_size, sentence_processor)
        dp_valid = DataProcessor(validset, config.batch_size, sentence_processor, shuffle=False)

        summary_writer = tf.summary.create_file_writer(log_dir)

        for epoch in range(args.max_epoch):
            for data in dp_train.get_batch_data():

                start_time = time.time()
                loss, ppl = train(model, data, optimizer)
                use_time = time.time() - start_time

                if global_step % args.print_per_step == 0:
                    print('global_step: %d, loss: %g, ppl: %g, time: %gs'
                          % (global_step, loss, ppl, use_time))
                    with summary_writer.as_default():
                        tf.summary.scalar('train_loss', loss, global_step)
                        tf.summary.scalar('train_ppl', ppl, global_step)
                        summary_writer.flush()

                global_step += 1

                if global_step % args.log_per_step == 0:
                    log_file = os.path.join(log_dir, 'model%012d.ckpt' % global_step)
                    model.save_weights(log_file)
                    ppls = []
                    for valid_data in dp_valid.get_batch_data():
                        ppl = valid(model, valid_data)
                        ppls.extend(ppl)
                    avg_ppl = np.exp(np.array(ppls).mean())
                    print('验证集上的困惑度: %g' % avg_ppl)
                    with summary_writer.as_default():
                        tf.summary.scalar('valid_ppl', avg_ppl, global_step)
                        summary_writer.flush()

            log_file = os.path.join(log_dir, 'model%012d.ckpt' % global_step)
            model.save_weights(log_file)
            ppls = []
            for valid_data in dp_valid.get_batch_data():
                ppl = valid(model, valid_data)
                ppls.extend(ppl)
            avg_ppl = np.exp(np.array(ppls).mean())
            print('验证集上的困惑度: %g' % avg_ppl)
            with summary_writer.as_default():
                tf.summary.scalar('valid_ppl', avg_ppl, global_step)
                summary_writer.flush()

    else:  # 测试

        dp_test = DataProcessor(testset, config.batch_size, sentence_processor, shuffle=False)

        ppls = []
        for test_data in dp_test.get_batch_data():
            ppl = valid(model, test_data)
            ppls.extend(ppl)
        avg_ppl = np.exp(np.array(ppls).mean())
        print('测试集上的困惑度: %g' % avg_ppl)

        result_dir = args.result_path
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_file = os.path.join(result_dir, model_file + '.txt')
        fw = open(result_file, 'w', encoding='utf8')

        for test_data in dp_test.get_batch_data():
            str_posts = test_data['str_posts']
            str_responses = test_data['str_responses']

            feed_input = {'posts': tf.convert_to_tensor(test_data['posts'], dtype=tf.int32)}
            outputs_prob = model(feed_input, inference=True, max_len=args.max_len)  # [batch, len_decoder, num_vocab]
            outputs_id = tf.argmax(outputs_prob, 2).numpy().tolist()  # [batch, len_decoder]

            for idx, result in enumerate(outputs_id):
                data = {}
                data['post'] = str_posts[idx]
                data['response'] = str_responses[idx]
                data['result'] = sentence_processor.index2word(result)
                fw.write(json.dumps(data) + '\n')

        fw.close()


def comput_losses(logits,  # [batch, len_decoder, num_vocab]
                  labels,  # [batch, len_decoder]
                  masks):  # [batch, len_decoder]

    len_decoder = tf.shape(logits)[1]
    len_masks = 1.0 * tf.reduce_sum(masks, 1)  # [batch]
    len_masks = tf.clip_by_value(len_masks, 1e-12, tf.cast(len_decoder, dtype=tf.float32))  # 防止长度为0

    logits = tf.reshape(logits, [-1, tf.shape(logits)[2]])  # [batch*len_decoder, num_vocab]
    logits = tf.clip_by_value(logits, 1e-12, 1.0)  # 防止log0
    labels = tf.reshape(labels, [-1])  # [batch*len_decoder]
    masks = tf.reshape(masks, [-1])  # [batch*len_decoder]

    losses = tf.keras.losses.sparse_categorical_crossentropy(y_pred=logits, y_true=labels)  # [batch*len_decoder]
    losses = losses * masks  # [batch*len_decoder]

    losses = tf.reshape(losses, [-1, len_decoder])  # [batch, len_decoder]
    losses = tf.reduce_sum(losses, 1)  # [batch]每个样本的损失

    ppls = losses / len_masks  # [batch]

    return losses, ppls


def train(model, data, optimizer):

    # 输入
    feed_input = {'posts': tf.convert_to_tensor(data['posts'], dtype=tf.int32),
                  'responses': tf.convert_to_tensor(data['responses'], dtype=tf.int32)}

    # 标签
    labels = tf.convert_to_tensor(data['responses'], dtype=tf.int32)[:, 1:]  # [batch, len_decoder] 去掉start_id

    # mask
    len_responses = tf.convert_to_tensor(data['len_responses'], dtype=tf.int32)
    len_labels = len_responses - 1
    id_masks = len_labels - 1
    masks = tf.cumsum(tf.one_hot(id_masks, tf.reduce_max(id_masks) + 1), 1, reverse=True)

    with tf.GradientTape() as tape:
        outputs_prob = model(feed_input)
        losses, ppls = comput_losses(outputs_prob, labels, masks)

    trainable_variables = model.trainable_variables
    gradients = tape.gradient(losses, trainable_variables)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, config.gradients_clip_norm)
    optimizer.apply_gradients(zip(clipped_gradients, trainable_variables))

    return tf.reduce_mean(losses).numpy(), tf.exp(tf.reduce_mean(ppls)).numpy()


def valid(model, data):

    # 输入
    feed_input = {'posts': tf.convert_to_tensor(data['posts'], dtype=tf.int32),
                  'responses': tf.convert_to_tensor(data['responses'], dtype=tf.int32)}

    # 标签
    labels = tf.convert_to_tensor(data['responses'], dtype=tf.int32)[:, 1:]  # [batch, len_decoder] 去掉start_id

    # mask
    len_responses = tf.convert_to_tensor(data['len_responses'], dtype=tf.int32)
    len_labels = len_responses - 1
    id_masks = len_labels - 1
    masks = tf.cumsum(tf.one_hot(id_masks, tf.reduce_max(id_masks) + 1), 1, reverse=True)

    outputs_prob = model(feed_input)
    _, ppls = comput_losses(outputs_prob, labels, masks)

    return ppls.numpy().tolist()  # [batch]


if __name__ == '__main__':
    main()
