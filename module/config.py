

class Config(object):

    # 其他参数
    pad_id = 0
    start_id = 1
    end_id = 2
    unk_id = 3

    # 嵌入层参数
    num_vocab = 39000
    embedding_size = 300

    # 编解码器参数
    ende_rnn_type = 'LSTM'
    ende_num_layers = 2
    ende_output_size = 500
    encoder_bidirectional = True

    batch_size = 8
    lr = 0.0001
    gradients_clip_norm = 5

